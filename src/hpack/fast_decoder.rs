//! Zero-allocation, arena-based HPACK decoder with unsafe fast paths.
//!
//! All decoded strings within a HEADERS frame are written into a single
//! contiguous `BytesMut` buffer. After decoding, the buffer is frozen into
//! one `Bytes` and individual header name/value pairs are obtained via
//! `Bytes::slice()` — zero-copy sub-references sharing one refcount.
//!
//! **Static table entries** are materialized directly without arena writes
//! or validation — their names and values are compile-time constants.
//!
//! **Known header names** bypass `HeaderName::from_lowercase()` via a
//! length-discriminated fast match returning pre-built constants.
//!
//! **Header values** use `HeaderValue::from_maybe_shared_unchecked()` to
//! skip per-byte validation (gated behind the `fast-hpack` feature flag).

use super::header::BytesStr;
use super::{huffman, Header};
use super::{DecoderError, NeedMore};

use bytes::{Buf, Bytes, BytesMut};
use http::header::{self, HeaderName, HeaderValue};
use http::{Method, StatusCode};

use std::collections::VecDeque;
use std::io::Cursor;

// ===== BytesMut Arena =====

/// A write buffer that collects all decoded strings, then freezes into
/// a single `Bytes` for zero-copy slicing.
struct WriteArena {
    buf: BytesMut,
}

impl WriteArena {
    fn new() -> Self {
        WriteArena {
            buf: BytesMut::with_capacity(4096),
        }
    }

    /// Append raw bytes. Returns (offset, len).
    #[inline]
    fn write(&mut self, data: &[u8]) -> (u32, u16) {
        let offset = self.buf.len() as u32;
        self.buf.extend_from_slice(data);
        (offset, data.len() as u16)
    }

    /// Reserve space for Huffman output, returning start offset and mutable slice.
    #[inline]
    fn reserve_mut(&mut self, max_len: usize) -> (u32, usize) {
        let start = self.buf.len();
        self.buf.resize(start + max_len, 0);
        (start as u32, start)
    }

    /// After Huffman decode, truncate to the actual output length.
    #[inline]
    fn truncate_to(&mut self, new_len: usize) {
        self.buf.truncate(new_len);
    }

    /// Get a read-only slice (used for dynamic table insertion before freeze).
    #[inline]
    fn slice_ref(&self, offset: u32, len: u16) -> &[u8] {
        &self.buf[offset as usize..offset as usize + len as usize]
    }

    /// Freeze into a single Bytes.
    #[inline]
    fn freeze(self) -> Bytes {
        self.buf.freeze()
    }

    /// Get a mutable reference to the underlying buffer for Huffman decode.
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buf[..]
    }
}

// ===== DecodedHeader =====

/// A decoded header reference — determines the materialization fast path.
#[derive(Clone, Copy)]
enum DecodedHeader {
    /// Fully indexed from static table (index 1-61).
    /// Materialized directly — no arena data, no validation.
    StaticFull(u8),
    /// Name from static table index, value in arena.
    /// Uses pre-built HeaderName constant + unsafe HeaderValue.
    StaticName {
        static_idx: u8,
        value_offset: u32,
        value_len: u16,
    },
    /// Name and value both in arena (dynamic table ref or new literal).
    /// Uses known-header fast match + unsafe HeaderValue.
    Arena {
        name_offset: u32,
        name_len: u16,
        value_offset: u32,
        value_len: u16,
    },
}

impl DecodedHeader {
    #[inline]
    fn materialize(&self, frozen: &Bytes) -> Result<Header, DecoderError> {
        match *self {
            DecodedHeader::StaticFull(idx) => Ok(materialize_static_full(idx)),
            DecodedHeader::StaticName {
                static_idx,
                value_offset,
                value_len,
            } => {
                let value = frozen.slice(
                    value_offset as usize..value_offset as usize + value_len as usize,
                );
                materialize_static_name(static_idx, value)
            }
            DecodedHeader::Arena {
                name_offset,
                name_len,
                value_offset,
                value_len,
            } => {
                let name = frozen.slice(
                    name_offset as usize..name_offset as usize + name_len as usize,
                );
                let value = frozen.slice(
                    value_offset as usize..value_offset as usize + value_len as usize,
                );
                materialize_field(name, value)
            }
        }
    }
}

/// Helper for tracking name source in decode_literal_fast.
enum NameSource {
    Static(u8),
    Arena { offset: u32, len: u16 },
}

// ===== DynTable =====

/// Dynamic table entry referencing persistent storage.
#[derive(Clone, Copy)]
struct DynEntry {
    name_start: u32,
    name_len: u16,
    value_start: u32,
    value_len: u16,
    entry_size: usize,
}

/// Ring-buffer dynamic table storing raw bytes — no Header cloning.
struct DynTable {
    storage: Vec<u8>,
    storage_pos: usize,
    entries: VecDeque<DynEntry>,
    size: usize,
    max_size: usize,
}

impl DynTable {
    fn new(max_size: usize) -> Self {
        DynTable {
            storage: Vec::with_capacity(max_size.min(16384)),
            storage_pos: 0,
            entries: VecDeque::with_capacity(64),
            size: 0,
            max_size,
        }
    }

    fn get(&self, index: usize) -> Option<&DynEntry> {
        self.entries.get(index)
    }

    fn insert_raw(&mut self, name: &[u8], value: &[u8]) {
        let entry_size = name.len() + value.len() + 32;
        self.reserve(entry_size);
        if self.size + entry_size > self.max_size {
            return;
        }

        let total = name.len() + value.len();
        if self.storage_pos + total > self.storage.len() {
            self.storage.resize(self.storage_pos + total, 0);
        }

        let name_start = self.storage_pos as u32;
        self.storage[self.storage_pos..self.storage_pos + name.len()].copy_from_slice(name);
        self.storage_pos += name.len();

        let value_start = self.storage_pos as u32;
        self.storage[self.storage_pos..self.storage_pos + value.len()].copy_from_slice(value);
        self.storage_pos += value.len();

        self.entries.push_front(DynEntry {
            name_start,
            name_len: name.len() as u16,
            value_start,
            value_len: value.len() as u16,
            entry_size,
        });
        self.size += entry_size;
    }

    /// Copy a dynamic table entry's name and value into the arena.
    fn copy_to_arena(&self, entry: &DynEntry, arena: &mut WriteArena) -> DecodedHeader {
        let name = &self.storage
            [entry.name_start as usize..entry.name_start as usize + entry.name_len as usize];
        let value = &self.storage
            [entry.value_start as usize..entry.value_start as usize + entry.value_len as usize];
        let (name_offset, name_len) = arena.write(name);
        let (value_offset, value_len) = arena.write(value);
        DecodedHeader::Arena {
            name_offset,
            name_len,
            value_offset,
            value_len,
        }
    }

    fn name_slice(&self, entry: &DynEntry) -> &[u8] {
        &self.storage
            [entry.name_start as usize..entry.name_start as usize + entry.name_len as usize]
    }

    fn reserve(&mut self, size: usize) {
        while self.size + size > self.max_size {
            match self.entries.pop_back() {
                Some(last) => self.size -= last.entry_size,
                None => return,
            }
        }
    }

    fn set_max_size(&mut self, size: usize) {
        self.max_size = size;
        while self.size > self.max_size {
            match self.entries.pop_back() {
                Some(last) => self.size -= last.entry_size,
                None => panic!("Size of table != 0, but no headers left!"),
            }
        }
    }

    fn size(&self) -> usize {
        self.size
    }
}

// ===== Static Table =====

static STATIC_TABLE: [(&[u8], &[u8]); 62] = [
    (b"", b""),                                // 0: unused
    (b":authority", b""),                       // 1
    (b":method", b"GET"),                       // 2
    (b":method", b"POST"),                      // 3
    (b":path", b"/"),                           // 4
    (b":path", b"/index.html"),                 // 5
    (b":scheme", b"http"),                      // 6
    (b":scheme", b"https"),                     // 7
    (b":status", b"200"),                       // 8
    (b":status", b"204"),                       // 9
    (b":status", b"206"),                       // 10
    (b":status", b"304"),                       // 11
    (b":status", b"400"),                       // 12
    (b":status", b"404"),                       // 13
    (b":status", b"500"),                       // 14
    (b"accept-charset", b""),                   // 15
    (b"accept-encoding", b"gzip, deflate"),     // 16
    (b"accept-language", b""),                  // 17
    (b"accept-ranges", b""),                    // 18
    (b"accept", b""),                           // 19
    (b"access-control-allow-origin", b""),      // 20
    (b"age", b""),                              // 21
    (b"allow", b""),                            // 22
    (b"authorization", b""),                    // 23
    (b"cache-control", b""),                    // 24
    (b"content-disposition", b""),              // 25
    (b"content-encoding", b""),                 // 26
    (b"content-language", b""),                 // 27
    (b"content-length", b""),                   // 28
    (b"content-location", b""),                 // 29
    (b"content-range", b""),                    // 30
    (b"content-type", b""),                     // 31
    (b"cookie", b""),                           // 32
    (b"date", b""),                             // 33
    (b"etag", b""),                             // 34
    (b"expect", b""),                           // 35
    (b"expires", b""),                          // 36
    (b"from", b""),                             // 37
    (b"host", b""),                             // 38
    (b"if-match", b""),                         // 39
    (b"if-modified-since", b""),                // 40
    (b"if-none-match", b""),                    // 41
    (b"if-range", b""),                         // 42
    (b"if-unmodified-since", b""),              // 43
    (b"last-modified", b""),                    // 44
    (b"link", b""),                             // 45
    (b"location", b""),                         // 46
    (b"max-forwards", b""),                     // 47
    (b"proxy-authenticate", b""),               // 48
    (b"proxy-authorization", b""),              // 49
    (b"range", b""),                            // 50
    (b"referer", b""),                          // 51
    (b"refresh", b""),                          // 52
    (b"retry-after", b""),                      // 53
    (b"server", b""),                           // 54
    (b"set-cookie", b""),                       // 55
    (b"strict-transport-security", b""),        // 56
    (b"transfer-encoding", b""),                // 57
    (b"user-agent", b""),                       // 58
    (b"vary", b""),                             // 59
    (b"via", b""),                              // 60
    (b"www-authenticate", b""),                 // 61
];

// ===== Optimized Integer Decoding =====

#[inline(always)]
fn decode_int_fast(src: &[u8], pos: &mut usize, prefix_size: u8) -> Result<usize, DecoderError> {
    if *pos >= src.len() {
        return Err(DecoderError::NeedMore(NeedMore::IntegerUnderflow));
    }

    let mask = if prefix_size == 8 {
        0xFFu8
    } else {
        (1u8 << prefix_size).wrapping_sub(1)
    };

    let val = src[*pos] & mask;
    *pos += 1;

    if val < mask {
        return Ok(val as usize);
    }

    decode_int_slow(src, pos, val as usize)
}

#[cold]
#[inline(never)]
fn decode_int_slow(src: &[u8], pos: &mut usize, mut ret: usize) -> Result<usize, DecoderError> {
    const MAX_BYTES: usize = 5;
    let mut shift = 0u32;
    for _ in 1..MAX_BYTES {
        if *pos >= src.len() {
            return Err(DecoderError::NeedMore(NeedMore::IntegerUnderflow));
        }
        let b = src[*pos];
        *pos += 1;
        ret += ((b & 0x7F) as usize) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            return Ok(ret);
        }
    }
    Err(DecoderError::IntegerOverflow)
}

// ===== Huffman decode into arena =====

#[inline]
fn huffman_decode_to_arena(
    src: &[u8],
    arena: &mut WriteArena,
) -> Result<(u32, u16), DecoderError> {
    let max_out = src.len() * 2;
    let (start_offset, start_idx) = arena.reserve_mut(max_out);
    let actual_len = huffman::decode_to_slice(src, &mut arena.as_mut_slice()[start_idx..])?;
    arena.truncate_to(start_idx + actual_len);
    Ok((start_offset, actual_len as u16))
}

// ===== Materialization Functions =====

/// Materialize a fully-indexed static table entry directly.
/// No arena access, no validation — compile-time known values.
#[inline]
fn materialize_static_full(idx: u8) -> Header {
    match idx {
        1 => Header::Authority(BytesStr::from_static("")),
        2 => Header::Method(Method::GET),
        3 => Header::Method(Method::POST),
        4 => Header::Path(BytesStr::from_static("/")),
        5 => Header::Path(BytesStr::from_static("/index.html")),
        6 => Header::Scheme(BytesStr::from_static("http")),
        7 => Header::Scheme(BytesStr::from_static("https")),
        8 => Header::Status(StatusCode::OK),
        9 => Header::Status(StatusCode::NO_CONTENT),
        10 => Header::Status(StatusCode::PARTIAL_CONTENT),
        11 => Header::Status(StatusCode::NOT_MODIFIED),
        12 => Header::Status(StatusCode::BAD_REQUEST),
        13 => Header::Status(StatusCode::NOT_FOUND),
        14 => Header::Status(StatusCode::INTERNAL_SERVER_ERROR),
        idx @ 15..=61 => Header::Field {
            name: static_idx_to_header_name(idx),
            value: HeaderValue::from_static(""),
        },
        _ => unreachable!(),
    }
}

/// Map static table index (15-61) to pre-built HeaderName constant.
#[inline]
fn static_idx_to_header_name(idx: u8) -> HeaderName {
    match idx {
        15 => header::ACCEPT_CHARSET,
        16 => header::ACCEPT_ENCODING,
        17 => header::ACCEPT_LANGUAGE,
        18 => header::ACCEPT_RANGES,
        19 => header::ACCEPT,
        20 => header::ACCESS_CONTROL_ALLOW_ORIGIN,
        21 => header::AGE,
        22 => header::ALLOW,
        23 => header::AUTHORIZATION,
        24 => header::CACHE_CONTROL,
        25 => header::CONTENT_DISPOSITION,
        26 => header::CONTENT_ENCODING,
        27 => header::CONTENT_LANGUAGE,
        28 => header::CONTENT_LENGTH,
        29 => header::CONTENT_LOCATION,
        30 => header::CONTENT_RANGE,
        31 => header::CONTENT_TYPE,
        32 => header::COOKIE,
        33 => header::DATE,
        34 => header::ETAG,
        35 => header::EXPECT,
        36 => header::EXPIRES,
        37 => header::FROM,
        38 => header::HOST,
        39 => header::IF_MATCH,
        40 => header::IF_MODIFIED_SINCE,
        41 => header::IF_NONE_MATCH,
        42 => header::IF_RANGE,
        43 => header::IF_UNMODIFIED_SINCE,
        44 => header::LAST_MODIFIED,
        45 => header::LINK,
        46 => header::LOCATION,
        47 => header::MAX_FORWARDS,
        48 => header::PROXY_AUTHENTICATE,
        49 => header::PROXY_AUTHORIZATION,
        50 => header::RANGE,
        51 => header::REFERER,
        52 => HeaderName::from_static("refresh"),
        53 => header::RETRY_AFTER,
        54 => header::SERVER,
        55 => header::SET_COOKIE,
        56 => header::STRICT_TRANSPORT_SECURITY,
        57 => header::TRANSFER_ENCODING,
        58 => header::USER_AGENT,
        59 => header::VARY,
        60 => header::VIA,
        61 => header::WWW_AUTHENTICATE,
        _ => unreachable!(),
    }
}

/// Materialize a header with a static table name and an arena value.
/// Pseudo-headers (1-14) use safe construction; field headers (15-61)
/// use the pre-built HeaderName constant + zero-copy HeaderValue.
#[inline]
fn materialize_static_name(idx: u8, value: Bytes) -> Result<Header, DecoderError> {
    match idx {
        1 => Ok(Header::Authority(BytesStr::try_from(value)?)),
        2 | 3 => Ok(Header::Method(Method::from_bytes(&value)?)),
        4 | 5 => Ok(Header::Path(BytesStr::try_from(value)?)),
        6 | 7 => Ok(Header::Scheme(BytesStr::try_from(value)?)),
        8..=14 => match StatusCode::from_bytes(&value) {
            Ok(status) => Ok(Header::Status(status)),
            Err(_) => Err(DecoderError::InvalidStatusCode),
        },
        idx @ 15..=61 => {
            let name = static_idx_to_header_name(idx);
            // Validated + zero-copy: takes Bytes ownership, no allocation
            let value = HeaderValue::from_maybe_shared(value)?;
            Ok(Header::Field { name, value })
        }
        _ => Err(DecoderError::InvalidTableIndex),
    }
}

/// Materialize a header where name and value are both arena bytes.
/// Used for dynamic table references and new literal names.
/// Pseudo-headers fall back to the safe `Header::new()` path.
/// Regular headers use known-header fast match + zero-copy HeaderValue.
#[inline]
fn materialize_field(name: Bytes, value: Bytes) -> Result<Header, DecoderError> {
    if name.is_empty() {
        return Err(DecoderError::NeedMore(NeedMore::UnexpectedEndOfStream));
    }
    if name[0] == b':' {
        // Pseudo-header from dynamic table — use safe construction
        return Header::new(name, value);
    }
    // Fast match known header names to pre-built constants
    let header_name = match match_known_header(&name) {
        Some(known) => known,
        None => HeaderName::from_lowercase(&name)?,
    };
    // Validated + zero-copy: takes Bytes ownership, no allocation
    let value = HeaderValue::from_maybe_shared(value)?;
    Ok(Header::Field {
        name: header_name,
        value,
    })
}

// ===== Known Header Fast Match =====

/// Fast-path: map well-known header name bytes to pre-built HeaderName constants.
/// Discriminates by length, then compares bytes. Returns None for unknown names.
/// Covers all HPACK static table headers (indices 15-61) plus common extras.
#[inline]
fn match_known_header(name: &[u8]) -> Option<HeaderName> {
    match name.len() {
        3 => {
            if name == b"age" { return Some(header::AGE); }
            if name == b"via" { return Some(header::VIA); }
        }
        4 => {
            if name == b"date" { return Some(header::DATE); }
            if name == b"etag" { return Some(header::ETAG); }
            if name == b"from" { return Some(header::FROM); }
            if name == b"host" { return Some(header::HOST); }
            if name == b"link" { return Some(header::LINK); }
            if name == b"vary" { return Some(header::VARY); }
        }
        5 => {
            if name == b"allow" { return Some(header::ALLOW); }
            if name == b"range" { return Some(header::RANGE); }
        }
        6 => {
            if name == b"accept" { return Some(header::ACCEPT); }
            if name == b"cookie" { return Some(header::COOKIE); }
            if name == b"expect" { return Some(header::EXPECT); }
            if name == b"server" { return Some(header::SERVER); }
        }
        7 => {
            if name == b"expires" { return Some(header::EXPIRES); }
            if name == b"referer" { return Some(header::REFERER); }
        }
        8 => {
            if name == b"if-match" { return Some(header::IF_MATCH); }
            if name == b"if-range" { return Some(header::IF_RANGE); }
            if name == b"location" { return Some(header::LOCATION); }
        }
        10 => {
            if name == b"set-cookie" { return Some(header::SET_COOKIE); }
            if name == b"user-agent" { return Some(header::USER_AGENT); }
        }
        11 => {
            if name == b"retry-after" { return Some(header::RETRY_AFTER); }
        }
        12 => {
            if name == b"content-type" { return Some(header::CONTENT_TYPE); }
            if name == b"max-forwards" { return Some(header::MAX_FORWARDS); }
        }
        13 => {
            if name == b"accept-ranges" { return Some(header::ACCEPT_RANGES); }
            if name == b"authorization" { return Some(header::AUTHORIZATION); }
            if name == b"cache-control" { return Some(header::CACHE_CONTROL); }
            if name == b"content-range" { return Some(header::CONTENT_RANGE); }
            if name == b"if-none-match" { return Some(header::IF_NONE_MATCH); }
            if name == b"last-modified" { return Some(header::LAST_MODIFIED); }
        }
        14 => {
            if name == b"accept-charset" { return Some(header::ACCEPT_CHARSET); }
            if name == b"content-length" { return Some(header::CONTENT_LENGTH); }
        }
        15 => {
            if name == b"accept-encoding" { return Some(header::ACCEPT_ENCODING); }
            if name == b"accept-language" { return Some(header::ACCEPT_LANGUAGE); }
        }
        16 => {
            if name == b"content-encoding" { return Some(header::CONTENT_ENCODING); }
            if name == b"content-language" { return Some(header::CONTENT_LANGUAGE); }
            if name == b"content-location" { return Some(header::CONTENT_LOCATION); }
            if name == b"www-authenticate" { return Some(header::WWW_AUTHENTICATE); }
        }
        17 => {
            if name == b"if-modified-since" { return Some(header::IF_MODIFIED_SINCE); }
            if name == b"transfer-encoding" { return Some(header::TRANSFER_ENCODING); }
        }
        18 => {
            if name == b"proxy-authenticate" { return Some(header::PROXY_AUTHENTICATE); }
        }
        19 => {
            if name == b"content-disposition" { return Some(header::CONTENT_DISPOSITION); }
            if name == b"if-unmodified-since" { return Some(header::IF_UNMODIFIED_SINCE); }
            if name == b"proxy-authorization" { return Some(header::PROXY_AUTHORIZATION); }
        }
        25 => {
            if name == b"strict-transport-security" {
                return Some(header::STRICT_TRANSPORT_SECURITY);
            }
        }
        27 => {
            if name == b"access-control-allow-origin" {
                return Some(header::ACCESS_CONTROL_ALLOW_ORIGIN);
            }
        }
        _ => {}
    }
    None
}

// ===== FastDecoder =====

/// Zero-allocation HPACK decoder with unsafe fast paths.
///
/// Static table entries are materialized directly (no arena, no validation).
/// Known header names bypass `HeaderName::from_lowercase()`.
/// Header values skip per-byte validation via `from_maybe_shared_unchecked`.
pub struct FastDecoder {
    max_size_update: Option<usize>,
    last_max_update: usize,
    dyn_table: DynTable,
    /// Reusable decoded header output vec
    decoded_headers: Vec<DecodedHeader>,
}

impl FastDecoder {
    pub fn new(size: usize) -> Self {
        FastDecoder {
            max_size_update: None,
            last_max_update: size,
            dyn_table: DynTable::new(size),
            decoded_headers: Vec::with_capacity(32),
        }
    }

    #[allow(dead_code)]
    pub fn queue_size_update(&mut self, size: usize) {
        let size = match self.max_size_update {
            Some(v) => std::cmp::max(v, size),
            None => size,
        };
        self.max_size_update = Some(size);
    }

    pub fn decode<F>(
        &mut self,
        src: &mut Cursor<&mut BytesMut>,
        mut f: F,
    ) -> Result<(), DecoderError>
    where
        F: FnMut(Header),
    {
        let mut can_resize = true;

        if let Some(size) = self.max_size_update.take() {
            self.last_max_update = size;
        }

        let data = &src.chunk()[..src.remaining()];
        let data_len = data.len();

        // Fresh arena per frame
        let mut arena = WriteArena::new();
        self.decoded_headers.clear();

        let mut pos = 0;
        while pos < data.len() {
            let byte = data[pos];

            if byte & 0x80 != 0 {
                can_resize = false;
                let index = decode_int_fast(data, &mut pos, 7)?;
                let decoded = self.get_indexed(index, &mut arena)?;
                self.decoded_headers.push(decoded);
            } else if byte & 0x40 != 0 {
                can_resize = false;
                let decoded =
                    self.decode_literal_fast(data, &mut pos, 6, true, &mut arena)?;
                self.decoded_headers.push(decoded);
            } else if byte & 0xE0 == 0x20 {
                if !can_resize {
                    return Err(DecoderError::InvalidMaxDynamicSize);
                }
                let new_size = decode_int_fast(data, &mut pos, 5)?;
                if new_size > self.last_max_update {
                    return Err(DecoderError::InvalidMaxDynamicSize);
                }
                self.dyn_table.set_max_size(new_size);
            } else {
                can_resize = false;
                let decoded =
                    self.decode_literal_fast(data, &mut pos, 4, false, &mut arena)?;
                self.decoded_headers.push(decoded);
            }
        }

        // Freeze the arena into a single Bytes — one allocation shared by all headers
        let frozen = arena.freeze();

        // Materialize all headers via fast paths
        for decoded in &self.decoded_headers {
            let header = decoded.materialize(&frozen)?;
            f(header);
        }

        // Advance cursor and consume from BytesMut
        src.advance(data_len);
        let cursor_pos = src.position() as usize;
        let _ = src.get_mut().split_to(cursor_pos);
        src.set_position(0);

        Ok(())
    }

    #[inline]
    fn get_indexed(
        &mut self,
        index: usize,
        arena: &mut WriteArena,
    ) -> Result<DecodedHeader, DecoderError> {
        if index == 0 {
            return Err(DecoderError::InvalidTableIndex);
        }

        if index <= 61 {
            // Static table — direct materialization, no arena writes
            return Ok(DecodedHeader::StaticFull(index as u8));
        }

        let dyn_idx = index - 62;
        match self.dyn_table.get(dyn_idx) {
            Some(entry) => {
                let entry = *entry;
                Ok(self.dyn_table.copy_to_arena(&entry, arena))
            }
            None => Err(DecoderError::InvalidTableIndex),
        }
    }

    fn decode_literal_fast(
        &mut self,
        data: &[u8],
        pos: &mut usize,
        prefix: u8,
        index: bool,
        arena: &mut WriteArena,
    ) -> Result<DecodedHeader, DecoderError> {
        let table_idx = decode_int_fast(data, pos, prefix)?;

        let name_source = if table_idx == 0 {
            // New literal name — decode from wire into arena
            let (offset, len) = self.decode_string_fast(data, pos, arena)?;
            NameSource::Arena { offset, len }
        } else if table_idx <= 61 {
            // Name from static table — no arena write needed
            NameSource::Static(table_idx as u8)
        } else {
            // Name from dynamic table — copy to arena
            let dyn_idx = table_idx - 62;
            match self.dyn_table.get(dyn_idx) {
                Some(entry) => {
                    let name = self.dyn_table.name_slice(entry);
                    let (offset, len) = arena.write(name);
                    NameSource::Arena { offset, len }
                }
                None => return Err(DecoderError::InvalidTableIndex),
            }
        };

        // Decode value string
        let (value_offset, value_len) = self.decode_string_fast(data, pos, arena)?;

        // Insert into dynamic table if needed
        if index {
            let name_bytes: &[u8] = match name_source {
                NameSource::Static(idx) => STATIC_TABLE[idx as usize].0,
                NameSource::Arena { offset, len } => arena.slice_ref(offset, len),
            };
            let value_bytes = arena.slice_ref(value_offset, value_len);
            self.dyn_table.insert_raw(name_bytes, value_bytes);
        }

        // Build the appropriate DecodedHeader variant
        match name_source {
            NameSource::Static(idx) => Ok(DecodedHeader::StaticName {
                static_idx: idx,
                value_offset,
                value_len,
            }),
            NameSource::Arena { offset, len } => Ok(DecodedHeader::Arena {
                name_offset: offset,
                name_len: len,
                value_offset,
                value_len,
            }),
        }
    }

    fn decode_string_fast(
        &mut self,
        data: &[u8],
        pos: &mut usize,
        arena: &mut WriteArena,
    ) -> Result<(u32, u16), DecoderError> {
        if *pos >= data.len() {
            return Err(DecoderError::NeedMore(NeedMore::UnexpectedEndOfStream));
        }

        const HUFF_FLAG: u8 = 0b1000_0000;
        let huff = (data[*pos] & HUFF_FLAG) == HUFF_FLAG;
        let len = decode_int_fast(data, pos, 7)?;

        if *pos + len > data.len() {
            return Err(DecoderError::NeedMore(NeedMore::StringUnderflow));
        }

        let raw = &data[*pos..*pos + len];
        *pos += len;

        if huff {
            huffman_decode_to_arena(raw, arena)
        } else {
            Ok(arena.write(raw))
        }
    }
}

impl Default for FastDecoder {
    fn default() -> Self {
        FastDecoder::new(4096)
    }
}

impl std::fmt::Debug for FastDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastDecoder")
            .field("dyn_table_size", &self.dyn_table.size())
            .field("dyn_table_max", &self.dyn_table.max_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::header;
    use http::Method;

    #[test]
    fn test_decode_int_fast_prefix() {
        let data = [0x05];
        let mut pos = 0;
        assert_eq!(decode_int_fast(&data, &mut pos, 7).unwrap(), 5);
        assert_eq!(pos, 1);
    }

    #[test]
    fn test_decode_int_fast_multi_byte() {
        let data = [0x7F, 0x00];
        let mut pos = 0;
        assert_eq!(decode_int_fast(&data, &mut pos, 7).unwrap(), 127);
        assert_eq!(pos, 2);
    }

    #[test]
    fn test_dyn_table_insert_and_get() {
        let mut table = DynTable::new(4096);
        table.insert_raw(b"content-type", b"application/json");
        assert_eq!(table.entries.len(), 1);
        let entry = table.get(0).unwrap();
        assert_eq!(entry.name_len, 12);
        assert_eq!(entry.value_len, 16);
    }

    #[test]
    fn test_static_table_entries() {
        assert_eq!(STATIC_TABLE[1].0, b":authority");
        assert_eq!(STATIC_TABLE[2].0, b":method");
        assert_eq!(STATIC_TABLE[2].1, b"GET");
        assert_eq!(STATIC_TABLE[4].0, b":path");
        assert_eq!(STATIC_TABLE[4].1, b"/");
        assert_eq!(STATIC_TABLE[61].0, b"www-authenticate");
    }

    #[test]
    fn test_fast_decoder_indexed_static() {
        let mut decoder = FastDecoder::new(4096);
        let mut buf = BytesMut::from(&[0x82u8][..]);
        let mut headers = vec![];
        decoder
            .decode(&mut Cursor::new(&mut buf), |h| headers.push(h))
            .unwrap();
        assert_eq!(headers.len(), 1);
        match &headers[0] {
            Header::Method(m) => assert_eq!(*m, Method::GET),
            other => panic!("Expected Method, got {:?}", other),
        }
    }

    #[test]
    fn test_fast_decoder_literal_with_indexing() {
        let mut decoder = FastDecoder::new(4096);
        let mut buf = BytesMut::new();
        buf.extend_from_slice(&[0x40]);
        buf.extend_from_slice(&[0x03]);
        buf.extend_from_slice(b"foo");
        buf.extend_from_slice(&[0x03]);
        buf.extend_from_slice(b"bar");

        let mut headers = vec![];
        decoder
            .decode(&mut Cursor::new(&mut buf), |h| headers.push(h))
            .unwrap();
        assert_eq!(headers.len(), 1);
        match &headers[0] {
            Header::Field { name, value } => {
                assert_eq!(name.as_str(), "foo");
                assert_eq!(value.as_bytes(), b"bar");
            }
            other => panic!("Expected Field, got {:?}", other),
        }
        assert_eq!(decoder.dyn_table.entries.len(), 1);
    }

    #[test]
    fn test_fast_decoder_literal_indexed_name() {
        let mut decoder = FastDecoder::new(4096);
        let mut buf = BytesMut::new();
        buf.extend_from_slice(&[0x5F]); // Literal with indexing, name index 31 (content-type)
        buf.extend_from_slice(&[0x0A]);
        buf.extend_from_slice(b"text/plain");

        let mut headers = vec![];
        decoder
            .decode(&mut Cursor::new(&mut buf), |h| headers.push(h))
            .unwrap();
        assert_eq!(headers.len(), 1);
        match &headers[0] {
            Header::Field { name, value } => {
                assert_eq!(name, &header::CONTENT_TYPE);
                assert_eq!(value.as_bytes(), b"text/plain");
            }
            other => panic!("Expected Field, got {:?}", other),
        }
    }

    #[test]
    fn test_fast_decoder_multiple_headers() {
        let mut decoder = FastDecoder::new(4096);
        let mut buf = BytesMut::new();
        buf.extend_from_slice(&[0x82]); // :method GET
        buf.extend_from_slice(&[0x84]); // :path /
        buf.extend_from_slice(&[0x86]); // :scheme http

        let mut headers = vec![];
        decoder
            .decode(&mut Cursor::new(&mut buf), |h| headers.push(h))
            .unwrap();
        assert_eq!(headers.len(), 3);
        assert!(matches!(&headers[0], Header::Method(m) if *m == Method::GET));
        assert!(matches!(&headers[1], Header::Path(p) if p.as_str() == "/"));
        assert!(matches!(&headers[2], Header::Scheme(s) if s.as_str() == "http"));
    }

    #[test]
    fn test_materialize_static_full() {
        // Verify all static entries can be materialized
        for idx in 1..=61u8 {
            let header = materialize_static_full(idx);
            // Just verify it doesn't panic
            let _ = header.len();
        }
    }

    #[test]
    fn test_match_known_header() {
        assert_eq!(match_known_header(b"content-type"), Some(header::CONTENT_TYPE));
        assert_eq!(match_known_header(b"host"), Some(header::HOST));
        assert_eq!(match_known_header(b"user-agent"), Some(header::USER_AGENT));
        assert_eq!(match_known_header(b"accept-encoding"), Some(header::ACCEPT_ENCODING));
        assert_eq!(match_known_header(b"x-custom"), None);
        assert_eq!(match_known_header(b""), None);
    }
}

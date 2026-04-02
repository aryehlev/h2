mod table;

use self::table::{DECODE_TABLE, ENCODE_TABLE};
use crate::hpack::DecoderError;

use bytes::{BufMut, BytesMut};

// ===== 10-bit peekahead decode table =====
//
// For each possible 10-bit prefix of the bitstream, stores:
//   (symbol: u8, bits_consumed: u8)
// where bits_consumed > 0 means "this is a complete symbol of that length".
// bits_consumed == 0 means the code is longer than 10 bits → fall back to
// the 4-bit state machine for this and all subsequent symbols.
//
// All HPACK Huffman codes ≤10 bits cover virtually every HTTP/2 header
// byte (all printable ASCII with code lengths 5–10).  The only common
// exception is DEL (127) and a handful of high-byte symbols that have
// codes of 19–30 bits — they fall through to the scalar path.
//
// The table is generated at compile time from ENCODE_TABLE so there is
// no runtime cost and the 2 KB footprint fits inside one L1 cache set.

#[cfg(feature = "fast-hpack")]
const PEEK_BITS: u32 = 10;
#[cfg(feature = "fast-hpack")]
const PEEK_SIZE: usize = 1 << PEEK_BITS; // 1024

#[cfg(feature = "fast-hpack")]
const PEEK_TABLE: [(u8, u8); PEEK_SIZE] = build_peek_table();

#[cfg(feature = "fast-hpack")]
const fn build_peek_table() -> [(u8, u8); PEEK_SIZE] {
    let mut table = [(0u8, 0u8); PEEK_SIZE];
    let mut sym = 0usize;
    while sym < 256 {
        let (nbits, code) = ENCODE_TABLE[sym];
        // code is right-aligned; codes longer than PEEK_BITS can't be fast-pathed
        if nbits > 0 && nbits <= PEEK_BITS as usize {
            // Fill every table slot whose top `nbits` bits match `code`.
            let prefix = (code as usize) << (PEEK_BITS as usize - nbits);
            let count = 1usize << (PEEK_BITS as usize - nbits);
            let mut j = 0usize;
            while j < count {
                table[prefix + j] = (sym as u8, nbits as u8);
                j += 1;
            }
        }
        sym += 1;
    }
    table
}

// Constructed in the generated `table.rs` file
struct Decoder {
    state: u8,
    maybe_eos: bool,
}

// These flags must match the ones in genhuff.rs

const MAYBE_EOS: u8 = 1;
const DECODED: u8 = 2;
const ERROR: u8 = 4;

pub fn decode(src: &[u8], buf: &mut BytesMut) -> Result<BytesMut, DecoderError> {
    let mut decoder = Decoder::new();

    // Max compression ratio is >= 0.5
    buf.reserve(src.len() << 1);

    for b in src {
        if let Some(b) = decoder.decode4(b >> 4)? {
            buf.put_u8(b);
        }

        if let Some(b) = decoder.decode4(b & 0xf)? {
            buf.put_u8(b);
        }
    }

    if !decoder.is_final() {
        return Err(DecoderError::InvalidHuffmanCode);
    }

    Ok(buf.split())
}

pub fn encode(src: &[u8], dst: &mut BytesMut) {
    let mut bits: u64 = 0;
    let mut bits_left = 40;

    for &b in src {
        let (nbits, code) = ENCODE_TABLE[b as usize];

        bits |= code << (bits_left - nbits);
        bits_left -= nbits;

        while bits_left <= 32 {
            dst.put_u8((bits >> 32) as u8);

            bits <<= 8;
            bits_left += 8;
        }
    }

    if bits_left != 40 {
        // This writes the EOS token
        bits |= (1 << bits_left) - 1;
        dst.put_u8((bits >> 32) as u8);
    }
}

impl Decoder {
    fn new() -> Decoder {
        Decoder {
            state: 0,
            maybe_eos: false,
        }
    }

    // Decodes 4 bits
    fn decode4(&mut self, input: u8) -> Result<Option<u8>, DecoderError> {
        // (next-state, byte, flags)
        let (next, byte, flags) = DECODE_TABLE[self.state as usize][input as usize];

        if flags & ERROR == ERROR {
            // Data followed the EOS marker
            return Err(DecoderError::InvalidHuffmanCode);
        }

        let mut ret = None;

        if flags & DECODED == DECODED {
            ret = Some(byte);
        }

        self.state = next;
        self.maybe_eos = flags & MAYBE_EOS == MAYBE_EOS;

        Ok(ret)
    }

    fn is_final(&self) -> bool {
        self.state == 0 || self.maybe_eos
    }
}

/// Fast Huffman decode directly into a pre-allocated byte slice.
/// Returns the number of bytes written to `dst`.
/// `dst` must be at least `src.len() * 2` bytes long.
///
/// Uses a 10-bit peekahead table for the common fast path (all HPACK codes
/// ≤10 bits, covering ~100% of HTTP/2 header traffic).  Codes longer than
/// 10 bits fall back to the 4-bit scalar state machine.
///
/// Inner loop is unsafe to eliminate bounds-checks on the pre-allocated dst.
#[cfg(feature = "fast-hpack")]
pub fn decode_to_slice(src: &[u8], dst: &mut [u8]) -> Result<usize, DecoderError> {
    debug_assert!(
        dst.len() >= src.len() * 2,
        "dst must be at least src.len()*2 bytes for Huffman decode"
    );

    // ── Peekahead fast path ──────────────────────────────────────────────────
    // Maintain a 64-bit MSB-first bit buffer.  We keep bits left-aligned so
    // the top `PEEK_BITS` bits are always the next peek window.
    let mut bit_buf: u64 = 0;
    let mut bits: u32 = 0; // number of valid bits in bit_buf (from MSB)
    let mut src_pos: usize = 0;
    let mut dst_pos: usize = 0;

    // Fill the buffer to at least PEEK_BITS.
    macro_rules! refill {
        () => {
            while src_pos < src.len() && bits <= 56 {
                // SAFETY: src_pos < src.len() checked in while condition
                bit_buf |= unsafe { (*src.get_unchecked(src_pos)) as u64 }
                    << (56 - bits);
                bits += 8;
                src_pos += 1;
            }
        };
    }

    refill!();

    // Process symbols as long as there are bits available.
    // The PEEK_TABLE is indexed by the top PEEK_BITS bits of bit_buf; when
    // bits < PEEK_BITS the lookup is zero-padded, which is still correct for
    // codes ≤ bits because the table stores all suffixes of each code.
    while bits > 0 {
        // Top PEEK_BITS bits of bit_buf select the table entry.
        let peek = (bit_buf >> (64 - PEEK_BITS)) as usize;
        // SAFETY: peek is always < PEEK_SIZE = 1024 (top 10 bits of u64)
        let (sym, consumed) = unsafe { *PEEK_TABLE.get_unchecked(peek) };

        if consumed == 0 || consumed as u32 > bits {
            // consumed == 0: code is longer than PEEK_BITS → scalar fallback.
            // consumed > bits: not enough bits for this code; could be EOS
            //   padding or a >PEEK_BITS code — scalar fallback handles both.
            break;
        }

        // SAFETY: dst is pre-allocated to src.len()*2, and the maximum
        // expansion ratio for Huffman decode is <2× — dst_pos never exceeds
        // the allocated length for valid input.
        unsafe { *dst.get_unchecked_mut(dst_pos) = sym };
        dst_pos += 1;

        bit_buf <<= consumed;
        bits -= consumed as u32;

        if bits < PEEK_BITS {
            refill!();
        }
    }

    // ── Scalar fallback ─────────────────────────────────────────────────────
    // Handles: (a) codes > 10 bits (rare in HTTP/2), (b) symbols straddling
    // the end-of-buffer when bits < PEEK_BITS, (c) final EOS padding.
    // We always run the scalar path when there are leftover bits or bytes so
    // that is_final() provides the authoritative EOS validation.
    if src_pos < src.len() || bits > 0 {
        let mut decoder = Decoder::new();

        // Unified bit buffer: start with whatever the fast path left behind,
        // then refill byte-by-byte from the remaining source as needed.
        // Processing everything through a single nibble loop ensures no bits
        // are silently dropped when `bits % 4 != 0` (e.g. after decoding a
        // symbol whose code length is not a multiple of 4).
        let mut rem_buf = bit_buf;
        let mut rem_bits = bits;
        let mut byte_idx = src_pos;

        loop {
            // Refill from source bytes until we have at least 4 bits.
            // This ensures leftover bits (rem_bits % 4 != 0) are combined
            // with the next source byte rather than silently dropped.
            while rem_bits < 4 && byte_idx < src.len() {
                rem_buf |= (src[byte_idx] as u64) << (56 - rem_bits);
                rem_bits += 8;
                byte_idx += 1;
            }

            if rem_bits < 4 {
                // 0–3 bits remain with no more source data.
                break;
            }

            let nybble = ((rem_buf >> 60) as u8) & 0xF;
            rem_buf <<= 4;
            rem_bits -= 4;

            if let Some(c) = decoder.decode4(nybble)? {
                if dst_pos >= dst.len() {
                    return Err(DecoderError::InvalidHuffmanCode);
                }
                dst[dst_pos] = c;
                dst_pos += 1;
            }
        }

        // Handle the final 0–3 remaining bits.
        //
        // HPACK mandates EOS padding = a prefix of the EOS symbol = all 1-bits.
        // If the remaining bits contain any 0, they are DATA (part of the last
        // symbol) and must be decoded regardless of is_final().  Only when all
        // remaining bits are 1s can they be genuine EOS padding; in that case
        // we check is_final() and skip them if the stream is already valid.
        if rem_bits > 0 {
            // Extract the rem_bits significant bits (right-aligned) from rem_buf.
            let remaining_top = rem_buf >> (64 - rem_bits);
            let all_ones_mask = (1u64 << rem_bits) - 1;
            let is_eos_padding = remaining_top == all_ones_mask;

            if !is_eos_padding || !decoder.is_final() {
                // Either data bits (any 0 present) or decoder not yet final.
                // Pad with 1s to form a complete nibble and feed to the decoder.
                let shift = 4 - rem_bits;
                let nybble = (((rem_buf >> 60) as u8) & 0xF) | ((1u8 << shift) - 1);
                if let Some(c) = decoder.decode4(nybble)? {
                    if dst_pos >= dst.len() {
                        return Err(DecoderError::InvalidHuffmanCode);
                    }
                    dst[dst_pos] = c;
                    dst_pos += 1;
                }
            }
        }

        if !decoder.is_final() {
            return Err(DecoderError::InvalidHuffmanCode);
        }
    }
    // else: fast path consumed every bit cleanly — no EOS tail to validate
    // because the final symbol left exactly 0 buffered bits.

    Ok(dst_pos)
}

// ===== SIMD Header Validation =====

/// Validate that all bytes are valid HTTP header value bytes.
/// Valid range: 0x09 (tab), 0x20..=0x7E, 0x80..=0xFF.
#[cfg(feature = "fast-hpack")]
#[allow(dead_code)]
pub fn validate_header_value(data: &[u8]) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        if data.len() >= 16 {
            return unsafe { validate_header_value_neon(data) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if data.len() >= 16 {
            return unsafe { validate_header_value_sse2(data) };
        }
    }
    validate_header_value_scalar(data)
}

#[cfg(feature = "fast-hpack")]
fn validate_header_value_scalar(data: &[u8]) -> bool {
    data.iter().all(|&b| b == 0x09 || (b >= 0x20 && b != 0x7F))
}

#[cfg(all(feature = "fast-hpack", target_arch = "aarch64"))]
unsafe fn validate_header_value_neon(data: &[u8]) -> bool {
    use std::arch::aarch64::*;

    let min_val = vdupq_n_u8(0x20);
    let del_val = vdupq_n_u8(0x7F);
    let tab_val = vdupq_n_u8(0x09);

    let mut i = 0;
    while i + 16 <= data.len() {
        let v = vld1q_u8(data.as_ptr().add(i));
        // Check >= 0x20
        let ge_space = vcgeq_u8(v, min_val);
        // Check != 0x7F (DEL)
        let not_del = vmvnq_u8(vceqq_u8(v, del_val));
        // Check == 0x09 (TAB, also valid)
        let is_tab = vceqq_u8(v, tab_val);
        // Valid if (>= 0x20 AND != DEL) OR is_tab
        let valid = vorrq_u8(vandq_u8(ge_space, not_del), is_tab);
        if vminvq_u8(valid) == 0 {
            return false;
        }
        i += 16;
    }
    // Scalar remainder
    validate_header_value_scalar(&data[i..])
}

/// Validate header name: must be lowercase token chars.
/// Quick check: no uppercase (0x41-0x5A) and no control chars.
#[cfg(feature = "fast-hpack")]
#[allow(dead_code)]
pub fn validate_header_name(data: &[u8]) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        if data.len() >= 16 {
            return unsafe { validate_header_name_neon(data) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if data.len() >= 16 {
            return unsafe { validate_header_name_sse2(data) };
        }
    }
    validate_header_name_scalar(data)
}

#[cfg(feature = "fast-hpack")]
fn validate_header_name_scalar(data: &[u8]) -> bool {
    data.iter()
        .all(|&b| b >= 0x21 && b != 0x7F && !(b >= 0x41 && b <= 0x5A))
}

#[cfg(all(feature = "fast-hpack", target_arch = "aarch64"))]
unsafe fn validate_header_name_neon(data: &[u8]) -> bool {
    use std::arch::aarch64::*;

    let upper_a = vdupq_n_u8(0x41);
    let upper_z = vdupq_n_u8(0x5A);
    let min_val = vdupq_n_u8(0x21);
    let del_val = vdupq_n_u8(0x7F);

    let mut i = 0;
    while i + 16 <= data.len() {
        let v = vld1q_u8(data.as_ptr().add(i));
        // Reject uppercase: in [A-Z] range
        let ge_a = vcgeq_u8(v, upper_a);
        let le_z = vcleq_u8(v, upper_z);
        let is_upper = vandq_u8(ge_a, le_z);
        // Reject control chars: < 0x21
        let is_ctl = vcltq_u8(v, min_val);
        // Reject DEL
        let is_del = vceqq_u8(v, del_val);
        let bad = vorrq_u8(vorrq_u8(is_upper, is_ctl), is_del);
        if vmaxvq_u8(bad) != 0 {
            return false;
        }
        i += 16;
    }
    validate_header_name_scalar(&data[i..])
}

// ===== x86_64 SSE2 Validation =====

#[cfg(all(feature = "fast-hpack", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn validate_header_value_sse2(data: &[u8]) -> bool {
    use std::arch::x86_64::*;

    let min_val = _mm_set1_epi8(0x20u8 as i8);
    let del_val = _mm_set1_epi8(0x7Fu8 as i8);
    let tab_val = _mm_set1_epi8(0x09u8 as i8);

    let mut i = 0;
    while i + 16 <= data.len() {
        let v = _mm_loadu_si128(data.as_ptr().add(i) as *const __m128i);
        // Unsigned compare >= 0x20: subtract 0x20 saturating, result > 0 means >= 0x20
        // Use signed comparison trick: XOR with 0x80 to convert unsigned to signed order
        let bias = _mm_set1_epi8(i8::MIN); // 0x80
        let v_biased = _mm_xor_si128(v, bias);
        let min_biased = _mm_xor_si128(min_val, bias);
        // v >= 0x20 iff v_biased >= min_biased (signed)
        // _mm_cmplt_epi8 gives v < min, so NOT that for >=
        let lt_space = _mm_cmplt_epi8(v_biased, min_biased);
        // v == DEL
        let is_del = _mm_cmpeq_epi8(v, del_val);
        // v == TAB (also valid)
        let is_tab = _mm_cmpeq_epi8(v, tab_val);
        // Bad if (< 0x20 AND not TAB) OR is_del
        let bad_low = _mm_andnot_si128(is_tab, lt_space);
        let bad = _mm_or_si128(bad_low, is_del);
        if _mm_movemask_epi8(bad) != 0 {
            return false;
        }
        i += 16;
    }
    validate_header_value_scalar(&data[i..])
}

#[cfg(all(feature = "fast-hpack", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn validate_header_name_sse2(data: &[u8]) -> bool {
    use std::arch::x86_64::*;

    let upper_a = _mm_set1_epi8(0x41u8 as i8);
    let upper_z = _mm_set1_epi8(0x5Au8 as i8);
    let min_val = _mm_set1_epi8(0x21u8 as i8);
    let del_val = _mm_set1_epi8(0x7Fu8 as i8);
    let bias = _mm_set1_epi8(i8::MIN); // 0x80

    let mut i = 0;
    while i + 16 <= data.len() {
        let v = _mm_loadu_si128(data.as_ptr().add(i) as *const __m128i);
        let v_biased = _mm_xor_si128(v, bias);

        // Reject control chars: v < 0x21
        let min_biased = _mm_xor_si128(min_val, bias);
        let is_ctl = _mm_cmplt_epi8(v_biased, min_biased);

        // Reject uppercase: v >= 'A' && v <= 'Z'
        let a_biased = _mm_xor_si128(upper_a, bias);
        let z_biased = _mm_xor_si128(upper_z, bias);
        // ge_a: NOT (v < A) = NOT cmplt
        let lt_a = _mm_cmplt_epi8(v_biased, a_biased);
        // gt_z: v > Z = cmpgt
        let gt_z = _mm_cmpgt_epi8(v_biased, z_biased);
        // in_upper_range = NOT lt_a AND NOT gt_z
        let not_in_range = _mm_or_si128(lt_a, gt_z);
        let is_upper = _mm_andnot_si128(not_in_range, _mm_set1_epi8(-1));

        // Reject DEL
        let is_del = _mm_cmpeq_epi8(v, del_val);

        let bad = _mm_or_si128(_mm_or_si128(is_ctl, is_upper), is_del);
        if _mm_movemask_epi8(bad) != 0 {
            return false;
        }
        i += 16;
    }
    validate_header_name_scalar(&data[i..])
}

#[cfg(test)]
mod test {
    use super::*;

    fn decode(src: &[u8]) -> Result<BytesMut, DecoderError> {
        let mut buf = BytesMut::new();
        super::decode(src, &mut buf)
    }

    #[test]
    fn decode_single_byte() {
        assert_eq!("o", decode(&[0b00111111]).unwrap());
        assert_eq!("0", decode(&[7]).unwrap());
        assert_eq!("A", decode(&[(0x21 << 2) + 3]).unwrap());
    }

    #[test]
    fn single_char_multi_byte() {
        assert_eq!("#", decode(&[255, 160 + 15]).unwrap());
        assert_eq!("$", decode(&[255, 200 + 7]).unwrap());
        assert_eq!("\x0a", decode(&[255, 255, 255, 240 + 3]).unwrap());
    }

    #[test]
    fn multi_char() {
        assert_eq!("!0", decode(&[254, 1]).unwrap());
        assert_eq!(" !", decode(&[0b01010011, 0b11111000]).unwrap());
    }

    #[test]
    fn encode_single_byte() {
        let mut dst = BytesMut::with_capacity(1);

        encode(b"o", &mut dst);
        assert_eq!(&dst[..], &[0b00111111]);

        dst.clear();
        encode(b"0", &mut dst);
        assert_eq!(&dst[..], &[7]);

        dst.clear();
        encode(b"A", &mut dst);
        assert_eq!(&dst[..], &[(0x21 << 2) + 3]);
    }

    #[test]
    fn encode_decode_str() {
        const DATA: &[&str] = &[
            "hello world",
            ":method",
            ":scheme",
            ":authority",
            "yahoo.co.jp",
            "GET",
            "http",
            ":path",
            "/images/top/sp2/cmn/logo-ns-130528.png",
            "example.com",
            "hpack-test",
            "xxxxxxx1",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:16.0) Gecko/20100101 Firefox/16.0",
            "accept",
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "cookie",
            "B=76j09a189a6h4&b=3&s=0b",
            "TE",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi non bibendum libero. \
             Etiam ultrices lorem ut.",
        ];

        for s in DATA {
            let mut dst = BytesMut::with_capacity(s.len());

            encode(s.as_bytes(), &mut dst);

            let decoded = decode(&dst).unwrap();

            assert_eq!(&decoded[..], s.as_bytes());
        }
    }

    #[test]
    fn encode_decode_u8() {
        const DATA: &[&[u8]] = &[b"\0", b"\0\0\0", b"\0\x01\x02\x03\x04\x05", b"\xFF\xF8"];

        for s in DATA {
            let mut dst = BytesMut::with_capacity(s.len());

            encode(s, &mut dst);

            let decoded = decode(&dst).unwrap();

            assert_eq!(&decoded[..], &s[..]);
        }
    }
}

#[cfg(all(test, feature = "fast-hpack"))]
mod slice_tests {
    use super::*;

    fn roundtrip(input: &[u8]) {
        // encode using encode()
        let mut encoded = bytes::BytesMut::new();
        encode(input, &mut encoded);
        let encoded_bytes = encoded.freeze();

        // decode using decode_to_slice
        let mut dst = vec![0u8; encoded_bytes.len() * 2 + 4];
        let n = decode_to_slice(&encoded_bytes, &mut dst)
            .unwrap_or_else(|e| panic!("decode_to_slice failed for input {:?} (encoded {:?}): {:?}", input, &encoded_bytes[..], e));
        assert_eq!(&dst[..n], input, "roundtrip mismatch for input {:?} (encoded {:?})", input, &encoded_bytes[..]);
    }

    #[test]
    fn test_roundtrip_ascii() {
        roundtrip(b"application/json");
        roundtrip(b"GET");
        roundtrip(b"https");
        roundtrip(b"/index.html");
        roundtrip(b"www.example.com");
        roundtrip(b"no-cache");
    }

    #[test]
    fn test_roundtrip_all_bytes() {
        for b in 0u8..=127 {
            roundtrip(&[b]);
        }
        for b in 0u8..=127 {
            roundtrip(&[b, b, b]);
        }
    }

    #[test]
    fn test_roundtrip_long() {
        let long_str: Vec<u8> = b"content-type: application/json; charset=utf-8".to_vec();
        roundtrip(&long_str);
    }

    #[test]
    fn test_roundtrip_plus_sign() {
        // '+' has an 11-bit Huffman code (> PEEK_BITS=10), triggering the
        // scalar fallback.  The leftover bits after fast-path decoding must
        // be properly carried into the scalar path — not silently dropped.
        roundtrip(b"+");
        roundtrip(b"+xml");
        roundtrip(b"xhtml+xml");
        roundtrip(b"application/xhtml+xml");
        roundtrip(b"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");
    }

    #[test]
    fn test_roundtrip_mixed_bit_lengths() {
        // Mix of 5-bit ('t'), 7-bit ('x') and 11-bit ('+') codes exercises
        // all transitions between fast-path and scalar fallback.
        roundtrip(b"text/html,application/xhtml");
        roundtrip(b"text/html,application/xhtml+");
        roundtrip(b"text/html,application/xhtml+xml");
        roundtrip(b"text/html,application/xhtml+xml,application/xml");
    }

    #[test]
    fn test_roundtrip_cookie() {
        // This cookie string contains chars with >10-bit Huffman codes
        // (#=12bits, $=13bits, '=11bits, +=11bits, ]=13bits, ^=14bits)
        // which trigger the scalar fallback path. The last character 'y'
        // must not be dropped.
        roundtrip(b"anj=Kfu=8fG68%Cxrx)0s]#%2L_'x%SEV/hnJPh4FQV_eKj?9AMF4:V)4hY/82QjU'-Rw1Ra^uI$+VZ; path=/; expires=Fri, 01-Feb-2013 13:29:47 GMT; domain=.adnxs.com; HttpOnly");
        roundtrip(b"HttpOnly");
        roundtrip(b"domain=.adnxs.com; HttpOnly");
    }
}

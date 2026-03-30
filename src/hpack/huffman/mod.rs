mod table;

use self::table::{DECODE_TABLE, ENCODE_TABLE};
use crate::hpack::DecoderError;

use bytes::{BufMut, BytesMut};

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

    /// Decode a full input byte (8 bits) by composing two 4-bit lookups.
    /// Writes 0-2 output bytes to `dst` at `dst_pos`, returns new dst_pos.
    #[cfg(feature = "fast-hpack")]
    #[inline(always)]
    fn decode_byte(&mut self, input: u8, dst: &mut [u8], mut dst_pos: usize) -> Result<usize, DecoderError> {
        let hi = (input >> 4) as usize;
        let lo = (input & 0x0F) as usize;

        // High nibble
        let (next1, byte1, flags1) = DECODE_TABLE[self.state as usize][hi];
        if flags1 & ERROR != 0 {
            return Err(DecoderError::InvalidHuffmanCode);
        }
        if flags1 & DECODED != 0 {
            if dst_pos >= dst.len() {
                return Err(DecoderError::InvalidHuffmanCode);
            }
            dst[dst_pos] = byte1;
            dst_pos += 1;
        }

        // Low nibble
        let (next2, byte2, flags2) = DECODE_TABLE[next1 as usize][lo];
        if flags2 & ERROR != 0 {
            return Err(DecoderError::InvalidHuffmanCode);
        }
        if flags2 & DECODED != 0 {
            if dst_pos >= dst.len() {
                return Err(DecoderError::InvalidHuffmanCode);
            }
            dst[dst_pos] = byte2;
            dst_pos += 1;
        }

        self.state = next2;
        self.maybe_eos = flags2 & MAYBE_EOS != 0;

        Ok(dst_pos)
    }

    fn is_final(&self) -> bool {
        self.state == 0 || self.maybe_eos
    }
}

/// Fast Huffman decode directly into a pre-allocated byte slice.
/// Returns the number of bytes written to `dst`.
/// `dst` must be at least `src.len() * 2` bytes long.
///
/// This avoids all BytesMut overhead: no reserve, no split, no freeze.
#[cfg(feature = "fast-hpack")]
pub fn decode_to_slice(src: &[u8], dst: &mut [u8]) -> Result<usize, DecoderError> {
    debug_assert!(
        dst.len() >= src.len() * 2,
        "dst must be at least src.len()*2 bytes for Huffman decode"
    );
    let mut decoder = Decoder::new();
    let mut pos = 0;

    for &b in src {
        pos = decoder.decode_byte(b, dst, pos)?;
    }

    if !decoder.is_final() {
        return Err(DecoderError::InvalidHuffmanCode);
    }

    Ok(pos)
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

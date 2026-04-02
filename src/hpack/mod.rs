mod decoder;
mod encoder;
pub(crate) mod header;
pub(crate) mod huffman;
mod table;

#[cfg(feature = "fast-hpack")]
pub(crate) mod fast_decoder;

#[cfg(test)]
mod test;

use bytes::BytesMut;
use std::io::Cursor;

pub use self::decoder::{Decoder, DecoderError, NeedMore};
pub use self::encoder::Encoder;
pub use self::header::{BytesStr, Header};

#[cfg(feature = "fast-hpack")]
pub use self::fast_decoder::FastDecoder;

/// Trait abstracting over HPACK decoders so HeaderBlock::load can be generic.
pub(crate) trait HpackDecode {
    fn decode<F>(
        &mut self,
        src: &mut Cursor<&mut BytesMut>,
        f: F,
    ) -> Result<(), DecoderError>
    where
        F: FnMut(Header);
}

impl HpackDecode for Decoder {
    fn decode<F>(
        &mut self,
        src: &mut Cursor<&mut BytesMut>,
        f: F,
    ) -> Result<(), DecoderError>
    where
        F: FnMut(Header),
    {
        Decoder::decode(self, src, f)
    }
}

#[cfg(feature = "fast-hpack")]
impl HpackDecode for FastDecoder {
    fn decode<F>(
        &mut self,
        src: &mut Cursor<&mut BytesMut>,
        f: F,
    ) -> Result<(), DecoderError>
    where
        F: FnMut(Header),
    {
        FastDecoder::decode(self, src, f)
    }
}

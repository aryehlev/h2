#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use bytes::BytesMut;
use std::io::Cursor;

/// Structure-aware HPACK decode fuzzing.
/// Tests the decoder with various table sizes and multiple sequential frames,
/// exercising dynamic table state management across frames.
#[derive(Arbitrary, Debug)]
struct HpackInput {
    table_size: u16,
    frames: Vec<Vec<u8>>,
}

fuzz_target!(|input: HpackInput| {
    let mut decoder = h2::hpack::Decoder::new(input.table_size as usize);
    for frame in &input.frames {
        let mut buf = BytesMut::from(frame.as_slice());
        let _ = decoder.decode(&mut Cursor::new(&mut buf), |_h| {});
    }
});

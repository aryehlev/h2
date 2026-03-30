use bytes::BytesMut;
use std::io::Cursor;

/// Focused benchmark for HPACK decode performance.
/// Run with:
///   cargo run --release --bench hpack_decode
///   cargo run --release --bench hpack_decode --features fast-hpack
fn main() {
    // Pre-built HPACK encoded block representing ~15 typical HTTP/2 headers.
    // Contains: indexed static entries, literal with indexing (Huffman + raw),
    // and literal without indexing.
    let encoded_bytes = build_encoded_block();

    println!("Encoded header block size: {} bytes", encoded_bytes.len());
    println!(
        "Feature: {}",
        if cfg!(feature = "fast-hpack") {
            "fast-hpack (arena + fast huffman)"
        } else {
            "default (standard decoder)"
        }
    );

    // Warm up
    for _ in 0..1000 {
        decode_once(&encoded_bytes);
    }

    // Benchmark 1: New decoder per request (worst case — no table reuse)
    let iterations = 1_000_000;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        decode_once(&encoded_bytes);
    }
    let elapsed = start.elapsed();

    let per_iter = elapsed / iterations as u32;
    let throughput = (encoded_bytes.len() as f64 * iterations as f64) / elapsed.as_secs_f64();

    println!("\n--- New decoder per request ---");
    println!("Iterations:  {}", iterations);
    println!("Total time:  {:?}", elapsed);
    println!("Per decode:  {:?}", per_iter);
    println!(
        "Throughput:  {:.1} MB/s",
        throughput / (1024.0 * 1024.0)
    );
    println!(
        "Decodes/sec: {:.0}",
        iterations as f64 / elapsed.as_secs_f64()
    );

    // Benchmark 2: Reused decoder (steady state — dynamic table populated)
    println!("\n--- Reused decoder (steady state) ---");
    let mut decoder = new_decoder();
    // Prime the dynamic table
    for _ in 0..100 {
        let mut buf = BytesMut::from(encoded_bytes.as_slice());
        decoder
            .decode(&mut Cursor::new(&mut buf), |_h| {})
            .unwrap();
    }

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let mut buf = BytesMut::from(encoded_bytes.as_slice());
        let mut count = 0usize;
        decoder
            .decode(&mut Cursor::new(&mut buf), |_h| {
                count += 1;
            })
            .unwrap();
        std::hint::black_box(count);
    }
    let elapsed = start.elapsed();

    let per_iter = elapsed / iterations as u32;
    let throughput = (encoded_bytes.len() as f64 * iterations as f64) / elapsed.as_secs_f64();

    println!("Iterations:  {}", iterations);
    println!("Total time:  {:?}", elapsed);
    println!("Per decode:  {:?}", per_iter);
    println!(
        "Throughput:  {:.1} MB/s",
        throughput / (1024.0 * 1024.0)
    );
    println!(
        "Decodes/sec: {:.0}",
        iterations as f64 / elapsed.as_secs_f64()
    );
}

#[cfg(not(feature = "fast-hpack"))]
fn new_decoder() -> h2::hpack::Decoder {
    h2::hpack::Decoder::new(4096)
}

#[cfg(feature = "fast-hpack")]
fn new_decoder() -> h2::hpack::FastDecoder {
    h2::hpack::FastDecoder::new(4096)
}

fn decode_once(encoded: &[u8]) {
    let mut decoder = new_decoder();
    let mut buf = BytesMut::from(encoded);
    let mut count = 0usize;
    decoder
        .decode(&mut Cursor::new(&mut buf), |_h| {
            count += 1;
        })
        .unwrap();
    std::hint::black_box(count);
}

/// Build an HPACK-encoded header block using the encoder.
fn build_encoded_block() -> Vec<u8> {
    use h2::hpack::{Encoder, Header};
    use http::header::HeaderName;
    use http::header::HeaderValue;

    let mut encoder = Encoder::default();

    // Build headers using only public API
    let headers: Vec<Header<Option<HeaderName>>> = vec![
        Header::Method(http::Method::POST),
        Header::Status(http::StatusCode::OK),
        Header::Field {
            name: Some(http::header::CONTENT_TYPE),
            value: HeaderValue::from_static("application/json"),
        },
        Header::Field {
            name: Some(http::header::CONTENT_LENGTH),
            value: HeaderValue::from_static("1234"),
        },
        Header::Field {
            name: Some(http::header::ACCEPT),
            value: HeaderValue::from_static("application/json"),
        },
        Header::Field {
            name: Some(http::header::ACCEPT_ENCODING),
            value: HeaderValue::from_static("gzip, deflate"),
        },
        Header::Field {
            name: Some(http::header::USER_AGENT),
            value: HeaderValue::from_static("BidClient/2.0"),
        },
        Header::Field {
            name: Some(http::header::HOST),
            value: HeaderValue::from_static("bidder.example.com"),
        },
        Header::Field {
            name: Some(HeaderName::from_static("x-openrtb-version")),
            value: HeaderValue::from_static("2.6"),
        },
        Header::Field {
            name: Some(HeaderName::from_static("x-request-id")),
            value: HeaderValue::from_static("550e8400-e29b-41d4-a716-446655440000"),
        },
        Header::Field {
            name: Some(http::header::DATE),
            value: HeaderValue::from_static("Mon, 30 Mar 2026 12:00:00 GMT"),
        },
        Header::Field {
            name: Some(HeaderName::from_static("x-forwarded-for")),
            value: HeaderValue::from_static("203.0.113.50"),
        },
        Header::Field {
            name: Some(HeaderName::from_static("x-forwarded-proto")),
            value: HeaderValue::from_static("https"),
        },
    ];

    let mut dst = BytesMut::new();
    encoder.encode(headers.into_iter(), &mut dst);
    dst.to_vec()
}

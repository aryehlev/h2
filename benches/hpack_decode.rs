use bytes::BytesMut;
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn main() {
    let cdn_block = build_cdn_block();

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!("=== CDN response (41 headers, {} bytes) ===", cdn_block.len());

    // --- Single-threaded allocation count ---
    let warmup = 1000;
    for _ in 0..warmup {
        decode_once(&cdn_block);
    }
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
    let alloc_iters = 1000;
    for _ in 0..alloc_iters {
        decode_once(&cdn_block);
    }
    println!(
        "Allocs/decode: {:.1}",
        ALLOC_COUNT.load(Ordering::SeqCst) as f64 / alloc_iters as f64
    );
    println!(
        "Bytes/decode:  {:.0}",
        ALLOC_BYTES.load(Ordering::SeqCst) as f64 / alloc_iters as f64
    );

    // --- Single-threaded speed ---
    let iterations = 1_000_000u64;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        decode_once(&cdn_block);
    }
    let elapsed = start.elapsed();
    println!(
        "\n1 thread:  {:>7.0} decodes/sec  ({:.2} µs/decode)",
        iterations as f64 / elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64,
    );

    // --- Multi-threaded: scale up contention ---
    let block = Arc::new(cdn_block);
    for threads in [2, num_threads, num_threads * 2] {
        let per_thread = iterations / threads as u64;
        let block = Arc::clone(&block);

        let start = std::time::Instant::now();
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let block = Arc::clone(&block);
                std::thread::spawn(move || {
                    for _ in 0..per_thread {
                        decode_once(&block);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let elapsed = start.elapsed();
        let total_decodes = per_thread * threads as u64;

        println!(
            "{} threads: {:>7.0} decodes/sec  ({:.2} µs/decode)",
            threads,
            total_decodes as f64 / elapsed.as_secs_f64(),
            elapsed.as_secs_f64() * 1_000_000.0 / (total_decodes as f64 / threads as f64),
        );
    }
}

fn decode_once(encoded: &[u8]) {
    let mut decoder = h2::hpack::Decoder::new(4096);
    let mut buf = BytesMut::from(encoded);
    let mut count = 0usize;
    decoder
        .decode(&mut Cursor::new(&mut buf), |_h| {
            count += 1;
        })
        .unwrap();
    std::hint::black_box(count);
}

fn build_cdn_block() -> Vec<u8> {
    use h2::hpack::{Encoder, Header};
    use http::header::HeaderName;
    use http::header::HeaderValue;

    let mut encoder = Encoder::default();

    let h = |name: HeaderName, val: &'static str| Header::Field {
        name: Some(name),
        value: HeaderValue::from_static(val),
    };
    let hx = |name: &'static str, val: &'static str| Header::Field {
        name: Some(HeaderName::from_static(name)),
        value: HeaderValue::from_static(val),
    };

    let headers: Vec<Header<Option<HeaderName>>> = vec![
        Header::Status(http::StatusCode::OK),
        h(http::header::CONTENT_TYPE, "application/json; charset=utf-8"),
        h(http::header::CONTENT_LENGTH, "4096"),
        h(http::header::CONTENT_ENCODING, "gzip"),
        h(http::header::DATE, "Mon, 30 Mar 2026 12:00:00 GMT"),
        h(http::header::SERVER, "nginx/1.25.4"),
        h(http::header::VARY, "Accept-Encoding, Origin"),
        h(http::header::CACHE_CONTROL, "public, max-age=3600, s-maxage=7200"),
        h(http::header::ETAG, "\"a1b2c3d4e5f6\""),
        h(http::header::LAST_MODIFIED, "Sun, 29 Mar 2026 08:00:00 GMT"),
        h(http::header::EXPIRES, "Mon, 30 Mar 2026 13:00:00 GMT"),
        h(http::header::SET_COOKIE, "session=abc123; Path=/; HttpOnly; Secure"),
        h(http::header::SET_COOKIE, "tracking=xyz789; Path=/; SameSite=Lax"),
        h(http::header::STRICT_TRANSPORT_SECURITY, "max-age=31536000; includeSubDomains"),
        h(http::header::ACCESS_CONTROL_ALLOW_ORIGIN, "https://app.example.com"),
        h(http::header::CONTENT_DISPOSITION, "inline"),
        h(http::header::ACCEPT_RANGES, "bytes"),
        h(http::header::AGE, "120"),
        h(http::header::LINK, "</style.css>; rel=preload; as=style"),
        h(http::header::RETRY_AFTER, "60"),
        h(http::header::VIA, "1.1 cdn-edge-01"),
        h(http::header::LOCATION, "https://api.example.com/v2/resource"),
        h(http::header::AUTHORIZATION, "Bearer eyJhbGciOiJSUzI1NiJ9.short"),
        h(http::header::COOKIE, "session=abc123; pref=dark"),
        h(http::header::ACCEPT, "application/json"),
        h(http::header::ACCEPT_LANGUAGE, "en-US,en;q=0.9"),
        h(http::header::ACCEPT_ENCODING, "gzip, deflate, br"),
        h(http::header::USER_AGENT, "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"),
        h(http::header::HOST, "api.example.com"),
        h(http::header::REFERER, "https://app.example.com/dashboard"),
        h(http::header::IF_NONE_MATCH, "\"a1b2c3d4e5f6\""),
        h(http::header::IF_MODIFIED_SINCE, "Sun, 29 Mar 2026 08:00:00 GMT"),
        hx("x-request-id", "550e8400-e29b-41d4-a716-446655440000"),
        hx("x-trace-id", "abcdef1234567890abcdef1234567890"),
        hx("x-ratelimit-limit", "1000"),
        hx("x-ratelimit-remaining", "997"),
        hx("x-ratelimit-reset", "1711800000"),
        hx("x-cdn-pop", "SFO"),
        hx("x-cache", "HIT"),
        hx("x-powered-by", "h2/0.4"),
        hx("x-content-type-options", "nosniff"),
    ];

    let mut dst = BytesMut::new();
    encoder.encode(headers.into_iter(), &mut dst);
    dst.to_vec()
}

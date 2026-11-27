//! `VICTORY_FETCH_WAITS=1`, for the log: the waits for the card to fetch
//! what shaders wrote to host copies (`Host::card_back`,
//! `Host::relay_card_back`), by who waited and on which queue, how much
//! work of the card's was ahead of the fetch, and whether the same pieces
//! are fetched frame after frame.
//!
//! In Leafy's run (leafy-0314) 11 fetches a frame waited 10 ms in all, 16.8
//! MB of 273 pieces. With the relay on they were the relay thread's, and no
//! submission waited for the relay (0 settles): those waits held up no
//! frame.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FETCH_WAITS").is_some_and(|v| v == "1"))
}

/// Who waited for a fetch: the submitting thread (on the queue of the
/// submissions), the relay on its copy queue, the relay on the queue of the
/// submissions.
pub(super) const SUBMITTING: usize = 0;
pub(super) const RELAY_COPY: usize = 1;
pub(super) const RELAY_MAIN: usize = 2;
const WHO: [&str; 3] = ["submitting thread", "relay, copy queue", "relay, main queue"];

/// Per who: fetches, microseconds waited, pieces, bytes, pieces fetched the
/// frame before too, their bytes, submissions still on the card when the
/// fetch was sent (the submitting thread's only), the longest wait in
/// microseconds.
static COUNTS: [[AtomicU64; 8]; 3] = [const { [const { AtomicU64::new(0) }; 8] }; 3];

/// The pieces fetched (title address of the host copy, from, to) and the
/// last frame each was.
static SEEN: Mutex<Option<HashMap<(usize, usize, usize), u64>>> = Mutex::new(None);

/// A fetch by `who` of `pieces` (as in [`SEEN`]) that waited `us`, sent
/// with `ahead` submissions still on the card.
pub(super) fn fetched(who: usize, pieces: impl Iterator<Item = (usize, usize, usize)>, us: u64, ahead: u64) {
    let frame = crate::frame::current();
    let (mut n, mut bytes, mut again, mut again_bytes) = (0u64, 0u64, 0u64, 0u64);
    {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let seen = seen.get_or_insert_with(Default::default);
        if seen.len() >= 1 << 16 {
            seen.retain(|_, at| *at + 2 >= frame);
        }
        for piece in pieces {
            let len = (piece.2 - piece.1) as u64;
            n += 1;
            bytes += len;
            if let Some(at) = seen.insert(piece, frame)
                && at + 1 == frame
            {
                again += 1;
                again_bytes += len;
            }
        }
    }
    let c = &COUNTS[who];
    for (i, add) in [1, us, n, bytes, again, again_bytes, ahead].into_iter().enumerate() {
        c[i].fetch_add(add, Relaxed);
    }
    c[7].fetch_max(us, Relaxed);
}

pub(super) fn log(frames: f64) {
    if !on() {
        return;
    }
    let lines: Vec<String> = (0..3)
        .filter_map(|who| {
            let c = COUNTS[who].each_ref().map(|c| c.swap(0, Relaxed));
            (c[0] > 0).then(|| {
                let ahead = match who {
                    SUBMITTING => format!(", {:.1} submissions on the card ahead of each", c[6] as f64 / c[0] as f64),
                    _ => String::new(),
                };
                format!(
                    "{}: {:.1} fetches {:.2} ms (longest {:.2} ms), {:.0} pieces {:.2} MB, {:.0} pieces {:.2} MB fetched the frame \
                     before too{ahead}",
                    WHO[who],
                    c[0] as f64 / frames,
                    c[1] as f64 / 1e3 / frames,
                    c[7] as f64 / 1e3,
                    c[2] as f64 / frames,
                    c[3] as f64 / 1e6 / frames,
                    c[4] as f64 / frames,
                    c[5] as f64 / 1e6 / frames
                )
            })
        })
        .collect();
    tracing::info!(
        "fetches of what shaders wrote, per frame (VICTORY_FETCH_WAITS=1): {}",
        if lines.is_empty() { "none".to_string() } else { lines.join("; ") }
    );
}

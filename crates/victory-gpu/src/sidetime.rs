//! The card's time in the sends the main lists' stamps leave out
//! (`VICTORY_SIDE_STAMPS=1`, diagnostics builds): the list that sends host
//! copies up to the card's memory and hashes them, the compare behind a
//! submission's lists, the fetch of what differed, and the list of steps
//! done on the card (`VICTORY_CARD_STEPS`). Each list gets a pair of
//! timestamps (the first list two, sends and hashes); the frame line adds
//! them up by kind.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

pub(crate) const SYNC: usize = 0;
pub(crate) const CHECK: usize = 1;
pub(crate) const FETCH: usize = 2;
pub(crate) const HASH: usize = 3;
pub(crate) const STEPS: usize = 4;
const KINDS: usize = 5;
const NAMES: [&str; KINDS] = ["sends", "compares", "fetches", "hashes", "steps on the card"];
/// The order on the frame line.
const ORDER: [usize; KINDS] = [SYNC, HASH, CHECK, FETCH, STEPS];

/// Pairs of stamps in the ring.
const PAIRS: u32 = 4096;

struct Ring {
    next: u32,
    /// Pairs given out and not read yet, with their kind.
    out: VecDeque<(u32, usize)>,
    ticks: [u64; KINDS],
    lists: [u64; KINDS],
    /// Pairs the ring came round to before they were read.
    lost: u64,
}

struct Stamps {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    data: usize,
    hz: f64,
    ring: Mutex<Ring>,
}

// SAFETY: free-threaded objects; the mapping is only touched under the lock.
unsafe impl Send for Stamps {}
// SAFETY: as above.
unsafe impl Sync for Stamps {}

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SIDE_STAMPS").is_ok_and(|v| v == "1"))
}

static STAMPS: OnceLock<Option<Stamps>> = OnceLock::new();

fn stamps(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> Option<&'static Stamps> {
    if !on() {
        return None;
    }
    STAMPS
        .get_or_init(|| {
            let mut heap: Option<ID3D12QueryHeap> = None;
            let mut back: Option<ID3D12Resource> = None;
            let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: u64::from(PAIRS) * 16,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            // SAFETY: plain object creation; the read-back buffer stays
            // mapped for the run.
            unsafe {
                let hz = queue.GetTimestampFrequency().ok()? as f64;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: PAIRS * 2, NodeMask: 0 },
                        &mut heap,
                    )
                    .ok()?;
                device
                    .CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back)
                    .ok()?;
                let back = back?;
                let mut data = std::ptr::null_mut();
                back.Map(0, None, Some(&mut data)).ok()?;
                std::ptr::write_bytes(data.cast::<u8>(), 0xff, PAIRS as usize * 16);
                tracing::info!("side sends stamped (VICTORY_SIDE_STAMPS=1): the frame line gives their card time by kind");
                Some(Stamps {
                    heap: heap?,
                    back,
                    data: data as usize,
                    hz,
                    ring: Mutex::new(Ring { next: 0, out: VecDeque::new(), ticks: [0; KINDS], lists: [0; KINDS], lost: 0 }),
                })
            }
        })
        .as_ref()
}

/// Adds up the pairs the card has written and forgets them.
fn harvest(s: &Stamps, ring: &mut Ring) {
    let data = s.data as *mut u64;
    let Ring { out, ticks, lists, .. } = ring;
    out.retain(|&(pair, kind)| {
        // SAFETY: two slots of the mapping, inside it.
        let (a, b) = unsafe { (data.add(pair as usize * 2).read_volatile(), data.add(pair as usize * 2 + 1).read_volatile()) };
        if a == u64::MAX || b == u64::MAX {
            return true;
        }
        ticks[kind] += b.saturating_sub(a);
        lists[kind] += 1;
        false
    });
}

/// The first stamp, at the start of a side list being recorded; the pair
/// to give [`end`] before the list is closed.
///
/// # Safety
/// `list` is open.
pub(crate) unsafe fn begin(device: &ID3D12Device, queue: &ID3D12CommandQueue, list: &ID3D12GraphicsCommandList, kind: usize) -> Option<u32> {
    let s = stamps(device, queue)?;
    let mut ring = s.ring.lock().unwrap_or_else(|e| e.into_inner());
    if ring.out.len() >= PAIRS as usize / 2 {
        harvest(s, &mut ring);
    }
    if ring.out.len() >= PAIRS as usize - 1 {
        ring.out.pop_front();
        ring.lost += 1;
    }
    let pair = ring.next;
    ring.next = (pair + 1) % PAIRS;
    ring.out.retain(|&(p, _)| p != pair);
    let data = s.data as *mut u64;
    // SAFETY: two slots of the mapping; the card writes them only after
    // this list runs.
    unsafe {
        data.add(pair as usize * 2).write_volatile(u64::MAX);
        data.add(pair as usize * 2 + 1).write_volatile(u64::MAX);
    }
    ring.out.push_back((pair, kind));
    // SAFETY: an open list (the caller's word); a slot of our heap.
    unsafe { list.EndQuery(&s.heap, D3D12_QUERY_TYPE_TIMESTAMP, pair * 2) };
    Some(pair)
}

/// The second stamp of `pair`, and both into the read-back buffer.
///
/// # Safety
/// `list` is open, the one [`begin`] was given.
pub(crate) unsafe fn end(list: &ID3D12GraphicsCommandList, pair: Option<u32>) {
    let (Some(pair), Some(Some(s))) = (pair, STAMPS.get()) else {
        return;
    };
    // SAFETY: an open list; slots of our heap and buffer.
    unsafe {
        list.EndQuery(&s.heap, D3D12_QUERY_TYPE_TIMESTAMP, pair * 2 + 1);
        list.ResolveQueryData(&s.heap, D3D12_QUERY_TYPE_TIMESTAMP, pair * 2, 2, &s.back, u64::from(pair) * 16);
    }
}

/// For the frame line: card milliseconds and lists per frame of each kind
/// read since the last line.
pub(crate) fn line(frames: f64) -> Option<String> {
    let s = STAMPS.get()?.as_ref()?;
    let mut ring = s.ring.lock().unwrap_or_else(|e| e.into_inner());
    harvest(s, &mut ring);
    let ticks = std::mem::take(&mut ring.ticks);
    let lists = std::mem::take(&mut ring.lists);
    let mut text = String::new();
    for (i, kind) in ORDER.into_iter().enumerate() {
        text += &format!(
            "{}{} {:.1} ms ({:.1} lists)",
            if i == 0 { "" } else { ", " },
            NAMES[kind],
            ticks[kind] as f64 / s.hz * 1000.0 / frames,
            lists[kind] as f64 / frames
        );
    }
    let total: u64 = ticks.iter().sum();
    text += &format!(", in all {:.1} ms", total as f64 / s.hz * 1000.0 / frames);
    if ring.lost > 0 {
        text += &format!(" ({} pairs lost since the start)", ring.lost);
    }
    Some(text)
}

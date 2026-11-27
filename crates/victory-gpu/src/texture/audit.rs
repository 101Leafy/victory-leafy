//! `VICTORY_TEXTURE_AUDIT=<from>-<to>` (frames): what each staged texture
//! of 256 KiB or more was last read from is kept, 64 KiB at a time, and on
//! every 30th frame of that range the title memory of each is compared with
//! it. A texture whose memory changed since is logged once per read, with
//! whether anything told of a write there since (a mapping or a kernel write
//! the runtime announced, or a write the guards do not see) and which part
//! of it changed (the largest level lies last). One that changed with
//! nothing told missed its write; one that was told and is still old was
//! told and not read again. The shirt that showed random blocks in a
//! close-up (dc-d, frame 2469) is to be one or the other, or neither: then
//! the title's memory held those bytes.
//!
//! `VICTORY_TEXTURE_AUDIT_TELL=1`: a changed one is also told of, as a
//! write the runtime announced and as one the guards do not see, so the
//! frames after show whether telling would have brought it up to date.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Textures smaller than this are left out.
const LEAST: usize = 256 << 10;
/// Bytes compared at a time.
const BLOCK: usize = 64 << 10;
/// Frames between comparisons.
const EVERY: u64 = 30;

struct Read {
    blocks: Vec<u128>,
    frame: u64,
    /// The last frame something told of a write there, and how: 1 mapped
    /// or announced, 2 behind the guards' back.
    told: Option<(u64, u8)>,
    logged: bool,
}

/// By memory address and bytes.
static READS: Mutex<BTreeMap<(usize, usize), Read>> = Mutex::new(BTreeMap::new());
static LARGEST: AtomicU64 = AtomicU64::new(0);
static LAST: AtomicU64 = AtomicU64::new(u64::MAX);

fn range() -> Option<(u64, u64)> {
    static RANGE: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();
    *RANGE.get_or_init(|| {
        let text = std::env::var("VICTORY_TEXTURE_AUDIT").ok()?;
        let (from, to) = text.split_once('-')?;
        Some((from.trim().parse().ok()?, to.trim().parse().ok()?))
    })
}

fn tell() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_AUDIT_TELL").is_some_and(|v| v != "0"))
}

fn blocks(bytes: &[u8]) -> Vec<u128> {
    bytes.chunks(BLOCK).map(|b| super::digest(b).0).collect()
}

/// The blocks that differ: how many, the first and the last.
fn differ(old: &[u128], new: &[u128]) -> Option<(usize, usize, usize)> {
    let changed: Vec<usize> = old.iter().zip(new).enumerate().filter(|(_, (a, b))| a != b).map(|(i, _)| i).collect();
    Some((changed.len(), *changed.first()?, *changed.last()?))
}

/// The texture at `memory` was read from `bytes` (title memory, tiled).
pub(crate) fn read(memory: usize, bytes: &[u8]) {
    let Some((_, to)) = range() else {
        return;
    };
    let frame = crate::frame::current();
    if bytes.len() < LEAST || frame > to {
        return;
    }
    let read = Read { blocks: blocks(bytes), frame, told: None, logged: false };
    LARGEST.fetch_max(bytes.len() as u64, Relaxed);
    READS.lock().unwrap_or_else(|e| e.into_inner()).insert((memory, bytes.len()), read);
}

/// Something told of a write to title memory `[at, at + len)`, `how` as in
/// [`Read::told`]. Locks one mutex and calls nothing.
pub(crate) fn told(at: usize, len: usize, how: u8) {
    if range().is_none() {
        return;
    }
    let end = at.saturating_add(len);
    let earliest = at.saturating_sub(LARGEST.load(Relaxed) as usize);
    let frame = crate::frame::current();
    let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
    for (&(memory, bytes), read) in reads.range_mut((earliest, 0)..(end, 0)) {
        if memory + bytes > at {
            read.told = Some((frame, how));
        }
    }
}

/// Once a frame: compares on every [`EVERY`]th frame of the range.
pub(crate) fn check(frame: u64) {
    let Some((from, to)) = range() else {
        return;
    };
    if frame < from || frame > to || (frame - from) % EVERY != 0 || LAST.swap(frame, Relaxed) == frame {
        return;
    }
    let keys: Vec<(usize, usize)> = READS.lock().unwrap_or_else(|e| e.into_inner()).keys().copied().collect();
    let mut buffer = Vec::new();
    let (mut compared, mut bytes, mut changed, mut untold) = (0u64, 0u64, 0u64, 0u64);
    let mut to_tell = Vec::new();
    for (memory, len) in keys {
        buffer.resize(len, 0);
        crate::host::copy_mapped(buffer.as_mut_ptr(), memory as *const u8, len, false);
        let now = blocks(&buffer);
        compared += 1;
        bytes += len as u64;
        // (Before `READS`: a refresh holds its texture's lock as it reads.)
        let staging = super::staged_under(memory, memory + 1).into_iter().find(|s| s.memory() == (memory, len));
        let partly = staging.as_ref().and_then(|s| s.last.try_lock().ok().map(|last| last.partial.is_some()));
        // Whether a list sampled it since: a texture nothing draws is not
        // read again, and its old bytes show nowhere.
        let used = staging.as_ref().map(|s| frame.saturating_sub(s.used.load(Relaxed) >> 32));
        let once = staging.as_ref().is_some_and(|s| s.once.load(Relaxed));
        let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(read) = reads.get_mut(&(memory, len)) else {
            continue;
        };
        let Some((count, first, last)) = differ(&read.blocks, &now) else {
            continue;
        };
        changed += 1;
        if read.told.is_none_or(|(at, _)| at < read.frame) {
            untold += 1;
        }
        if read.logged {
            continue;
        }
        read.logged = true;
        let zero = buffer[first * BLOCK..].iter().take(BLOCK).all(|&b| b == 0);
        let staged = staging.is_some();
        let told = read.told.map(|(at, how)| format!("{} frames ago ({})", frame - at, if how == 1 { "announced" } else { "behind the guards" }));
        tracing::warn!(
            address = format_args!("{memory:#x}"),
            bytes = len,
            frames_since_read = frame - read.frame,
            blocks = read.blocks.len(),
            changed_blocks = count,
            first_changed = first,
            last_changed = last,
            first_changed_zero = zero,
            staged,
            partly_mapped = partly.unwrap_or(false),
            frames_since_used = used.unwrap_or(u64::MAX),
            once,
            told = told.as_deref().unwrap_or("never since its read"),
            "texture audit: title memory changed since the texture was last read"
        );
        if tell() {
            to_tell.push((memory, len));
        }
    }
    for (memory, len) in to_tell {
        crate::descriptor::host::remapped(memory, len);
        crate::descriptor::host::page_written(memory, len);
    }
    tracing::info!(frame, compared, bytes, changed, untold, "texture audit");
}

#[cfg(test)]
mod tests {
    #[test]
    fn differ_names_the_changed_blocks() {
        let old = [1, 2, 3, 4, 5];
        assert_eq!(super::differ(&old, &old), None);
        assert_eq!(super::differ(&old, &[1, 2, 9, 4, 9]), Some((2, 2, 4)));
    }
}

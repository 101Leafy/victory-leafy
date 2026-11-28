//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! A watched change detiles, of each subresource, only the rows whose
//! tiled bytes the pages written touch; the staging buffer holds the rest
//! from before.
//!
//! shortcut: the whole staging buffer is still copied to the host texture.
//! A change to an unwatched texture, a first detiling, and a texture with
//! levels on demand detile the whole texture. Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

pub(crate) mod ahead;
#[cfg(feature = "diagnostics")]
pub(crate) mod audit;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

pub(crate) mod prepare;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    /// A buffer of its own, or the block [`Staging::piece`] is part of.
    buffer: ID3D12Resource,
    /// Where in `buffer` it begins (the footprints count from there).
    base: u64,
    /// Back in the pool with the last list that names it.
    _piece: Option<crate::pool::StagingPiece>,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice, of the title's texture.
    mips: u32,
    /// The first of them that is read and uploaded
    /// (`crate::resource::held_only`), the one the host copy begins at, and
    /// the bytes of title memory read.
    first: u32,
    level0: u32,
    read: usize,
    /// Which host copy it uploads into, and what that is told to hold.
    serial: u64,
    holds: Arc<crate::resource::Holds>,
    /// Made by the pass over textures picked by index, for one upload
    /// ([`once`]): read when the list is submitted, and never looked at again.
    once: std::sync::atomic::AtomicBool,
    /// The staging buffer's bytes (counted in `STAGED` while it lives).
    bytes: u64,
    memory: *const u8,
    last: Mutex<Last>,
    /// Whether the staging buffer was detiled since a submitted list last
    /// uploaded it, and whether one has uploaded it at all: a list records
    /// the upload only then. (Every list that sampled a title texture
    /// uploaded all of it again, changed or not: 5,411 subresource copies
    /// of 1,120 textures in one frame of run 18.)
    dirty: std::sync::atomic::AtomicBool,
    uploaded: std::sync::atomic::AtomicBool,
    /// When a list last recorded a use of it (`crate::host::moment`).
    used: std::sync::atomic::AtomicU64,
    /// The frame until which lists copy the texture changed or not
    /// ([`Last::copy_until`]), for [`upload`], which runs on the recording
    /// threads and does not take `last`.
    hot: std::sync::atomic::AtomicU64,
    /// Copies lists recorded since the last refresh only because the
    /// texture was hot, and how many times it was detiled so far: when the
    /// refresh after them detiles nothing, each of them copied what the
    /// host texture held already.
    forced: std::sync::atomic::AtomicU32,
    fills: std::sync::atomic::AtomicU32,
    /// How many bytes of the texture's memory were mapped when a list last
    /// recorded an upload of it ([`fill_unlocked`]; 0: not asked).
    expected: std::sync::atomic::AtomicU64,
    /// Levels on demand (`crate::tiles`): the texture's host copy was made
    /// without memory of its own when the staging buffer was made.
    on_demand: bool,
    /// For such a texture: the levels that were the title's at the last
    /// detiling (a bit a level; the staging buffer holds zeros for the
    /// others), and what that was judged by ([`Staging::judgement`]).
    detiled: std::sync::atomic::AtomicU32,
    judged: [std::sync::atomic::AtomicU64; 2],
    /// For such a texture: when an upload last left a held level without
    /// tiles (`crate::host::moment`, never 0; 0: none did). The upload is
    /// recorded again about a second later if the card has room by then
    /// ([`upload`]).
    waiting: std::sync::atomic::AtomicU64,
    /// Recorded for a one-shot decoration bake (`BAKING`, with
    /// `VICTORY_BAKE_SYNC=1`): the check at submission waits for title
    /// memory to stop remapping and takes none of its shortcuts.
    exact: std::sync::atomic::AtomicBool,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<Digest>,
    asked: u64,
    /// The frame a partly mapped texture was last read in.
    partial: Option<u64>,
    /// Until this frame a guarded texture is hashed once a frame from its
    /// first check on, for the ones written behind the guard from the start
    /// (the six-star screen of the first seconds, which the hash never
    /// caught changing). 0: not checked under the guard yet.
    started: u64,
    /// Until this frame the texture is hashed at every check: its memory
    /// changed behind the guard's back (a write through another view of
    /// the same pages, as a video decoder's planes get) when last hashed.
    hot_until: u64,
    /// The frame the hash last found the texture changed in. The texture's
    /// next check hashes it once more (`verify`): if it changed again it is
    /// hot. A texture that streams in changes once and is not hashed on. A
    /// video's planes cycle through a ring of textures, each sampled only
    /// every third frame, so "again" cannot mean "the next frame".
    changed_at: u64,
    verify: bool,
    /// The frame the texture was last hashed on its turn or in its first
    /// frames: either is a sample, and the frame's later checks pass it over.
    turned: u64,
}

/// Counts `size` bytes hashed, by `why` (see `Staging::refresh_timed`).
fn count_hashed(size: usize, why: u8) {
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::TexHashed, size as u64);
        match why {
            1 => add(C::TexHashedTurn, size as u64),
            2 => add(C::TexHashedHot, size as u64),
            3 => {
                add(C::TexHashedHot, size as u64);
                add(C::TexHashedFirst, size as u64);
            }
            _ => {}
        }
    }
    let _ = (size, why);
}

/// How many frames a texture that keeps changing stays hashed at every
/// check after its last change, and a guarded texture once a frame from its
/// first check.
const HOT_FRAMES: u64 = 64;

/// Every how many frames a partly mapped texture is read while the guards
/// see no write to it (reading around unmapped pages is slow).
const PARTIAL_EVERY: u64 = 30;

/// Whether a partly mapped texture last read in frame `at` is read again in
/// `frame`: once a frame while the guards see writes to it, else every
/// [`PARTIAL_EVERY`]th frame. A texture streamed into pages mapped with
/// another's bytes was read with those, and kept them for up to 30 frames
/// after its own texels landed: blocks of noise on the shirt of a close-up
/// (dc-d frame 2469, detile-audit-tell frame 2639).
fn partial_due(at: Option<u64>, frame: u64, written: bool) -> bool {
    at.is_none_or(|at| if written { frame != at } else { frame >= at + PARTIAL_EVERY })
}
/// How long after a change the next check still counts as "again".
const VERIFY_FRAMES: u64 = 24;

impl Last {
    /// The hash found the texture changed in `frame`.
    fn changed(&mut self, frame: u64) {
        if (self.verify && frame <= self.changed_at + VERIFY_FRAMES) || frame < self.hot_until {
            self.hot_until = frame + HOT_FRAMES;
        }
        self.changed_at = frame;
        self.verify = true;
    }

    /// The hash found the texture as it was in `frame`. With `repeat`, a
    /// check in the frame right after the change leaves the verifying on:
    /// the frame may sample the same video frame again.
    fn unchanged(&mut self, frame: u64, repeat: bool) {
        if repeat && frame <= self.changed_at + 1 {
            return;
        }
        self.verify = false;
    }

    /// Whether the hash is due at `frame` whatever the turns say.
    fn hash_due(&self, frame: u64) -> bool {
        frame < self.hot_until || (self.verify && frame > self.changed_at && frame <= self.changed_at + VERIFY_FRAMES)
    }

    /// The frame until which the next check may hash the texture again
    /// ([`hash_due`](Self::hash_due)). The first frames' hashing is left
    /// out: every texture has them.
    fn due_until(&self) -> u64 {
        let verifying = if self.verify { self.changed_at + VERIFY_FRAMES + 1 } else { 0 };
        self.hot_until.max(verifying)
    }

    /// The frame until which every list that samples the texture records
    /// its copy, changed or not ([`hot_level`] 2). With `verifying`, also in
    /// the frames after a single change ([`hot_copy`]).
    fn copy_until(&self, verifying: bool) -> u64 {
        if verifying { self.due_until() } else { self.hot_until }
    }
}

/// How a guarded texture that changes behind the guard's back is followed
/// (`VICTORY_TEXTURE_HOT`). A title that writes such a texture through
/// another view of its pages (the guard sees none of it) every frame, a
/// video's frames in a ring of three, showed each one up to eight frames
/// late and in turns: the boot logo video and the loading animation
/// flickered between old and new pictures (fx-flicker run 1441). 0: on its
/// turns only (every eighth frame), as before; 1: hashed at every check
/// after a change the guard did not see and while it keeps changing
/// ([`Last::hash_due`]), and once a frame for its first frames
/// ([`HOT_FRAMES`], [`first_once`]); 2 (default): as
/// 1, a check in the frame right after a change does not end the
/// verifying, and while the next check may hash the texture every list
/// that samples it records the copy into its host texture.
fn hot_level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var_os("VICTORY_TEXTURE_HOT") {
        Some(v) if v == "0" => 0,
        Some(v) if v == "1" => 1,
        _ => 2,
    })
}

/// Whether a texture that changed once is copied by every list that samples
/// it while its next checks verify the change (`VICTORY_TEXTURE_HOT_COPY=1`,
/// as before), or only once it is hot (0): found changing again within
/// [`VERIFY_FRAMES`], or behind the guard's back. Every texture that streams
/// in changes once, and in the opening of the game many do in a frame;
/// each was copied again by every list for 24 frames. With 0 a second
/// change is copied by the next list after its check, a frame late at most,
/// and makes the texture hot, so a video's frames are copied as before
/// from then on.
fn hot_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_HOT_COPY").is_none_or(|v| v != "0"))
}

/// Whether a texture in its first frames, not changing, is hashed once a
/// frame rather than at every check (`VICTORY_TEXTURE_FIRST_ONCE=0`, as
/// before). Every texture streamed in has these frames, and each list that
/// samples one checked it: 100-158 MB a frame hashed while due against
/// 18-48 MB changed in the first gameplay seconds of run play-b, up to
/// 50 ms of texture time a frame. A change made after the frame's hash is
/// found by the next frame's.
fn first_once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_FIRST_ONCE").is_none_or(|v| v != "0"))
}

/// Every how many frames a guarded texture that was never found changing
/// again is hashed on its turn (`VICTORY_TEXTURE_TURN`). One that was keeps
/// every eighth frame. 8 as before when bit 1 of `VICTORY_PAGE_WRITES` is
/// off; 64 when it is on, since DirectStorage's writes then open the
/// texture's blocks and the guard sees them: in runs play-d to play-f
/// (`all`) the turns hashed 4.5-6 MB a frame in the open world and found
/// no change there, one in each run in the menus. A texture is hashed once
/// a frame in its first frames anyway ([`HOT_FRAMES`]).
fn turn_every() -> u64 {
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let set = std::env::var("VICTORY_TEXTURE_TURN").ok().and_then(|v| v.parse().ok());
        set.unwrap_or(if crate::watch::page_writes() & 1 != 0 { 64 } else { 8 }).max(1)
    })
}

/// Where in the `every` frames a texture's turn falls. By its 64 KB block
/// alone, big textures at aligned addresses share one turn: in face-prio
/// 26 MB were hashed on every frame 62 of 64 and about 0.5 MB on the
/// others. `VICTORY_TEXTURE_TURN_SPREAD=1` mixes the page number instead,
/// so the same bytes spread over all the frames.
fn turn_phase(start: usize) -> u64 {
    static SPREAD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *SPREAD.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_TURN_SPREAD").is_some_and(|v| v == "1")) {
        (start as u64 >> 12).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 40
    } else {
        (start >> 16) as u64
    }
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Every staging state, grouped by the address where its texture's title
/// memory starts. [`STAGING`] is keyed by resource, which cannot answer
/// "which textures lie in this memory range?"; this index can, so a remap
/// does not have to scan every staged texture.
///
/// Several textures can start at the same address (for example two
/// formats viewing the same memory), hence the `Vec`. Entries are `Weak`
/// so the index never keeps a staging buffer alive on its own; entries
/// whose staging state is gone are removed by [`prune_address_index`].
static STAGED_BY_ADDRESS: Mutex<std::collections::BTreeMap<usize, Vec<Weak<Staging>>>> =
    Mutex::new(std::collections::BTreeMap::new());

/// The size in bytes of the largest texture ever staged. A texture that
/// overlaps some memory range cannot start more than this many bytes
/// before the range, which bounds the search in [`remapped`].
static LARGEST_STAGED_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Adds a newly made staging state to [`STAGED_BY_ADDRESS`].
fn add_to_address_index(staging: &Arc<Staging>) {
    let (start, bytes) = staging.memory();
    LARGEST_STAGED_BYTES.fetch_max(bytes, std::sync::atomic::Ordering::Relaxed);
    let mut index = STAGED_BY_ADDRESS.lock().unwrap_or_else(|e| e.into_inner());
    let entries = index.entry(start).or_default();
    // A texture staged here before may have been let go since.
    entries.retain(|entry| entry.strong_count() > 0);
    entries.push(Arc::downgrade(staging));
}

/// Removes entries of [`STAGED_BY_ADDRESS`] whose staging state was let go.
fn prune_address_index() {
    let mut index = STAGED_BY_ADDRESS.lock().unwrap_or_else(|e| e.into_inner());
    index.retain(|_, entries| {
        entries.retain(|entry| entry.strong_count() > 0);
        !entries.is_empty()
    });
}

/// Called when the runtime maps or unmaps title memory `[at, at + len)`.
///
/// A texture in that range may now show completely different bytes, but
/// its staging state still says "uploaded and unchanged". In that state
/// [`upload`] skips the copy and only schedules a check, so the first list
/// that samples the texture after the remap reads the old texels, and the
/// new ones only arrive one list later (#574).
///
/// Marking the texture dirty makes its next [`upload`] record the copy.
/// The check that runs when that list is submitted reads the new memory
/// into the staging buffer before the copy executes, so the very first
/// list after the remap sees the new texels.
pub(crate) fn remapped(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    for staging in staged_under(at, at.saturating_add(len)) {
        staging.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The staging states whose texture memory overlaps `[at, end)`.
fn staged_under(at: usize, end: usize) -> Vec<Arc<Staging>> {
    if end <= at {
        return Vec::new();
    }
    let earliest_start = at.saturating_sub(LARGEST_STAGED_BYTES.load(std::sync::atomic::Ordering::Relaxed));
    let index = STAGED_BY_ADDRESS.lock().unwrap_or_else(|e| e.into_inner());
    // Textures that start before `end`, and are not so far below `at` that
    // they cannot reach it...
    index
        .range(earliest_start..end)
        .flat_map(|(_, entries)| entries.iter().filter_map(Weak::upgrade))
        // ...of which only those whose memory reaches past `at` overlap the
        // range.
        .filter(|staging| {
            let (start, bytes) = staging.memory();
            start.saturating_add(bytes) > at
        })
        .collect()
}

/// `VICTORY_TABLED_REREAD`: whether a texture draws read through a table
/// is read again whole once a write the guards do not see (DirectStorage's
/// reads into physical pages) has settled in its memory after its upload.
///
/// Such a texture keeps its staging buffer and is checked at every
/// submission that samples it ([`Staging::refresh`]). That check skips the
/// read while the guards saw no write to the texture's blocks, unless it is
/// the texture's turn ([`turn_every`]) or the texture changed lately. A
/// write into the physical pages through the runtime's own view does not
/// fault at the guard: the check sees it only if bit 1 of
/// `VICTORY_PAGE_WRITES` opened the blocks, or at the texture's turn. The
/// texture pass, which makes slots of textures picked by index due again
/// after such writes, leaves out the slots draws read through tables
/// (`crate::descriptor::host` `slots_written_under`, `tabled_lately`), and
/// drops a table's slot from its rechecks (`crate::draw` `heap_textures`).
///
/// On (default): a staged texture that was uploaded when such a write
/// landed in its memory settles as those slots do
/// (`crate::descriptor::host::page_settle`) and is then read again whole:
/// what its staging buffer was filled from is forgotten, so the check at
/// the next submission that samples it reads and detiles the memory, and
/// the list uploads it. A texture uploaded while its memory was all zero,
/// before DirectStorage brought its texels, is one of these. At most
/// [`REREAD_MOST`] textures and [`REREAD_MOST_BYTES`] of their memory a
/// frame; the rest wait for the next frame. `0`: left to their own checks,
/// as before.
pub(crate) fn tabled_reread() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TABLED_REREAD").is_none_or(|v| v != "0"))
}

/// Staged textures one frame reads again at most ([`tabled_reread`]).
/// shortcut: not a measured value; the same as the slots one pass makes due
/// (`crate::descriptor::host` `SETTLE_MOST`).
const REREAD_MOST: usize = 64;

/// The bytes of title memory of the staged textures one frame reads again
/// at most ([`tabled_reread`]); the first is taken whatever its size.
/// shortcut: not a measured value. In the owner's session a slow frame
/// spent 193 ms preparing 198 MB of textures; 16 MB is a share of that size.
const REREAD_MOST_BYTES: u64 = 16 << 20;

/// Since the start ([`tabled_reread`]): staged textures found uploaded
/// under a write the guards do not see (once per written range); of those,
/// ones that were settling already; settled ones read again and their bytes
/// of title memory; settled ones let go of, or not uploaded, by then; and
/// how many are settling now.
pub(crate) static TABLED_REREAD: [std::sync::atomic::AtomicU64; 6] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// Things settling under writes the guards do not see, by key: the moments
/// (ms) of the first and of the last write noted under each, the bytes of
/// its memory, and what the pass acts on once its writes settled.
struct Settling<V> {
    map: std::collections::BTreeMap<usize, (u64, u64, u64, V)>,
}

impl<V> Settling<V> {
    const fn new() -> Self {
        Self {
            map: std::collections::BTreeMap::new(),
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    /// A write under `key` at `now`; `value` for a key not settling yet.
    /// Whether it was settling already.
    fn note(&mut self, key: usize, now: u64, bytes: u64, value: impl FnOnce() -> V) -> bool {
        if let Some(entry) = self.map.get_mut(&key) {
            entry.1 = now;
            return true;
        }
        self.map.insert(key, (now, now, bytes, value()));
        false
    }

    /// Takes the ones whose writes settled by `now`: no write for `quiet`
    /// ms, or `longest` ms since the first. Oldest first, at most `most` of
    /// them and `most_bytes` of their memory (the first whatever its size);
    /// the rest stay for the next call.
    fn settled(&mut self, now: u64, quiet: u64, longest: u64, most: usize, most_bytes: u64) -> Vec<V> {
        let mut ready: Vec<(u64, usize, u64)> = self
            .map
            .iter()
            .filter(|(_, e)| now.saturating_sub(e.1) >= quiet || now.saturating_sub(e.0) >= longest)
            .map(|(&key, e)| (e.0, key, e.2))
            .collect();
        ready.sort_unstable();
        let mut taken = Vec::new();
        let mut bytes = 0u64;
        for (_, key, size) in ready {
            if taken.len() >= most || (!taken.is_empty() && bytes.saturating_add(size) > most_bytes) {
                break;
            }
            bytes = bytes.saturating_add(size);
            if let Some((.., value)) = self.map.remove(&key) {
                taken.push(value);
            }
        }
        taken
    }
}

/// Staged textures settling under writes the guards do not see
/// ([`tabled_reread`]), by the address of their staging state; the `Weak`
/// keeps that allocation, so no other state takes the key meanwhile.
static SETTLING_STAGED: Mutex<Settling<Weak<Staging>>> = Mutex::new(Settling::new());

/// The frame [`page_settled`] last ran in.
static SETTLED_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// Title memory `ranges` (start, end) was written behind the guards' back
/// (taken by the texture pass from `crate::descriptor::host::page_written`)
/// at `now` (ms): the staged textures there that were uploaded settle
/// ([`tabled_reread`]). One not uploaded yet takes the bytes that are there
/// by its upload.
pub(crate) fn page_written(ranges: &[(u64, u64)], now: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let found: Vec<Arc<Staging>> = ranges
        .iter()
        .flat_map(|&(start, end)| staged_under(start as usize, end as usize))
        .filter(|staging| staging.uploaded.load(Relaxed))
        .collect();
    if found.is_empty() {
        return;
    }
    let mut settling = SETTLING_STAGED.lock().unwrap_or_else(|e| e.into_inner());
    let mut again = 0;
    for staging in &found {
        let bytes = staging.memory().1 as u64;
        if settling.note(Arc::as_ptr(staging) as usize, now, bytes, || Arc::downgrade(staging)) {
            again += 1;
        }
    }
    TABLED_REREAD[0].fetch_add(found.len() as u64, Relaxed);
    TABLED_REREAD[1].fetch_add(again, Relaxed);
    TABLED_REREAD[5].store(settling.len() as u64, Relaxed);
}

/// Once in `frame`: the staged textures whose writes settled by `now` (ms;
/// no write for `quiet` ms, or `longest` since the first; at most
/// [`REREAD_MOST`] and [`REREAD_MOST_BYTES`]) and that are still uploaded
/// are read again whole at the next submission that samples them, and that
/// submission's list uploads them ([`tabled_reread`]). Returns how many.
pub(crate) fn page_settled(frame: u64, now: u64, quiet: u64, longest: u64) -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    if SETTLED_FRAME.swap(frame, Relaxed) == frame {
        return 0;
    }
    let settled = {
        let mut settling = SETTLING_STAGED.lock().unwrap_or_else(|e| e.into_inner());
        if settling.map.is_empty() {
            return 0;
        }
        let settled = settling.settled(now, quiet, longest, REREAD_MOST, REREAD_MOST_BYTES);
        TABLED_REREAD[5].store(settling.len() as u64, Relaxed);
        settled
    };
    let (mut read, mut bytes, mut gone) = (0, 0, 0);
    for staging in settled {
        match staging.upgrade() {
            Some(staging) if staging.uploaded.load(Relaxed) => {
                // Forgotten, the next check reads and detiles the memory
                // whatever the guards say; dirty, the next list that
                // samples the texture records its upload.
                staging.forget();
                staging.dirty.store(true, Relaxed);
                read += 1;
                bytes += staging.memory().1 as u64;
            }
            _ => gone += 1,
        }
    }
    TABLED_REREAD[2].fetch_add(read, Relaxed);
    TABLED_REREAD[3].fetch_add(bytes, Relaxed);
    TABLED_REREAD[4].fetch_add(gone, Relaxed);
    read as usize
}

/// The buffer a texture the host writes is copied through into the texture
/// of a view the title placed in its memory in another format
/// ([`alias`]): every level of every layer as each of the two formats lays
/// it out for a copy (the same bytes: the texels are as wide), the written
/// texture's layout first. No buffer when the two formats copy straight
/// into each other ([`direct`]).
struct Through {
    buffer: Option<ID3D12Resource>,
    parts: Vec<(
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    )>,
}

/// The most levels times layers [`alias`] copies (a cube of nine levels).
const ALIAS_PARTS: u32 = 54;

// SAFETY: a device-local buffer only lists record copies with.
unsafe impl Send for Through {}
// SAFETY: as above.
unsafe impl Sync for Through {}

/// Whether [`alias`] copies texture to texture, without the buffer, when
/// the host may copy the written format into the read one as it is
/// (`VICTORY_ALIAS_DIRECT=1`; off by default).
///
/// D3D12 copies the bits of R32_UINT and R32_SINT into R9G9B9E5_SHAREDEXP
/// and back, so those need no buffer. They are every alias copy of the
/// leafy-play run, 81 a frame for 8.4 ms of card time (gpt-a1, stamped):
/// two copies and two buffer barriers each, where one copy does. The bits
/// land the same, so the picture should not change.
fn direct(
    written: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    read: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
) -> bool {
    use windows::Win32::Graphics::Dxgi::Common::*;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var_os("VICTORY_ALIAS_DIRECT").is_some_and(|v| v == "1"));
    let word = |f| f == DXGI_FORMAT_R32_UINT || f == DXGI_FORMAT_R32_SINT;
    let shared = |f| f == DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    on && (word(written) && shared(read) || shared(written) && word(read))
}

/// The copy buffers of [`alias`], by the placed view's texture.
static THROUGH: Mutex<Option<HashMap<usize, Arc<Through>>>> = Mutex::new(None);

/// Brings `placed`, the texture of a view the title placed in the memory
/// of `source` in another format (`crate::resource::alias_of`), up to date
/// with what the host wrote to `source`: the texels of every level and
/// layer are copied as they are, through a buffer (the host copies between
/// most formats of one texel width in no other way), or straight across
/// with [`direct`]. Recorded the first time
/// the list reads the view
/// and again whenever the list used `source` in another way since (wrote
/// it, or read it itself). Leaves `placed` in the copy destination state,
/// as [`upload`] does. False if the copy cannot be made: the caller fills
/// the texture from title memory, as before. The caller holds `source`
/// (`crate::resource::held_alias_of`); both host copies are held here until
/// the copy is recorded.
pub(crate) fn alias(list: &HostList, placed: This, source: This) -> bool {
    const TARGET_OR_UNORDERED: u32 = 0x4 | 0x8;
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    // (All of its levels: the copy is level by level.)
    crate::resource::want_level(placed, 0);
    let (Some(to), Some(from)) = (
        crate::resource::host_copy(placed),
        crate::resource::host_copy(source),
    ) else {
        return false;
    };
    let (to, from) = (&to.texture, &from.texture);
    let Some(host) = host() else { return false };
    let through = {
        let mut map = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(HashMap::new);
        match map.get(&(placed as usize)) {
            Some(through) => through.clone(),
            None => {
                // SAFETY: live textures.
                let (of_from, of_to) = unsafe { (from.GetDesc(), to.GetDesc()) };
                // Levels times layers (a volume's slices are one part).
                let count = |d: &D3D12_RESOURCE_DESC| {
                    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                        1
                    } else {
                        u32::from(d.DepthOrArraySize)
                    };
                    u32::from(d.MipLevels) * layers
                };
                let parts = count(&of_from);
                if parts == 0 || parts > ALIAS_PARTS || parts != count(&of_to) {
                    return false;
                }
                let mut rows = [(); 2]
                    .map(|()| vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); parts as usize]);
                let mut bytes = [0u64; 2];
                // SAFETY: plain queries; the out pointers are ours and
                // hold `parts` layouts each.
                unsafe {
                    host.device.GetCopyableFootprints(
                        &of_from,
                        0,
                        parts,
                        0,
                        Some(rows[0].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[0]),
                    );
                    host.device.GetCopyableFootprints(
                        &of_to,
                        0,
                        parts,
                        0,
                        Some(rows[1].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[1]),
                    );
                }
                let alike = rows[0].iter().zip(&rows[1]).all(|(a, b)| {
                    let (fa, fb) = (a.Footprint, b.Footprint);
                    a.Offset == b.Offset
                        && (fa.Width, fa.Height, fa.Depth, fa.RowPitch)
                            == (fb.Width, fb.Height, fb.Depth, fb.RowPitch)
                });
                if bytes[0] == 0 || bytes[0] != bytes[1] || !alike {
                    return false;
                }
                let buffer = if direct(of_from.Format, of_to.Format) {
                    None
                } else {
                    let Some(buffer) = host.uav_buffer(bytes[0]) else {
                        return false;
                    };
                    crate::host::name(&buffer, || {
                        format!(
                            "alias copy {}x{} f{} to f{}",
                            of_from.Width, of_from.Height, of_from.Format.0, of_to.Format.0
                        )
                    });
                    Some(buffer)
                };
                tracing::info!(
                    view = ?placed,
                    texture = ?source,
                    width = of_from.Width,
                    height = of_from.Height,
                    parts,
                    written_as = of_from.Format.0,
                    read_as = of_to.Format.0,
                    direct = buffer.is_none(),
                    "a view placed in the memory of a texture the host writes, in another format: read from that texture by a copy on the host"
                );
                let [of_written, of_read] = rows;
                let through = Arc::new(Through {
                    buffer,
                    parts: of_written.into_iter().zip(of_read).collect(),
                });
                map.insert(placed as usize, through.clone());
                through
            }
        }
    };
    // An upload from title memory can have already used this view. Only
    // a copy from the same GPU source can satisfy this read.
    list.first_use(placed as usize);
    let new_source = list.alias_source_changed(to, from);
    // The source's first use in the list records nothing (its prelude
    // brings it to the state); a use in another way since the last copy
    // shows as a change of state.
    let before = list.transition(source, COPY_SOURCE).map(|(texture, before, state)| {
        let barrier = crate::host::transition(
            texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(state as i32),
        );
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
        before
    });
    if !new_source && before.is_none() {
        return true;
    }
    card_writes(list, placed, to, None);
    crate::list::ensure_state(list, placed, COPY_DEST);
    let part = |texture: &ID3D12Resource, index: usize| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index as u32,
        },
    };
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("alias copy of {source:?} into {placed:?}"));
    let Some(buffer) = &through.buffer else {
        for index in 0..through.parts.len() {
            // SAFETY: an open list and live textures whose formats the
            // host copies into each other ([`direct`]), part for part alike.
            unsafe {
                list.list
                    .CopyTextureRegion(&part(to, index), 0, 0, 0, &part(from, index), None);
            }
        }
        if let Some(before) = before
            && before & TARGET_OR_UNORDERED != 0
        {
            crate::list::ensure_state(list, source, before);
        }
        return true;
    };
    let rows = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, read, written) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer is as big as
    // the footprints and rests in the common state between copies.
    unsafe {
        list.list
            .ResourceBarrier(&[crate::host::transition(buffer, common, written)]);
        for (index, (of_written, _)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&rows(of_written), 0, 0, 0, &part(from, index), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(buffer, written, read)]);
        for (index, (_, of_read)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&part(to, index), 0, 0, 0, &rows(of_read), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(buffer, read, common)]);
    }
    list.keep(buffer);
    // A draw that also writes the source (a view of it handled before
    // this one) needs it in that state again; the next read of the alias
    // then copies once more, as it must after a write.
    if let Some(before) = before
        && before & TARGET_OR_UNORDERED != 0
    {
        crate::list::ensure_state(list, source, before);
    }
    true
}

/// Whether a texture refresh that finds the runtime remapping title memory
/// goes on without waiting for it (`VICTORY_TEXTURE_NO_WAIT=1`; off by
/// default).
///
/// A texture whose memory is all mapped is read in place, under a lock
/// the runtime takes alone while it changes a mapping
/// (`crate::watch::title_memory_reading`), so that no page goes away under
/// the read. While the world streams in the runtime remaps all the time,
/// and each of the frame's 150-360 refreshes waited for it: runs 99-101
/// have 4, 15 and 16 frame lines with over 300 ms of "textures", up to
/// 3,708 ms, and on every one of them the wait for this lock is the whole
/// of it (1,345 of 1,381 ms; 3,630 of 3,708; the write-watch askings 3-12
/// ms, the texture's own lock 0).
///
/// With the switch the refresh asks for the lock without waiting. If the
/// runtime has it, the texture is read through the kernel instead, which
/// needs no lock: it fails if a page is not there, and then the texture is
/// left as it is and read at its next refresh, a frame later. A read
/// through the kernel costs about 0.6 ms a megabyte against 0.3 in place.
///
/// What changes for the picture: a texture that arrives while the runtime
/// is remapping may show a frame later. A staging buffer that holds nothing
/// yet is filled with what is there all the same ([`fill_unlocked`]): a list
/// has recorded the upload from it by then.
fn no_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_NO_WAIT").is_some_and(|v| v == "1"))
}

/// Whether a refresh that does not wait ([`no_wait`]) and finds a page
/// missing fills a staging buffer that was never filled from the pages that
/// are there (`VICTORY_TEXTURE_FILL=0`: it was left as it was made).
///
/// Left for the next refresh, it was uploaded as it was: the upload is
/// recorded before the refresh, and a texture picked by index has no next
/// refresh (its staging buffer goes with the upload). A texture the title
/// holds only the smaller levels of is never all there, so whenever the
/// runtime was remapping at that moment, its levels became zeros on the
/// card, good ones too: of 16,135 textures a recorded frame's slots show,
/// 2,382 had zeros in the first level shown; 137 of that frame's 787 repeated
/// uploads put zeros over a filled level (run pic1-1; 68,700 refreshes left
/// in the 150 s of world of run play1).
fn fill_unlocked() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_FILL").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line, with [`no_wait`]: refreshes that read
/// through the kernel because title memory was being remapped, and those
/// left for the next refresh because a page was not there; staging buffers
/// filled from the pages there ([`fill_unlocked`]), and of them those that
/// got less than was mapped (their slots are looked at again).
pub(crate) static UNLOCKED: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Since the last frame-rate line, exact refreshes (for a bake) that found
/// the runtime remapping title memory ([`exact_unlocked`]): read whole
/// through the kernel without waiting, and those that had to wait (a page
/// was not there).
pub(crate) static EXACT_UNLOCKED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether an exact refresh (the uploads before a bake, `BAKING`) that
/// finds the runtime remapping title memory first reads the texture whole
/// through the kernel, and waits for the remapping to end only if a page of
/// it is not there (`VICTORY_EXACT_NO_WAIT=0`: it always waits, as before).
/// A read through the kernel that succeeds got every byte as title memory
/// holds it at that moment, which is what waiting gave; it fails, rather
/// than reading something else, where a page is away. Waiting made the
/// texture refreshes of a cutscene second 866 ms a frame, 808 of them
/// waiting for title memory to hold still (diag run of 2026-11-26 15:58,
/// with `VICTORY_TEXTURE_NO_WAIT=1`, where only exact refreshes wait).
fn exact_unlocked() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_EXACT_NO_WAIT").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line, of texture refreshes on every thread, in
/// microseconds: waiting for the texture's own lock (another thread is
/// refreshing the same texture), asking the write watch and the guards,
/// waiting for title memory to hold still (the lock the runtime takes
/// alone while it remaps, `crate::watch::title_memory_reading`), and the
/// refreshes in all.
///
/// While the world streams in, the frame line's "textures" is seconds a
/// frame in some runs and tenths in others for the same megabytes (2,566
/// ms for 110 MB in run 93, 314 ms for 132 MB in run 91; before these
/// counters, runs 81, 83, 85 and 94 each had lines of 1.5-2.9 s), and the
/// submitting thread is not running while it lasts (run 93: 240-310 ms of
/// its own time a frame beside 1,100-1,800 ms of "textures"): it waits.
/// These say for what.
pub(crate) static REFRESH_WAITS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// `f`'s result, its time added to [`REFRESH_WAITS`]' place `slot`.
fn waited<T>(slot: usize, f: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = f();
    REFRESH_WAITS[slot].fetch_add(started.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    out
}

thread_local! {
    /// Room to read a texture's memory into: one buffer per submitting
    /// thread. Each texture kept its own, a second copy in memory of every
    /// write-watched or partly mapped texture ever sampled, for the run.
    static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Calls `f` with `size` bytes of this thread's read buffer (their
/// contents are whatever the last texture left).
fn with_read<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    READ.with(|read| {
        let mut read = read.borrow_mut();
        if read.len() < size {
            read.resize(size, 0);
        }
        f(&mut read[..size])
    })
}

/// Whether a texture all of whose memory is mapped is read through the
/// kernel's copy instead of in place (`VICTORY_TEXTURE_COPY=1`; off by
/// default). Round 3 of the memory work read every such texture that way
/// to be safe from pages taken away under the read; it cost 0.6 ms per MB
/// against 0.3 in place (90-105 ms a frame while the world streams in, runs
/// 72 and 73, against 20-40), and both runs with it lost the graphics
/// device, for a reason not found. The read in place is safe now for
/// another reason (`crate::watch::title_memory_reading`); this switch is
/// left to tell whether the copy was what lost the device.
fn through_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_COPY").is_some_and(|v| v == "1"))
}

/// Calls `f` with the `size` bytes of title memory at `memory`, read
/// through the kernel into this thread's read buffer; pages that are not
/// there read as zeros. The bytes are those a read in place would see.
fn with_title<R>(memory: *const u8, size: usize, f: impl FnOnce(&[u8]) -> R) -> R {
    with_read(size, |read| {
        crate::host::copy_mapped(read.as_mut_ptr(), memory, size, false);
        f(read)
    })
}

/// Staging buffers alive (also those only a list still holds) and their
/// bytes.
static STAGED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Staging {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        STAGED[0].fetch_sub(1, Relaxed);
        STAGED[1].fetch_sub(self.bytes, Relaxed);
    }
}

thread_local! {
    /// On this thread: staging buffers made, the microseconds that took
    /// (their upload buffers among them), and of those upload buffers the
    /// ones made new (not from the pool) and their microseconds.
    static STAGED_HERE: std::cell::Cell<[u64; 4]> = const { std::cell::Cell::new([0; 4]) };
}

/// (Neptune) What [`STAGED_HERE`] holds for the calling thread: the pass
/// over textures picked by index tells by these how much of its time went
/// into making staging buffers.
pub(crate) fn staged_here() -> [u64; 4] {
    STAGED_HERE.with(std::cell::Cell::get)
}

/// For the memory line: staging buffers alive, their bytes, and how many
/// of them their texture still has (the others wait for a list's reset).
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    let kept = STAGING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(0, |map| map.len() as u64);
    [STAGED[0].load(Relaxed), STAGED[1].load(Relaxed), kept]
}

/// Staging buffers let go as idle since the start, and their bytes.
static LET_GO: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// For the memory line: staging buffers let go as idle, and their bytes.
#[cfg(feature = "diagnostics")]
pub(crate) fn let_go() -> [u64; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    [LET_GO[0].load(Relaxed), LET_GO[1].load(Relaxed)]
}

/// Lets go of the staging buffers no list recorded a use of for `IDLE_MS`
/// and `IDLE_FRAMES` frames and no list holds: a staging buffer is system memory the size of its
/// texture, kept so that an unchanged texture is not uploaded again, and
/// those of textures no longer sampled were kept for the run (2,246 of
/// them, 2.1 GB, at the end of run 24). The next list that samples such a
/// texture makes a staging buffer anew, detiles and uploads once more.
/// Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    use std::sync::atomic::Ordering::Relaxed;
    const IDLE_FRAMES: u64 = 4;
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = map.as_mut() else {
        return;
    };
    if map.is_empty() {
        return;
    }
    let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    // A list that recorded an upload or a check holds the staging until it
    // is reset: one only the table holds is named by no list, recorded or
    // running.
    let (mut went, mut bytes) = (0u64, 0u64);
    map.retain(|_, s| {
        let keep = Arc::strong_count(s) > 1
            || !crate::host::long_ago(s.used.load(Relaxed), idle_ms, IDLE_FRAMES);
        if !keep {
            LET_GO[0].fetch_add(1, Relaxed);
            LET_GO[1].fetch_add(s.bytes, Relaxed);
            went += 1;
            bytes += s.bytes;
        }
        keep
    });
    if went > 0 {
        prune_address_index();
        crate::host::trail(|| format!("sweep: {went} staging buffers of textures let go ({} KiB), held by no list", bytes >> 10));
    }
}

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let new = {
        let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        set.get_or_insert_with(Default::default).insert(resource as usize)
    };
    // Diagnostics (run 177): where its title memory is, for placed views
    // the title reads that memory through ([`placed_in_host_written`]).
    if new && let Some((at, desc)) = crate::resource::texture_memory(resource) {
        // SAFETY: 56 bytes in the description's layout.
        let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
        let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
        if !at.is_null() && bytes > 0 {
            HOST_WRITTEN_AT
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(at as u64, (bytes, words[4], words[6], words[7], words[8], resource as usize));
        }
    }
}

/// The title memory of textures the host copied into, by address: bytes
/// and description words 4, 6, 7 and 8 (width, height, layers and levels,
/// format).
static HOST_WRITTEN_AT: Mutex<std::collections::BTreeMap<u64, (u64, u32, u32, u32, u32, usize)>> =
    Mutex::new(std::collections::BTreeMap::new());

/// A texture the title placed at `address` (description words `desc`) was
/// made: says so once if it lies in the memory of a texture the host
/// copied into ([`host_written`]). Its own host copy is filled from title
/// memory, which the copy never reached: the copied texels are on the
/// card in the other texture, where no shader that reads this view finds
/// them.
pub(crate) fn placed_in_host_written(address: u64, desc: &[u32; 14]) {
    let hit = {
        let map = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
        map.range(..=address)
            .next_back()
            .filter(|(at, d)| address < **at + d.0)
            .map(|(at, d)| (*at, *d))
    };
    if let Some((at, (bytes, width, height, layers_levels, format, _))) = hit {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::warn!(
                view_address = format_args!("{address:#x}"),
                view_offset = address - at,
                view_width = desc[4],
                view_height = desc[6],
                view_layers_levels = format_args!("{:#x}", desc[7]),
                view_format = desc[8],
                copied_address = format_args!("{at:#x}"),
                copied_bytes = bytes,
                copied_width = width,
                copied_height = height,
                copied_layers_levels = format_args!("{layers_levels:#x}"),
                copied_format = format,
                "HERO LEAD: a placed texture view lies in the memory of a texture the host copied into: it is filled from title memory, which the copy never reached"
            );
        }
    }
}

/// Whether the host GPU wrote `resource` by a copy or resolve
/// ([`host_written`]): its host copy cannot be made again from title
/// memory.
pub(crate) fn host_wrote(resource: usize) -> bool {
    HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&resource))
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    // (Neptune) And what a prepare thread made ready for it: its memory goes.
    prepare::forget(resource);
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
    // Its memory may be given to other textures now (run 177: most of the
    // "HERO LEAD" lines were views in memory a released texture had held).
    HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, d| d.5 != resource);
    let mut through = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(through) = through.as_mut() {
        through.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, _texture: &ID3D12Resource) {
    upload_from(list, resource, 0);
}

thread_local! {
    /// The thread records uploads for a draw that bakes from them.
    pub(crate) static BAKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The thread records the upload of a texture picked by index, of which
    /// that many bytes were found mapped just now (`u64::MAX`: not asked).
    pub(crate) static PICKING: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Whether a staging buffer made for one upload of a texture picked by index
/// is filled without asking the write watch and the guards
/// (`VICTORY_TEXTURE_ONCE=0`: with, as before). What they say is for the next
/// refresh, and there is none: the buffer goes with the upload. Asking guards
/// the texture's blocks (a change of protection, under the lock every other
/// refresh wants), and the title's next write there faults to open them.
fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_ONCE").is_none_or(|v| v != "0"))
}

/// A 128-bit digest of `bytes`, to tell a texture's memory changed: four
/// lanes of folded multiplications over 64 bytes at a time. (Not a
/// cryptographic one, which took 0.27 ms a megabyte, four times the detiling
/// it was to spare; nobody makes up textures to collide.)
pub(crate) fn digest(bytes: &[u8]) -> Digest {
    const K: [u64; 8] = [
        0xa076_1d64_78bd_642f,
        0xe703_7ed1_a0b4_28db,
        0x8ebc_6af0_9c88_c6e3,
        0x5899_65cc_7537_4cc3,
        0x1d8e_4e27_c47d_124f,
        0xeb44_acca_b455_d165,
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
    ];
    let fold = |a: u64, b: u64| {
        let p = u128::from(a) * u128::from(b);
        p as u64 ^ (p >> 64) as u64
    };
    let word = |b: &[u8]| {
        let mut w = [0; 8];
        w[..b.len()].copy_from_slice(b);
        u64::from_le_bytes(w)
    };
    let mut lanes = [K[0], K[1], K[2], K[3]];
    let mut step = |block: &[u8; 64]| {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let a = word(&block[16 * i..16 * i + 8]);
            let b = word(&block[16 * i + 8..16 * i + 16]);
            // (What went in is added again: a product with a factor of zero
            // forgets neither the lane nor the block.)
            *lane = fold(a ^ K[4 + i], b ^ *lane).wrapping_add(a.rotate_left(23) ^ lane.rotate_left(41)) ^ b;
        }
    };
    let mut blocks = bytes.chunks_exact(64);
    for block in &mut blocks {
        step(block.try_into().unwrap_or(&[0; 64]));
    }
    let rest = blocks.remainder();
    let mut last = [0u8; 64];
    last[..rest.len()].copy_from_slice(rest);
    step(&last);
    let len = bytes.len() as u64;
    let low = fold(lanes[0] ^ K[1], lanes[1] ^ len) ^ fold(lanes[2] ^ K[3], lanes[3] ^ K[5]);
    let high = fold(lanes[0].rotate_left(32) ^ K[6], lanes[2] ^ len) ^ fold(lanes[1].rotate_left(32) ^ K[7], lanes[3] ^ K[2]);
    Digest(u128::from(low) << 64 | u128::from(high))
}

/// See [`digest`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Digest(u128);

/// [`upload`] of the levels from `first` on, the ones a view shows: of a
/// texture the title placed there may be no memory of the larger ones, and
/// what lies there is another texture's. (After victory-amd 4436f8f1.)
pub(crate) fn upload_from(list: &HostList, resource: This, first: u32) {
    let first = if crate::resource::held_only() { first } else { 0 };
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written {
        return;
    }
    // Once a list for each first level; not again for one that the list's
    // upload of larger ones held too.
    if !list.first_use(1 << 63 | (resource as usize) << 4 | first as usize & 15) {
        return;
    }
    if !list.first_use(resource as usize) && staged_from(resource) <= first {
        return;
    }
    let Some(copy) = crate::resource::host_copy(resource) else {
        return;
    };
    let texture = &copy.texture;
    ahead::listed(&copy);
    let Some(staging) = staging(resource, &copy, first) else {
        return;
    };
    use std::sync::atomic::Ordering::Relaxed;
    let baking = BAKING.get() && crate::draw::bake_sync();
    if baking {
        staging.exact.store(true, Relaxed);
    }
    let picking = PICKING.get();
    if picking.is_none() {
        // A draw samples it through a table: kept and checked from here on.
        staging.once.store(false, Relaxed);
    }
    staging.used.store(crate::host::moment(), Relaxed);
    // Levels on demand: a view of the texture now starts at another
    // level, or another texture now starts inside its span, with no byte
    // of its memory changed. The levels are judged anew: detiled again at
    // this list's submission, and uploaded by this list.
    if staging.on_demand && staging.judgement() != staging.judged() {
        staging.forget();
        staging.dirty.store(true, Relaxed);
    }
    // Levels on demand: the last upload left a held level without tiles
    // (the card had no room, or no heap could be had), and nothing else
    // would record an upload again until the title writes the texture: the
    // level stayed empty, the texture blurred or black. Asked again about
    // once a second, and only when the card has room; the staging buffer
    // still holds the level's bytes.
    if staging.on_demand {
        let since = staging.waiting.load(Relaxed);
        if since != 0 && crate::host::long_ago(since, 1000, 2) {
            if crate::resource::room_lately() {
                staging.waiting.store(0, Relaxed);
                staging.dirty.store(true, Relaxed);
                crate::tiles::STATS[10].fetch_add(1, Relaxed);
            } else {
                staging.waiting.store(crate::host::moment().max(1), Relaxed);
            }
        }
    }
    // A plane the title's video decoder writes on the CPU: each list that
    // samples it copies it from title memory into its own scratch at
    // submission, so lists in flight never share a staging buffer and no
    // check hashes it.
    if movie_planes() && staging.movie_plane() {
        upload_plane(list, resource, texture, &staging);
        return;
    }
    // With `VICTORY_PRELUDE_UPLOADS=1`: whether to upload is decided when
    // the list is submitted, after the check that refreshes the staging
    // buffer, and the copy goes on the list's prelude ([`PreludeUpload`]).
    if prelude_uploads() && !baking {
        if no_wait() && fill_unlocked() {
            let there = match picking {
                Some(there) if there != u64::MAX => there,
                _ => mapped_bytes(staging.memory, staging.read),
            };
            staging.expected.store(there, Relaxed);
        }
        let on_demand = if staging.on_demand { crate::resource::host_tiled(resource) } else { None };
        list.defer_prelude_upload(PreludeUpload { staging, copy, on_demand });
        return;
    }
    // The host texture has the staging buffer's contents: only the check
    // at submission is left, and what it finds changed the next list that
    // samples the texture uploads (a frame late where every list of the
    // frame was recorded before that check).
    // A hot texture is copied by every list that samples it: its check
    // finds it changed, and the copy, which runs after the check, takes
    // what the check detiled.
    // A bake records the copy always, so what the check finds lands before
    // the bake draw runs.
    if !baking
        && staging.uploaded.load(Relaxed)
        && !staging.dirty.load(Relaxed)
        && !staging.hot_at(crate::frame::current())
    {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexSkipped, 1);
        list.defer_texture(staging);
        return;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        let bytes = staging.copied_bytes();
        add(C::TexUploads, 1);
        add(C::TexUploadBytes, bytes);
        if staging.uploaded.load(Relaxed) && !staging.dirty.load(Relaxed) {
            add(C::TexForced, 1);
            add(C::TexForcedBytes, bytes);
            staging.forced.fetch_add(1, Relaxed);
        }
    }
    if no_wait() && fill_unlocked() {
        let there = match picking {
            Some(there) if there != u64::MAX => there,
            _ => mapped_bytes(staging.memory, staging.read),
        };
        staging.expected.store(there, Relaxed);
    }
    record_upload(list, resource, texture, &staging, &staging.buffer, staging.base);
    // When the list is submitted, its copy takes what the staging buffer
    // holds by then: noted before the check below, which sets `dirty`
    // again if it detiles (the next list then uploads once more).
    {
        let staging = staging.clone();
        list.defer_host(move || {
            staging.uploaded.store(true, Relaxed);
            staging.dirty.store(false, Relaxed);
        });
    }
    // A host copy that took the place of one with fewer levels: the slots
    // that show that one are made again once this upload has run.
    if staging.first == staging.level0 && copy.grown.swap(false, Relaxed) {
        let start = staging.memory as u64;
        list.defer_host(move || crate::descriptor::host::levels_changed_after(start));
    }
    let noted = staging.clone();
    // Levels on demand (`crate::tiles`): after the check below, in the
    // order of this list's steps, and before the list is executed, the
    // levels the staging buffer holds as the title's are given the card's
    // memory if they have none. The list copies every level, so each one
    // given here is written whole by it (the rest are copies into no
    // memory, which the card drops).
    let on_demand = if staging.on_demand { crate::resource::host_tiled(resource) } else { None };
    let Some(on_demand) = on_demand else {
        // Once per submission however many of its lists sample the texture.
        list.defer_texture(staging);
        // Then what the list's copy puts in the host copy is known.
        list.defer_host(move || noted.note_held());
        return;
    };
    list.defer_texture(staging.clone());
    list.defer_host_checked(move || {
        let waiting = on_demand.uploaded(staging.detiled.load(Relaxed), staging.memory as u64);
        staging.waiting.store(if waiting { crate::host::moment().max(1) } else { 0 }, Relaxed);
        // A texture picked by index has no staging buffer after this
        // upload: its slots are made due again by the texture pass.
        if waiting {
            crate::descriptor::host::levels_waiting(staging.memory as u64);
        }
    });
}

/// Records the copy of every part of `staging` from `source`, its
/// footprints counted from `base`, into the host texture.
fn record_upload(
    list: &HostList,
    resource: This,
    texture: &ID3D12Resource,
    staging: &Staging,
    source: &ID3D12Resource,
    base: u64,
) {
    use std::sync::atomic::Ordering::Relaxed;
    const COPY_DEST: u32 = 0x400;
    // A texture cut from a block may lie where another did: what the card
    // holds of that one is done with before the first upload into this one.
    if crate::pool::on() && !staging.uploaded.load(Relaxed) && crate::pool::pooled(texture) {
        let aliasing = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                    pResourceBefore: std::mem::ManuallyDrop::new(None),
                    // SAFETY: a borrowed pointer the barrier never releases.
                    pResourceAfter: unsafe { std::mem::transmute_copy(texture) },
                }),
            },
        };
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[aliasing]) };
    }
    crate::list::ensure_state(list, resource, COPY_DEST);
    // `source` lives until the list is reset: the deferred refresh holds
    // the staging buffer, the list its scratch.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (source.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        base + footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + base + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: staging.first - staging.level0 + i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(source) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: base + footprint.Offset,
                    ..*footprint
                },
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

/// Whether a CPU-written plane is copied by each list that samples it
/// (`VICTORY_MOVIE_PLANES=1`; off by default). After codec_d's smooth
/// movie-plane uploads.
fn movie_planes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_MOVIE_PLANES").is_ok_and(|v| v.trim() == "1"))
}

/// [`upload_from`] of a [`Staging::movie_plane`]: from the list's scratch,
/// filled with the title's plane when the list is submitted.
fn upload_plane(list: &HostList, resource: This, texture: &ID3D12Resource, staging: &Staging) {
    let (at, len) = staging.memory();
    let pitch = staging.footprints[0].Footprint.RowPitch as usize;
    let bytes = pitch * staging.rows[0] as usize;
    // Placed footprints start on 512 bytes, scratch on 256.
    let Some((buffer, gpu, data)) = list.scratch_held(bytes as u64 + 256) else {
        once!("no scratch for a movie plane");
        return;
    };
    // SAFETY: a live buffer.
    let start = gpu - unsafe { buffer.GetGPUVirtualAddress() };
    let base = start.next_multiple_of(512);
    let dst = data as usize + (base - start) as usize;
    let layout = &staging.layout;
    let mip = &layout.mips[0];
    if mip.offset == 0 && mip.pitch as usize * layout.desc.element_bytes as usize == pitch && bytes <= len {
        list.defer_fill(dst as *mut u8, at as *const u8, bytes);
    } else {
        let layout = layout.clone();
        list.defer_host_read(at, len, move || {
            // SAFETY: the list holds its scratch until it is reset.
            let out = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, bytes) };
            with_title(at as *const u8, len, |read| {
                if let Err(error) = layout.detile(read, 0, 0, out, pitch, bytes) {
                    out.fill(0);
                    once!(?error, "movie plane not detiled");
                }
            });
        });
    }
    record_upload(list, resource, texture, staging, &buffer, base);
    staging.uploaded.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether title textures that lists sample are uploaded on each list's
/// prelude at submission, when the check that refreshes the staging buffer
/// has run, if it changed or was never uploaded (`VICTORY_PRELUDE_UPLOADS=1`;
/// off: the list records the upload when it samples the texture, if the
/// staging buffer was changed then, as before; bakes and movie planes keep
/// that either way).
///
/// Ported from Belobug/VIctory-Belobug-AMD 5d6bd84 ("Upload title textures
/// in submission prelude to fix frame-order & flickering"), where it is the
/// only way. Recorded when the list samples a texture, an upload is left out
/// if the staging buffer held what the host copy holds at that moment; the
/// check at submission may then find the memory changed and detile it, and
/// only the next list that samples the texture uploads that: a frame late
/// when every list of the frame was recorded before the check (a ring of
/// video buffers then shows an old picture, a texture streamed in shows the
/// last one for a frame). Decided at submission, the copy takes what the
/// check found, in the submission that samples it. A candidate of the
/// flicker at scene loads (analysis 6.12).
pub(crate) fn prelude_uploads() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PRELUDE_UPLOADS").is_some_and(|v| v == "1"))
}

/// Whether the prelude of the list being submitted records the upload into
/// a host texture ([`PreludeUpload::finish_refresh`]): `Some(first upload)`
/// when it was never uploaded or its staging buffer was detiled since the
/// last upload; `None` when the host texture holds what the staging buffer
/// holds. (Ported from Belobug/VIctory-Belobug-AMD 5d6bd84.)
fn claim_upload(uploaded: &std::sync::atomic::AtomicBool, dirty: &std::sync::atomic::AtomicBool) -> Option<bool> {
    use std::sync::atomic::Ordering::AcqRel;
    let was_uploaded = uploaded.swap(true, AcqRel);
    let was_dirty = dirty.swap(false, AcqRel);
    (!was_uploaded || was_dirty).then_some(!was_uploaded)
}

/// A title texture a list samples, its upload decided at the list's
/// submission ([`prelude_uploads`]): its staging buffer, its host copy (both
/// held by the list until it is reset), and its levels on demand. (Ported
/// from Belobug/VIctory-Belobug-AMD 5d6bd84, without its texture store's
/// prefill, which this tree does not have.)
#[derive(Clone)]
pub(crate) struct PreludeUpload {
    staging: Arc<Staging>,
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    on_demand: Option<crate::resource::OnDemand>,
}

impl PreludeUpload {
    /// The staging buffer, for the submission's check (`Step::Texture`).
    pub(crate) fn staging(&self) -> Arc<Staging> {
        self.staging.clone()
    }

    /// On the submitting thread, once the check that refreshes the staging
    /// buffer has run for this submission: `Some(first upload)` if the
    /// list's prelude records the copy into the host texture, with what the
    /// list's own upload noted when it was recorded (a host copy grown, the
    /// levels on demand given memory, what the host copy holds).
    pub(crate) fn finish_refresh(&self) -> Option<bool> {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(first_upload) = claim_upload(&self.staging.uploaded, &self.staging.dirty) else {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexSkipped, 1);
            return None;
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexUploads, 1);
        // A host copy that took the place of one with fewer levels: the slots
        // that show that one are made again once this upload has run.
        if self.staging.first == self.staging.level0 && self.copy.grown.swap(false, Relaxed) {
            crate::descriptor::host::levels_changed_after(self.staging.memory as u64);
        }
        // Levels on demand (`crate::tiles`): the held levels the staging
        // buffer holds as the title's are given the card's memory before the
        // prelude copies into them; else what the host copy now holds is
        // noted.
        match &self.on_demand {
            Some(on_demand) => {
                let waiting = on_demand.uploaded(self.staging.detiled.load(Relaxed), self.staging.memory as u64);
                self.staging.waiting.store(if waiting { crate::host::moment().max(1) } else { 0 }, Relaxed);
                // A texture picked by index has no staging buffer after this
                // upload: its slots are made due again by the texture pass.
                if waiting {
                    crate::descriptor::host::levels_waiting(self.staging.memory as u64);
                }
            }
            None => self.staging.note_held(),
        }
        Some(first_upload)
    }

    /// Records the copy of the staging buffer into the host texture on
    /// `list` (the prelude), the host texture in the copy destination state.
    pub(crate) fn record(&self, list: &ID3D12GraphicsCommandList) {
        let texture = &self.copy.texture;
        // The staging buffer lives as long as `self.staging`, which the
        // list's steps hold until it is reset.
        for (i, footprint) in self.staging.footprints.iter().enumerate() {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: self.staging.first - self.staging.level0 + i as u32,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&self.staging.buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: self.staging.base + footprint.Offset,
                        ..*footprint
                    },
                },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
}

/// Levels on demand: the levels a list claimed for its zeros
/// (`crate::tiles::Tiled::claim`), held by the list's submission-time
/// step. A list that is reset without having been submitted gives the
/// claim up, or the next list that writes those levels would send no zeros
/// and what its write does not cover would be what the tiles' last owner
/// left.
struct Claimed {
    on_demand: crate::resource::OnDemand,
    fresh: u32,
    ran: std::sync::atomic::AtomicBool,
}

impl Drop for Claimed {
    fn drop(&mut self) {
        if self.fresh != 0
            && !*self.ran.get_mut()
            && let Some(tiled) = self.on_demand.tiled()
        {
            tiled.unclaim(self.fresh);
        }
    }
}

/// Records copies of zeros into every subresource of `texture`, which is in
/// the copy destination state.
pub(crate) fn zeros(list: &HostList, host: &crate::host::Host, texture: &ID3D12Resource) {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { u32::from(d.DepthOrArraySize) };
    let count = u32::from(d.MipLevels) * layers;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut bytes = 0u64;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut bytes),
        );
    }
    if bytes == 0 || bytes == u64::MAX {
        return;
    }
    let most = footprints
        .iter()
        .zip(&rows)
        .map(|(f, &rows)| u64::from(f.Footprint.RowPitch) * u64::from(rows) * u64::from(f.Footprint.Depth))
        .max()
        .unwrap_or(0);
    let buffer = match shared_zeros(host, most) {
        Some(buffer) => {
            // Each subresource reads the same zeros from the start.
            for footprint in &mut footprints {
                footprint.Offset = 0;
            }
            buffer
        }
        None => {
            let Some((buffer, data)) = list.upload(host, bytes) else {
                return;
            };
            // SAFETY: a mapped buffer of `bytes` bytes, just made.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            buffer
        }
    };
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above; the list holds an upload buffer until it is
            // reset, and the shared zeros are never let go.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

/// A buffer of at least `size` bytes of zeros in card memory, which every
/// [`zeros`] copy reads and nothing writes; `None` past 256 MB or with
/// `VICTORY_SHARED_ZEROS=0`. Before it each texture's zeros were an upload
/// buffer of the whole texture (a level, for [`card_writes`]), made, waited
/// resident and cleared while the list was recorded, and sent over the bus
/// again. The buffer grows by powers of two; one outgrown stays, as a list
/// recorded with it may not have run.
fn shared_zeros(host: &crate::host::Host, size: u64) -> Option<ID3D12Resource> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_SHARED_ZEROS").is_none_or(|v| v != "0")) || size > 256 << 20 {
        return None;
    }
    static ZEROS: Mutex<(Option<(ID3D12Resource, u64)>, Vec<ID3D12Resource>)> = Mutex::new((None, Vec::new()));
    let mut zeros = ZEROS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((buffer, len)) = &zeros.0
        && *len >= size
    {
        return Some(buffer.clone());
    }
    let len = size.next_power_of_two().max(4 << 20);
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; memory made without CREATE_NOT_ZEROED holds zeros.
    let made = unsafe {
        host.device
            .CreateCommittedResource(
                &heap,
                crate::host::not_resident(),
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
            .and_then(|()| crate::host::resident_soon(&host.device, buffer.as_ref().ok_or_else(windows::core::Error::empty)?, false))
    };
    let buffer = match made {
        Ok(()) => buffer?,
        Err(e) => {
            tracing::warn!(error = %e, len, "no buffer of zeros");
            return None;
        }
    };
    crate::host::name(&buffer, || format!("zeros {len:#x}"));
    if let Some((outgrown, _)) = zeros.0.take() {
        zeros.1.push(outgrown);
    }
    zeros.0 = Some((buffer.clone(), len));
    Some(buffer)
}

/// Levels on demand (`crate::tiles`): the card itself is about to write
/// title texture `resource` in `list` (a copy or a resolve into level
/// `level`, or into every level). If its host copy `texture` was made
/// without memory, the levels written get theirs when the list is
/// submitted, ahead of it on the queue; the ones that have none yet are
/// first written with zeros, here in the same list, so that what the
/// card's write does not cover is not what the tiles' last owner left.
/// To be called before the list is brought to the state of that write.
/// shortcut: two lists that write the same new level, submitted in the
/// other order than recorded, lose the first one's write under the zeros
/// of the other; so does an upload recorded and not yet submitted.
pub(crate) fn card_writes(list: &HostList, resource: This, texture: &ID3D12Resource, level: Option<u32>) {
    let Some(on_demand) = crate::resource::host_tiled(resource) else {
        return;
    };
    let Some(tiled) = on_demand.tiled() else {
        return;
    };
    let levels = match level {
        Some(level) => tiled.with_level(level),
        None => tiled.every_level(),
    };
    if levels == 0 {
        return;
    }
    let fresh = tiled.claim(levels);
    // Of them, the ones whose zeros could not be recorded.
    let mut unzeroed = if host().is_none() { fresh } else { 0 };
    if fresh != 0
        && let Some(host) = host()
    {
        const COPY_DEST: u32 = 0x400;
        crate::list::ensure_state(list, resource, COPY_DEST);
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        for level in (0..u32::from(d.MipLevels).min(32)).filter(|&level| fresh & (1u32 << level) != 0) {
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut bytes = 0u64;
            // SAFETY: outputs for one subresource.
            unsafe {
                host.device
                    .GetCopyableFootprints(&d, level, 1, 0, Some(&mut footprint), None, None, Some(&mut bytes));
            }
            let made = shared_zeros(host, bytes).or_else(|| {
                let (buffer, data) = list.upload(host, bytes)?;
                // SAFETY: a mapped buffer of `bytes` bytes, just made.
                unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
                Some(buffer)
            });
            let Some(buffer) = made else {
                // (With the levels that share its tiles, if it is one of
                // the small ones.)
                unzeroed |= tiled.with_level(level);
                continue;
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above; as in [`zeros`].
                pResource: unsafe { std::mem::transmute_copy(&buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
    // A new level without its zeros gets no tiles from this list: the
    // card's write into it is lost (the level reads zeros, as before the
    // write), which is less wrong than a part write over what the tiles'
    // last owner left. The next list that writes it claims it again.
    unzeroed &= fresh;
    if unzeroed != 0 {
        tiled.unclaim(unzeroed);
    }
    let levels = levels & !unzeroed;
    if levels == 0 {
        return;
    }
    let memory = crate::resource::texture_memory(resource).map_or(0, |(memory, _)| memory as u64);
    let step = Claimed {
        on_demand,
        fresh: fresh & !unzeroed,
        ran: std::sync::atomic::AtomicBool::new(false),
    };
    list.defer_host(move || {
        step.ran.store(true, std::sync::atomic::Ordering::Relaxed);
        step.on_demand.card_wrote(levels, memory);
    });
}

impl Staging {
    /// Levels on demand: what says, beside the mapping, which levels of
    /// the texture are the title's now. First the lowest level a live view
    /// of the texture shows (0 without a view that says): the title's own
    /// word, a texture it holds from level 3 on is viewed from level 3.
    /// Then where the next texture placed in this one's span starts, in
    /// bytes from this one's start (`u64::MAX`: none does): for a texture
    /// viewed from level 0 although a neighbour sits where its large
    /// levels would be.
    fn judgement(&self) -> [u64; 2] {
        let start = self.memory as u64;
        let first = crate::descriptor::host::first_viewed(start).map_or(0, u64::from);
        let limit = crate::descriptor::host::next_start(start, start.saturating_add(self.layout.size))
            .map_or(u64::MAX, |at| at - start);
        [first, limit]
    }

    /// What [`Staging::judgement`] said at the last detiling.
    fn judged(&self) -> [u64; 2] {
        use std::sync::atomic::Ordering::Relaxed;
        [self.judged[0].load(Relaxed), self.judged[1].load(Relaxed)]
    }

    /// Levels on demand: the levels that are the title's now, a bit a
    /// level. A level is held if no view starts above it, its bytes end
    /// before the next texture starts, and all of them are mapped. The
    /// smallest levels share the texture's first block and are held
    /// together. (The console layout: that block first, then the other
    /// levels from the smallest up, level 0 last:
    /// `victory_formats::texture::Layout`.) A texture this cannot tell
    /// the levels of holds them all.
    fn held_levels(&self, [first, limit]: [u64; 2]) -> u32 {
        let mips = &self.layout.mips;
        if self.layout.slices != 1 || mips.is_empty() || mips.len() > 32 {
            return u32::MAX;
        }
        let tail = (self.layout.first_tail_mip as usize).min(mips.len());
        let there = |from: u64, to: u64| {
            to > from && to <= limit && committed(self.memory.wrapping_add(from as usize), (to - from) as usize)
        };
        let mut held = 0u32;
        for (level, m) in mips.iter().enumerate().take(tail) {
            let end = if level == 0 { self.layout.slice_size } else { mips[level - 1].offset };
            if level as u64 >= first && there(m.offset, end) {
                held |= 1 << level;
            }
        }
        if tail < mips.len() && there(0, self.layout.alignment.min(self.layout.slice_size)) {
            for level in tail..mips.len() {
                if level as u64 >= first {
                    held |= 1 << level;
                }
            }
        }
        held
    }

    /// Whether a list that samples the texture at `frame` records the copy
    /// into its host texture even if the check found nothing new: the
    /// check may hash it again ([`hot_level`] 2).
    fn hot_at(&self, frame: u64) -> bool {
        hot_level() >= 2 && frame < self.hot.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The hash found the memory changed; `first`: it was hashed for the
    /// first time, which is no change for the hot textures; `blind`: the
    /// guard saw no write, so the change came another way and the texture
    /// is hashed at every check for a while.
    fn found_changed(&self, last: &mut Last, first: bool, blind: bool) {
        if first || hot_level() == 0 {
            return;
        }
        let frame = crate::frame::current();
        last.changed(frame);
        if blind {
            last.hot_until = last.hot_until.max(frame + HOT_FRAMES);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexHot, 1);
        }
        self.hot.store(last.copy_until(hot_copy()), std::sync::atomic::Ordering::Relaxed);
    }

    /// The hash found the memory as it was.
    fn found_unchanged(&self, last: &mut Last) {
        if hot_level() == 0 {
            return;
        }
        last.unchanged(crate::frame::current(), hot_level() >= 2);
        self.hot.store(last.copy_until(hot_copy()), std::sync::atomic::Ordering::Relaxed);
    }

    /// Forgets what the staging buffer was last filled from: the next
    /// refresh detiles anew, whether the memory changed or not.
    fn forget(&self) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Last::default();
    }

    /// Whether it is one linear R8 plane, as the title's video decoder
    /// writes on the CPU each frame.
    fn movie_plane(&self) -> bool {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8_UNORM;
        let desc = &self.layout.desc;
        !self.on_demand
            && desc.dimension == victory_formats::texture::Dimension::Tex2d
            && desc.swizzle == victory_formats::texture::Swizzle::LINEAR
            && desc.depth == 1
            && desc.mips == 1
            && desc.samples == 1
            && self.footprints.len() == 1
            && self.footprints[0].Offset == 0
            && self.footprints[0].Footprint.Format == DXGI_FORMAT_R8_UNORM
    }

    /// The title memory the texture is read from: address and bytes.
    pub(crate) fn memory(&self) -> (usize, usize) {
        (self.memory as usize, self.read)
    }

    /// The bytes an upload copies from the staging buffer.
    #[cfg(feature = "diagnostics")]
    fn copied_bytes(&self) -> u64 {
        let parts = self.footprints.iter().zip(&self.rows);
        parts.map(|(f, &rows)| u64::from(f.Footprint.RowPitch) * u64::from(rows) * u64::from(f.Footprint.Depth)).sum()
    }

    /// The title's level and the array slice of the staging buffer's part
    /// `i`.
    fn part(&self, i: usize) -> (u32, u32) {
        let mips = self.mips - self.first;
        (self.first + i as u32 % mips, i as u32 / mips)
    }

    /// The host copy holds what the staging buffer was filled from
    /// ([`unchanged`]).
    fn note_held(&self) {
        let hash = self.last.lock().unwrap_or_else(|e| e.into_inner()).hash;
        *self.holds.lock().unwrap_or_else(|e| e.into_inner()) = hash.map(|hash| (self.first, hash));
    }

    /// `VICTORY_TEXTURE_REFRESH_EVERY`: a texture is read once every N frames
    /// at most, round-robin by its own address, instead of every frame (0, 1 or
    /// unset: every frame, as before). A title whose world is a frame a second
    /// spends most of that frame reading title memory for textures; spreading
    /// the reads costs at most N frames of staleness per texture and divides
    /// the cost. A texture skipped keeps its `exact` mark for its next turn.
    /// Only in builds with diagnostics: it trades the picture for frame time,
    /// which no build that ships should do.
    fn due(&self) -> bool {
        #[cfg(feature = "diagnostics")]
        {
            static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
            let every = *EVERY.get_or_init(|| {
                std::env::var("VICTORY_TEXTURE_REFRESH_EVERY")
                    .ok()
                    .and_then(|v| v.trim().parse::<u64>().ok())
                    .filter(|&n| n > 0)
                    .unwrap_or(1)
            });
            if every > 1 {
                return (crate::frame::current() + ((self.memory as usize >> 16) as u64)).is_multiple_of(every);
            }
        }
        true
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        if !self.due() {
            return;
        }
        let started = std::time::Instant::now();
        let _texture = crate::timeline::span("a texture's refresh");
        #[cfg(feature = "diagnostics")]
        let (forced, fills) = (self.forced.swap(0, Relaxed), self.fills.load(Relaxed));
        let read = self.refresh_now();
        // Of the copies recorded because the texture was hot, one takes what
        // this refresh detiled, if it detiled; the others copy what the host
        // texture holds already.
        #[cfg(feature = "diagnostics")]
        if forced > 0 {
            let wasted = u64::from(forced - u32::from(self.fills.load(Relaxed) != fills));
            crate::copystat::add(crate::copystat::C::TexForcedSame, wasted);
            crate::copystat::add(crate::copystat::C::TexForcedSameBytes, wasted * self.copied_bytes());
        }
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let whole = std::time::Instant::now();
        let read = self.refresh_timed();
        REFRESH_WAITS[3].fetch_add(whole.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        read
    }

    fn refresh_timed(&self) -> usize {
        let size = self.read;
        let start = self.memory as usize;
        let mut last = waited(0, || self.last.lock().unwrap_or_else(|e| e.into_inner()));
        let last = &mut *last;
        let exact = self.exact.swap(false, std::sync::atomic::Ordering::Relaxed);
        let alone = self.once.load(std::sync::atomic::Ordering::Relaxed);
        if alone && !exact && last.hash.is_some() {
            return 0;
        }
        if !alone && let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, false)) {
            if !exact && last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            let since = std::mem::replace(&mut last.asked, now);
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            let read = with_read(size, |read| {
                if since != 0
                    && !self.on_demand
                    && crate::toggled!("VICTORY_TEXTURE_ROWS")
                    && let Some(bytes) = self.detile_written(read, &written, since)
                {
                    return bytes;
                }
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                self.detile(read);
                size
            });
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, read as u64);
            return read;
        }
        // View memory has no write-watch, and a texture there was hashed
        // whole at every check: 229 MB and 51 ms a frame in a gameplay view
        // where two textures change (run 58). It is guarded now, as buffers
        // in view memory are: a write of the title's to it faults once and
        // is noted, a mapping is announced by the runtime. While neither
        // happened the texture is as it was. What the guard cannot see (a
        // write to the same pages through another view) is still found by
        // the hash, which is taken on turns only ([`turn_every`]), by the
        // texture's address, so the textures take turns.
        // `VICTORY_TEXTURE_GUARD=0` hashes at every check, as before.
        static GUARD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // The guard saw no write: a change the hash finds came another way.
        let mut blind = false;
        // The guard saw a write since the last check.
        let mut seen = false;
        // Why an unwritten texture is hashed: 0 written or unguarded, 1 its
        // turn, 2 due, 3 in its first frames only.
        let mut why = 0;
        if !alone
            && *GUARD.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_GUARD").is_none_or(|v| v != "0"))
            && let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, true))
        {
            let unwritten = last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked);
            last.asked = now;
            let frame = crate::frame::current();
            let every = if last.hot_until == 0 { turn_every() } else { 8 };
            let turn = frame.wrapping_add(turn_phase(start)).is_multiple_of(every);
            // A texture starts hot: one written that way from its first
            // frames on changes before the hash has ever caught it.
            if last.started == 0 {
                last.started = frame + HOT_FRAMES;
            }
            let first = frame < last.started && !last.hash_due(frame);
            let mut due = hot_level() >= 1 && (frame < last.started || last.hash_due(frame));
            let passed = due && first && last.turned == frame && first_once();
            if passed {
                due = false;
            }
            if !exact && unwritten && !due && (!turn || last.turned == frame) {
                #[cfg(feature = "diagnostics")]
                if passed {
                    crate::copystat::add(crate::copystat::C::TexFirstSkipped, size as u64);
                } else if turn {
                    crate::copystat::add(crate::copystat::C::TexTurnSkipped, size as u64);
                }
                return 0;
            }
            if unwritten {
                why = match (due, first) {
                    (false, _) => 1,
                    (true, false) => 2,
                    (true, true) => 3,
                };
                if !due || first {
                    last.turned = frame;
                }
            }
            blind = unwritten;
            seen = !unwritten;
        }
        // From here on the texture's memory may be read in place. Held until
        // that read is done: the runtime changes what title memory shows
        // only with this taken alone, so what `committed` finds mapped
        // stays mapped (see `crate::watch::title_memory_reading`; without
        // it a page went between the asking and the read in runs 71 and
        // 74).
        // `VICTORY_REFRESH_LOCKLESS=1` (off by default): read without the
        // lock first, kept only if no change of mapping began or ended
        // meanwhile.
        static LOCKLESS: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var_os("VICTORY_REFRESH_LOCKLESS").is_some_and(|v| v == "1"));
        if *LOCKLESS {
            let epoch = crate::watch::changes();
            if epoch.is_multiple_of(2)
                && with_read(size, |bytes| {
                    if !crate::host::read_into(self.memory, bytes) || epoch != crate::watch::changes() {
                        return false;
                    }
                    let hash = digest(bytes);
                    if last.hash == Some(hash) {
                        count_hashed(size, why);
                        self.found_unchanged(last);
                        return true;
                    }
                    // Streamed mip selection still needs the mapping guard.
                    if self.on_demand {
                        return false;
                    }
                    count_hashed(size, why);
                    self.found_changed(last, last.hash.is_none(), blind);
                    last.hash = Some(hash);
                    self.detile(bytes);
                    true
                })
            {
                return size;
            }
        }
        let _mapped = if no_wait() && !exact {
            // Not waited for: while the runtime remaps title memory the
            // texture is read through the kernel instead, whole or not at
            // all (`no_wait`).
            let Some(mapped) = crate::watch::title_memory_reading_now() else {
                return self.refresh_unlocked(last, size);
            };
            mapped
        } else if exact && exact_unlocked() {
            match crate::watch::title_memory_reading_now() {
                Some(mapped) => mapped,
                None if self.refresh_exact_unlocked(last, size, blind) => return size,
                None => waited(2, crate::watch::title_memory_reading),
            }
        } else {
            waited(2, crate::watch::title_memory_reading)
        };
        if !committed(self.memory, size) {
            // Partly mapped: a streamed texture with only some mips' pages
            // there, the usual state while it streams in. Read through the
            // copy, which stands unmapped pages (zeros), and upload what
            // is there; nothing at all mapped changes nothing.
            // shortcut: read again at most every 30 frames while the guards
            // see no write ([`partial_due`]): a change they miss shows up
            // to 30 frames late.
            let frame = crate::frame::current();
            if !exact && !partial_due(last.partial, frame, seen) {
                if seen {
                    // Read in the next frame: the write is not forgotten.
                    last.asked = 0;
                }
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexPartialSkipped, size as u64);
                return 0;
            }
            #[cfg(feature = "diagnostics")]
            if !partial_due(last.partial, frame, false) {
                crate::copystat::add(crate::copystat::C::TexPartialWritten, size as u64);
            }
            last.partial = Some(frame);
            count_hashed(size, why);
            let changed = with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                let hash = digest(read);
                if last.hash == Some(hash) {
                    return false;
                }
                last.hash = Some(hash);
                if read.iter().all(|&b| b == 0) {
                    return false;
                }
                self.detile(read);
                true
            });
            #[cfg(feature = "diagnostics")]
            if changed {
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
            }
            if changed {
                self.found_changed(last, false, blind);
            }
            return size;
        }
        if through_copy() {
            return with_title(self.memory, size, |tiled| {
                let hash = digest(tiled);
                count_hashed(size, why);
                if last.hash == Some(hash) {
                    self.found_unchanged(last);
                    return size;
                }
                self.found_changed(last, last.hash.is_none(), blind);
                last.hash = Some(hash);
                self.detile(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
                size
            });
        }
        // SAFETY: committed memory of the texture's layout size, and it
        // stays committed while `_mapped` is held.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = digest(tiled);
        count_hashed(size, why);
        if last.hash == Some(hash) {
            self.found_unchanged(last);
            return size;
        }
        self.found_changed(last, last.hash.is_none(), blind);
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// A refresh while the runtime is remapping title memory ([`no_wait`]):
    /// the texture is read through the kernel, which fails if a page of it
    /// is not there at this moment. Then a staging buffer that was filled
    /// before keeps what it holds and the texture is looked at again at its
    /// next refresh (what was learnt about writes to it is forgotten, so
    /// that it is not passed over as unwritten); one never filled takes the
    /// pages that are there ([`Staging::fill`]). Returns the bytes read.
    fn refresh_unlocked(&self, last: &mut Last, size: usize) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let read = with_read(size, |read| {
            if !crate::host::read_into(self.memory, read) {
                return false;
            }
            let hash = digest(read);
            if last.hash != Some(hash) {
                last.hash = Some(hash);
                self.detile(read);
            }
            true
        });
        if read {
            UNLOCKED[0].fetch_add(1, Relaxed);
            size
        } else if last.hash.is_none() && fill_unlocked() {
            self.fill(last, size)
        } else {
            UNLOCKED[1].fetch_add(1, Relaxed);
            last.asked = 0;
            0
        }
    }

    /// An exact refresh (for a bake) that finds the runtime remapping
    /// title memory ([`exact_unlocked`]): the texture is read whole through
    /// the kernel, which needs no lock and fails if a page of it is not
    /// there; read, it is what title memory holds now, as a read under the
    /// lock would be. Whether it was read (and detiled if it changed);
    /// if a page was missing the refresh waits for the lock as before.
    fn refresh_exact_unlocked(&self, last: &mut Last, size: usize, blind: bool) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let read = with_read(size, |read| {
            if !crate::host::read_into(self.memory, read) {
                return false;
            }
            let hash = digest(read);
            if last.hash == Some(hash) {
                self.found_unchanged(last);
            } else {
                self.found_changed(last, last.hash.is_none(), blind);
                last.hash = Some(hash);
                self.detile(read);
            }
            true
        });
        EXACT_UNLOCKED[usize::from(!read)].fetch_add(1, Relaxed);
        read
    }

    /// The first filling of the staging buffer while the runtime is
    /// remapping title memory, of a texture not all of which can be read:
    /// with the pages that are there, zeros for the others, as a partly
    /// mapped texture is filled under the lock. Pages that are away only
    /// for the moment of a change (the runtime unmaps the rest of a view to
    /// change one page of it) are read when that change is over. If less
    /// was read than was mapped when the upload was recorded, or is mapped
    /// now, the texture's slots are looked at again.
    fn fill(&self, last: &mut Last, size: usize) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let start = self.memory as u64;
        let mut there = 0;
        let got = with_read(size, |read| {
            let mut got = 0;
            for _ in 0..4 {
                got = crate::host::read_there(self.memory, read) as u64;
                there = mapped_bytes(self.memory, size);
                if got >= there.max(self.expected.load(Relaxed)) {
                    break;
                }
                crate::watch::change_over();
            }
            if got != 0 {
                last.hash = Some(digest(read));
                self.detile(read);
            }
            got
        });
        last.partial = Some(crate::frame::current());
        UNLOCKED[2].fetch_add(1, Relaxed);
        if got < there.max(self.expected.load(Relaxed)) {
            UNLOCKED[3].fetch_add(1, Relaxed);
            last.asked = 0;
            crate::descriptor::host::read_short(start);
        } else {
            crate::descriptor::host::read_all(start);
        }
        size
    }

    /// Of each part, reads into `read` and detiles only the rows whose tiled
    /// bytes the pages in `written` written after `since` touch (one page
    /// written into mip 0 of a 4K texture re-read and detiled all 16 MiB of
    /// it). Returns the bytes read, or `None` if the layout cannot say which
    /// rows or a part was not detiled (then it is detiled whole).
    /// `VICTORY_TEXTURE_ROWS=0`: every written texture is detiled whole, as
    /// before.
    fn detile_written(&self, read: &mut [u8], written: &[(usize, u64)], since: u64) -> Option<usize> {
        let (start, size) = (self.memory as usize, self.read);
        let page = crate::watch::PAGE;
        let mut changed: Vec<(usize, usize)> = Vec::new();
        for &(p, _) in written.iter().filter(|&&(_, w)| w > since) {
            let (a, b) = ((p * page).max(start) - start, ((p + 1) * page).min(start + size) - start);
            match changed.last_mut() {
                Some(last) if last.1 >= a => last.1 = last.1.max(b),
                _ if a < b => changed.push((a, b)),
                _ => {}
            }
        }
        let bands = (0..self.footprints.len())
            .map(|i| {
                let (mip, slice) = self.part(i);
                self.layout.touched_rows(mip, slice, &changed)
            })
            .collect::<Option<Vec<_>>>()?;
        let mut bytes = 0;
        let mut done: Vec<(usize, usize)> = Vec::new();
        for (rows, tiled) in &bands {
            let (a, b) = (tiled.start, tiled.end.min(size));
            if !rows.is_empty() && a < b && !done.contains(&(a, b)) {
                // SAFETY: inside the read buffer of the texture's size.
                crate::host::copy_mapped(unsafe { read.as_mut_ptr().add(a) }, self.memory.wrapping_add(a), b - a, false);
                (bytes, _) = (bytes + b - a, done.push((a, b)));
            }
        }
        if bands.iter().all(|(rows, _)| rows.is_empty()) {
            return Some(bytes);
        }
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        self.fills.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        let touched = bands.iter().map(|(rows, _)| rows.clone()).collect::<Vec<_>>();
        for (i, (band, _)) in bands.into_iter().enumerate() {
            if band.is_empty() {
                continue;
            }
            let (mip, slice) = self.part(i);
            let (footprint, rows) = (&self.footprints[i], self.rows[i]);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = (pitch * rows as usize * f.Depth as usize)
                .min(self.bytes.saturating_sub(footprint.Offset) as usize);
            // SAFETY: the part's piece of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) = self.layout.detile_rows(read, mip, slice, linear, pitch, pitch * rows as usize, band) {
                tracing::warn!(?e, mip, slice, "texture rows not detiled");
                return None;
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexRowsDetiled, bytes as u64);
        #[cfg(feature = "diagnostics")]
        self.check_rows(&touched);
        Some(bytes)
    }

    /// With `VICTORY_TEXTURE_ROWS_CHECK=1`: after a detile of written rows,
    /// detiles the whole texture again into a scratch copy and logs each
    /// part whose staging rows differ, inside the rows just detiled (the
    /// title wrote between the two reads, or the rows were detiled wrong)
    /// or outside them (a write the watch did not see). The first 16 such
    /// textures are saved to the capture folder as `rows-<memory>.tiled`
    /// and `rows-<memory>-<part>.staging` / `.whole`.
    #[cfg(feature = "diagnostics")]
    fn check_rows(&self, touched: &[std::ops::Range<u32>]) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static SAVED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if !*ON.get_or_init(|| crate::switch::on("VICTORY_TEXTURE_ROWS_CHECK")) {
            return;
        }
        let mut tiled = vec![0; self.read];
        crate::host::copy_mapped(tiled.as_mut_ptr(), self.memory, self.read, false);
        let mut differed = false;
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = self.part(i);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = (pitch * rows as usize * f.Depth as usize)
                .min(self.bytes.saturating_sub(footprint.Offset) as usize);
            // SAFETY: the part's piece of the mapped staging buffer.
            let staged = unsafe { std::slice::from_raw_parts(self.data.add(footprint.Offset as usize), len) };
            let mut whole = staged.to_vec();
            if self.layout.detile(&tiled, mip, slice, &mut whole, pitch, pitch * rows as usize).is_err() {
                continue;
            }
            let Some(first) = staged.iter().zip(&whole).position(|(a, b)| a != b) else {
                continue;
            };
            let bytes = staged.iter().zip(&whole).filter(|(a, b)| a != b).count();
            let row = first % (pitch * rows as usize).max(1) / pitch.max(1);
            tracing::warn!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                mip,
                slice,
                row,
                detiled = ?touched[i],
                bytes,
                "texture rows check: staging differs from a whole detile"
            );
            if SAVED.load(std::sync::atomic::Ordering::Relaxed) < 16
                && let Some(dir) = crate::capture::folder()
            {
                let name = format!("rows-{:x}-{i}", self.memory as usize);
                let _ = std::fs::write(dir.join(format!("{name}.staging")), staged);
                let _ = std::fs::write(dir.join(format!("{name}.whole")), &whole);
                differed = true;
            }
        }
        if differed
            && SAVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16
            && let Some(dir) = crate::capture::folder()
        {
            let _ = std::fs::write(dir.join(format!("rows-{:x}.tiled", self.memory as usize)), &tiled);
        }
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        self.fills.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Levels on demand: which levels are the title's is judged here,
        // once, for what the staging buffer holds from now on. The step
        // that gives levels their memory goes by this answer.
        let held = if self.on_demand {
            use std::sync::atomic::Ordering::Relaxed;
            let judged = self.judgement();
            self.judged[0].store(judged[0], Relaxed);
            self.judged[1].store(judged[1], Relaxed);
            let held = self.held_levels(judged);
            self.detiled.store(held, Relaxed);
            held
        } else {
            u32::MAX
        };
        #[cfg(feature = "diagnostics")]
        audit::read(self.memory as usize, tiled);
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = self.part(i);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            // The last row of the last subresource is not padded to its pitch
            // (VIctory 4fc9d70): to the end of the buffer at most, where a
            // block of the pool goes on with another texture's.
            let len = (pitch * rows as usize * f.Depth as usize)
                .min(self.bytes.saturating_sub(footprint.Offset) as usize);
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if mip < 32 && held & (1 << mip) == 0 {
                // Not the title's: zeros, not what lies in that memory.
                linear.fill(0);
                continue;
            }
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Lets go of a title texture's staging buffer (lists that recorded an
/// upload from it hold it until they have run); the next upload makes one
/// anew.
pub(crate) fn unstage(resource: This) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&(resource as usize));
    }
}

/// Where a staged title texture's memory is: address and bytes.
pub(crate) fn staged_memory(resource: This) -> Option<(u64, u64)> {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let (at, len) = map.as_ref()?.get(&(resource as usize))?.memory();
    Some((at as u64, len as u64))
}

/// A fingerprint of the title memory `[at, at + len)`, never 0: of all of
/// it up to 48 KiB, else of 16 KiB at its start, its middle and its end
/// (a texture streams in whole mips; the small ones lie first). Read
/// through the kernel, so pages not there count as zeros.
pub(crate) fn fingerprint(at: u64, len: u64) -> u64 {
    const WINDOW: usize = 16 << 10;
    thread_local! {
        static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let len = len as usize;
    let mut print = 0u128;
    READ.with(|read| {
        let mut read = read.borrow_mut();
        read.resize(3 * WINDOW, 0);
        let mut take = |from: usize, bytes: usize| {
            read[..bytes].fill(0);
            crate::host::copy_mapped(read.as_mut_ptr(), (at as usize + from) as *const u8, bytes, false);
            print = print.rotate_left(37) ^ digest(&read[..bytes]).0;
        };
        if len <= 3 * WINDOW {
            take(0, len);
        } else {
            take(0, WINDOW);
            take((len / 2) & !0xfff, WINDOW);
            take(len - WINDOW, WINDOW);
        }
    });
    let first = (print as u64 ^ (print >> 64) as u64).to_le_bytes();
    u64::from_le_bytes(first) | 1
}

/// Whether a title texture has a staging buffer: a list recorded its
/// upload, and the lists that sample it check it at submission.
/// For the frame listing: whether a list has uploaded `resource`'s staging
/// buffer, and whether it was detiled since. None: it has none.
#[cfg(feature = "diagnostics")]
pub(crate) fn staged_state(resource: This) -> Option<(bool, bool)> {
    use std::sync::atomic::Ordering::Relaxed;
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let staging = map.as_ref()?.get(&(resource as usize))?;
    Some((staging.uploaded.load(Relaxed), staging.dirty.load(Relaxed)))
}

/// The first level a title texture's staging buffer holds; `u32::MAX`
/// without one.
fn staged_from(resource: This) -> u32 {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref()
        .and_then(|map| map.get(&(resource as usize)))
        .map_or(u32::MAX, |s| s.first)
}

/// Whether the levels from `first` on of `resource`'s host copy were filled
/// from the bytes title memory holds now: an upload would bring nothing new.
/// (Of 787 repeated uploads of a frame that streams 398 were equal in every
/// byte, and a texture uploaded again costs a staging buffer, a detiling and
/// a copy more than this look.) `VICTORY_TEXTURE_SAME=0`: not asked.
pub(crate) fn unchanged(resource: This, first: u32) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_SAME").is_none_or(|v| v != "0")) {
        return false;
    }
    let Some(copy) = crate::resource::made_copy(resource) else {
        return false;
    };
    let Some((from, held)) = *copy.holds.lock().unwrap_or_else(|e| e.into_inner()) else {
        return false;
    };
    let Some((memory, desc)) = crate::resource::texture_memory(resource) else {
        return false;
    };
    let Some(layout) = crate::resource::texture_layout(&desc) else {
        return false;
    };
    let layout = &layout.planes[0].layout;
    if from > crate::resource::read_from(layout, first).max(copy.base) {
        return false;
    }
    let size = layout.bytes_from(from) as usize;
    with_read(size, |read| {
        if crate::host::read_there(memory, read) == 0 {
            return false;
        }
        digest(read) == held
    })
}

/// Where the levels from `first` on of a title texture are: address and
/// bytes.
pub(crate) fn title_span(resource: This, first: u32) -> Option<(u64, u64)> {
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let layout = crate::resource::texture_layout(&desc)?;
    let layout = &layout.planes[0].layout;
    Some((memory as u64, layout.bytes_from(crate::resource::read_from(layout, first))))
}

pub(crate) fn is_staged(resource: This) -> bool {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref().is_some_and(|map| map.contains_key(&(resource as usize)))
}

/// The bytes of a title texture's memory, and whether all of it is mapped,
/// if the texture has been staged for upload.
/// (Whether it is mapped is asked only if `ask`; else it counts as mapped.)
/// Only of a texture at an address the title mapped
/// ([`crate::resource::at_title_address`]): the memory of any other is all
/// there while it lives. The pass over textures picked by index asked it of
/// every texture object it took, 20,000-57,000 every 5 seconds, most of them
/// read through tables (a question per 256 pages: 18-111 us for 1-16 MB,
/// milliseconds when the pages had left the working set, and each such page
/// read back into it), for 66 found not all there in a whole run
/// (coolfox-20261126-223059); the pass's unattributed time went with their
/// count, 13 us each. Of one at a title address, the runtime's views of
/// physical pages answer where they cover the memory ([`all_there`]: one
/// look, under a microsecond), as they do for placed textures
/// ([`mapped_bytes`]). `VICTORY_STAGED_ASK_ALL=1`: Windows is asked of every
/// texture, as before.
pub(crate) fn staged(resource: This, ask: bool) -> Option<(u64, bool)> {
    static ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let s = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_ref()?.get(&(resource as usize))?.clone()
    };
    let whole = !ask
        || if *ALL.get_or_init(|| std::env::var_os("VICTORY_STAGED_ASK_ALL").is_some_and(|v| v == "1")) {
            committed(s.memory, s.read)
        } else {
            !crate::resource::at_title_address(resource) || all_there(s.memory, s.read)
        };
    Some((s.read as u64, whole))
}

/// Whether all of `[at, at + len)` is mapped: by the runtime's views of
/// physical pages where any lies there (as [`mapped_bytes`] asks), else by
/// asking Windows ([`committed`]).
fn all_there(at: *const u8, len: usize) -> bool {
    match by_views().then(|| crate::inplace::shown_bytes(at as usize, len)).flatten() {
        Some(there) => there >= len as u64,
        None => committed(at, len),
    }
}

/// Whether how much of title memory is mapped is asked of the runtime's
/// views of physical pages first (`VICTORY_MAPPED_BY_VIEWS=0`: of Windows).
fn by_views() -> bool {
    static BY_VIEWS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *BY_VIEWS.get_or_init(|| std::env::var_os("VICTORY_MAPPED_BY_VIEWS").is_none_or(|v| v != "0"))
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    if by_query() {
        return committed_by_query(at, len);
    }
    if len == 0 {
        return true;
    }
    let (there, of) = readable(at as usize, len, true);
    there == of
}

/// Whether the two questions above and below are put to `VirtualQuery`, as
/// they were (`VICTORY_COMMITTED_QUERY=1`). `VirtualQuery` answers with the
/// length of the run of pages in the same state, which it finds by walking
/// the page tables to the run's end: 8.8 ms for one question in the title's
/// 1.1 GB block (round 6). The pass over the heap's views asks three or
/// more of them for every texture slot that is due, 3,000 slots a frame
/// when the card has no room (run 75): that pass then ran 208-264 ms, and
/// while it ran the submitting thread's write-watch askings waited behind
/// the walks (run 76: 8-22 askings a frame of 6-21 ms each, 46-172 ms).
fn by_query() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMMITTED_QUERY").is_some_and(|v| v == "1"))
}

/// Of the kernel's pages that hold `[at, at + len)` (`len > 0`): how many
/// can be read, and how many there are; with `all`, it stops at the first
/// that cannot. One question to the kernel for up to 256 pages at once
/// (`QueryWorkingSetEx`: the state of each page asked about, no walk); a
/// page not in the working set may still be there (never touched, or
/// trimmed), and a one-byte read through the kernel says.
/// A page that is committed and may not be read counts as not there, which
/// is what the callers ask for: they read or write the memory next.
fn readable(at: usize, len: usize, all: bool) -> (usize, usize) {
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const KERNEL_PAGE: usize = 4096;
    let (first, last) = (at / KERNEL_PAGE, at.saturating_add(len - 1) / KERNEL_PAGE);
    let of = last - first + 1;
    let mut there = 0;
    // SAFETY: plain data.
    let mut info: [PSAPI_WORKING_SET_EX_INFORMATION; 256] = unsafe { std::mem::zeroed() };
    let mut page = first;
    while page <= last {
        let n = (last - page + 1).min(info.len());
        for (i, entry) in info[..n].iter_mut().enumerate() {
            entry.VirtualAddress = ((page + i) * KERNEL_PAGE) as *mut _;
        }
        // SAFETY: `n` entries of the size given; our own process.
        let asked = unsafe {
            K32QueryWorkingSetEx(
                GetCurrentProcess(),
                info.as_mut_ptr().cast(),
                (n * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
            )
        } != 0;
        for (i, entry) in info[..n].iter().enumerate() {
            // Bit 0 of the answer: the page is in the working set.
            // SAFETY: plain data the query filled in.
            let valid = asked && unsafe { entry.VirtualAttributes.Flags } & 1 != 0;
            let mut byte = [0u8];
            if valid || crate::host::read_into(((page + i) * KERNEL_PAGE) as *const u8, &mut byte) {
                there += 1;
            } else if all {
                return (there, of);
            }
        }
        page += n;
    }
    (there, of)
}

fn committed_by_query(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// How many bytes of `[at, at + len)` are committed memory: all of a
/// texture that is there, some of one streaming in (its smaller mips lie
/// at the start of its memory, and their pages are mapped first), none of
/// one not there. With neither the first nor the last page committed the
/// answer is none without a look at the pages between (two queries for the
/// slots whose texture is away, which are asked again and again).
pub(crate) fn mapped_bytes(at: *const u8, len: usize) -> u64 {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    if len == 0 {
        return 0;
    }
    let start = at as usize;
    let end = start.saturating_add(len);
    // Streamed textures lie in views of physical pages, of which the runtime
    // knows which are there: one look, where the host was asked for every
    // 64 KiB of the texture (and once more for each that is not there).
    // `VICTORY_MAPPED_BY_VIEWS=0`: the host is asked, as before.
    if by_views()
        && let Some(there) = crate::inplace::shown_bytes(start, len)
    {
        return there;
    }
    if !committed(at, 1) && !committed((end - 1) as *const u8, 1) {
        return 0;
    }
    if !by_query() {
        static BY_PAGE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *BY_PAGE.get_or_init(|| std::env::var_os("VICTORY_MAPPED_BY_PAGE").is_some_and(|v| v == "1")) {
            // By page, without the walk: whole pages, cut to the range.
            let (there, _) = readable(start, len, false);
            return (there as u64 * 4096).min(len as u64);
        }
        // By 64 KiB block, one page asked of each: the title's memory is
        // mapped in blocks of that size (the console's page), and the
        // answer is a measure of how much of a texture has arrived, not a
        // promise that every byte can be read (the upload reads through the
        // kernel). By page, a texture half there cost a failed kernel read
        // for each of its missing pages, every frame it stayed due: 1,663
        // textures "uploaded in part" kept the pass over the heap's views at
        // 90-160 ms a frame (run 81). `VICTORY_MAPPED_BY_PAGE=1`: by page.
        const BLOCK: usize = 64 << 10;
        let mut there = 0u64;
        let mut block = start & !(BLOCK - 1);
        while block < end {
            let (lo, hi) = (block.max(start), block.saturating_add(BLOCK).min(end));
            if readable(lo, 1, true).0 == 1 {
                there += (hi - lo) as u64;
            }
            block = block.saturating_add(BLOCK);
        }
        return there;
    }
    let (mut p, mut there) = (start, 0u64);
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(end);
        if n == 0 || next <= p {
            break;
        }
        if info.State == MEM_COMMIT {
            there += (next - p) as u64;
        }
        p = next;
    }
    there
}

/// (Neptune) The bytes of title memory a staging buffer of `resource` from
/// mip `first` on reads, as the pass counts them; 0 for what has no console
/// layout.
fn staged_bytes(resource: This, first: u32) -> u64 {
    title_span(resource, first).map_or(0, |(_, bytes)| bytes)
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, copy: &crate::resource::HostTexture, first: u32) -> Option<Arc<Staging>> {
    // Asked before the table is locked: the sweep lets go of a staging
    // buffer with its texture's host side locked, in that order.
    let on_demand = crate::resource::host_tiled(resource).is_some();
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize))
        && s.first <= first.max(copy.base)
        && s.serial == copy.serial
    {
        return Some(s.clone());
    }
    // (Neptune) One a prepare thread made and detiled (`prepare`): taken as
    // it is (`VICTORY_TEXTURE_PREPARE_THREADS`; none without), unless the
    // host copy was made anew since.
    let s = match prepare::take_ready(resource, first, on_demand).filter(|s| s.serial == copy.serial) {
        Some(s) => s,
        None => build_staging(resource, copy, first, on_demand)?,
    };
    map.insert(resource as usize, s.clone());
    add_to_address_index(&s);
    Some(s)
}

/// (Neptune) For a prepare thread (`prepare`): the staging state of title
/// texture `resource` from mip `first` on, made as [`staging`] makes it but
/// not put in the table (the pass that records its upload puts it there).
/// `None` if the table has one that serves already, or none can be made.
fn prepared_staging(resource: This, copy: &crate::resource::HostTexture, first: u32) -> Option<Arc<Staging>> {
    let on_demand = crate::resource::host_tiled(resource).is_some();
    let first = if on_demand { 0 } else { first };
    {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        if map
            .as_ref()
            .and_then(|map| map.get(&(resource as usize)))
            .is_some_and(|s| s.first <= first.max(copy.base) && s.serial == copy.serial)
        {
            return None;
        }
    }
    build_staging(resource, copy, first, on_demand)
}

/// A staging state for title texture `resource` from mip `first` on (as
/// [`staging`] decided it), its upload buffer made or taken from the pool;
/// nothing is detiled into it yet. Timed into [`STAGED_HERE`].
fn build_staging(resource: This, copy: &crate::resource::HostTexture, first: u32, on_demand: bool) -> Option<Arc<Staging>> {
    let started = std::time::Instant::now();
    let made = build_staging_untimed(resource, copy, first, on_demand);
    let us = started.elapsed().as_micros() as u64;
    STAGED_HERE.with(|s| {
        let mut now = s.get();
        now[0] += 1;
        now[1] += us;
        s.set(now);
    });
    made
}

fn build_staging_untimed(resource: This, copy: &crate::resource::HostTexture, first: u32, on_demand: bool) -> Option<Arc<Staging>> {
    let texture = &copy.texture;
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    // Render-target XOR layouts of one 2D slice and mip have addresses
    // (`Layout::addressable`); the other XOR modes do not.
    if !layout.addressable() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let level0 = copy.base;
    let mips = level0 + u32::from(d.MipLevels);
    let first = crate::resource::read_from(&layout, first).max(level0).min(mips - 1);
    let read = layout.bytes_from(first) as usize;
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        mips - first
    } else {
        (mips - first) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            first - level0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (piece, buffer, base, data) = match crate::pool::staging(host, total) {
        Some((piece, buffer, base, data)) => (Some(piece), buffer, base, data),
        None => {
            let started = std::time::Instant::now();
            let made = host.upload_buffer(total);
            let us = started.elapsed().as_micros() as u64;
            STAGED_HERE.with(|s| {
                let mut now = s.get();
                now[2] += 1;
                now[3] += us;
                s.set(now);
            });
            let (buffer, data) = made?;
            crate::host::name(&buffer, || {
                format!(
                    "staging of {resource:?} {}x{}x{} f{}",
                    d.Width, d.Height, d.DepthOrArraySize, d.Format.0
                )
            });
            (None, buffer, 0, data)
        }
    };
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    STAGED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    STAGED[1].fetch_add(total, std::sync::atomic::Ordering::Relaxed);
    let s = Arc::new(Staging {
        buffer,
        base,
        _piece: piece,
        #[cfg(feature = "diagnostics")]
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips,
        first,
        level0,
        read,
        serial: copy.serial,
        holds: copy.holds.clone(),
        once: std::sync::atomic::AtomicBool::new(once() && !on_demand && PICKING.get().is_some()),
        bytes: total,
        memory,
        last: Mutex::default(),
        dirty: std::sync::atomic::AtomicBool::new(true),
        uploaded: std::sync::atomic::AtomicBool::new(false),
        used: std::sync::atomic::AtomicU64::new(crate::host::moment()),
        hot: std::sync::atomic::AtomicU64::new(0),
        forced: std::sync::atomic::AtomicU32::new(0),
        fills: std::sync::atomic::AtomicU32::new(0),
        expected: std::sync::atomic::AtomicU64::new(0),
        on_demand,
        detiled: std::sync::atomic::AtomicU32::new(0),
        judged: [const { std::sync::atomic::AtomicU64::new(0) }; 2],
        waiting: std::sync::atomic::AtomicU64::new(0),
        exact: std::sync::atomic::AtomicBool::new(BAKING.get() && crate::draw::bake_sync()),
    });
    Some(s)
}

#[cfg(test)]
mod remap_tests {
    use std::sync::atomic::Ordering::Relaxed;

    /// After a remap of the memory under a texture that was already
    /// uploaded, the texture must be uploaded again (#574). A remap of
    /// unrelated memory, or of an empty range, must leave it alone.
    #[test]
    fn a_remap_under_a_staged_texture_makes_it_upload_again() {
        use windows_sys::Win32::System::Memory::*;
        const PAGE_RUN: usize = 0x1_0000;
        // Two 64 KiB runs: the texture lives in the first, the second is
        // memory next to it that no texture uses.
        // SAFETY: a private allocation of our own, released at the end.
        let base = unsafe {
            VirtualAlloc(std::ptr::null(), 2 * PAGE_RUN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
        };
        assert!(!base.is_null());
        let texture_start = base as usize;
        // Resource description words of a linear 4x4 RGBA8 texture with one
        // mip level, the shape the issue used to reproduce the bug.
        let description = [3, 0, 0, 0, 4, 0, 4, 0x1_0001, 28, 1, 0, 1, 0, 0];
        let staged = crate::resource::texture_at(texture_start as u64, &description).and_then(|resource| {
            let copy = crate::resource::host_copy(resource)?;
            super::staging(resource, &copy, 0)
        });
        // `None` means there is no GPU to make the host texture on: nothing
        // to check then.
        if let Some(staging) = staged {
            // The state after a submitted list uploaded the texture.
            staging.uploaded.store(true, Relaxed);
            staging.dirty.store(false, Relaxed);

            super::remapped(texture_start + PAGE_RUN, PAGE_RUN);
            super::remapped(texture_start, 0);
            assert!(!staging.dirty.load(Relaxed), "a remap elsewhere marked the texture");

            super::remapped(texture_start + 0x10, 0x10);
            assert!(staging.dirty.load(Relaxed), "a remap under the texture did not mark it");
        }
        // SAFETY: the allocation above, released once.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
    }

    /// A write the guards do not see under a staged texture that was
    /// uploaded (a texture draws read through a table) has it read again
    /// once the write settled: before that it is left alone, and a write
    /// elsewhere does nothing.
    #[test]
    fn a_page_write_under_an_uploaded_staged_texture_reads_it_again_once_settled() {
        use windows_sys::Win32::System::Memory::*;
        const PAGE_RUN: usize = 0x1_0000;
        // SAFETY: a private allocation of our own, released at the end.
        let base = unsafe {
            VirtualAlloc(std::ptr::null(), 2 * PAGE_RUN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
        };
        assert!(!base.is_null());
        let texture_start = base as usize;
        let description = [3, 0, 0, 0, 4, 0, 4, 0x1_0001, 28, 1, 0, 1, 0, 0];
        let staged = crate::resource::texture_at(texture_start as u64, &description).and_then(|resource| {
            let copy = crate::resource::host_copy(resource)?;
            super::staging(resource, &copy, 0)
        });
        // `None`: no GPU to make the host texture on.
        if let Some(staging) = staged {
            staging.uploaded.store(true, Relaxed);
            staging.dirty.store(false, Relaxed);
            staging.last.lock().unwrap().asked = 7;
            let at = texture_start as u64;
            // Frames far from any other test's, so the once-a-frame gate is ours.
            super::page_written(&[(at + PAGE_RUN as u64, at + 2 * PAGE_RUN as u64)], 1_000);
            super::page_written(&[(at + 0x10, at + 0x20)], 1_000);
            super::page_written(&[(at + 0x20, at + 0x30)], 1_100);
            assert_eq!(super::page_settled(1 << 40, 1_200, 200, 2_000), 0, "read again before its writes settled");
            assert!(!staging.dirty.load(Relaxed));
            assert_eq!(staging.last.lock().unwrap().asked, 7);
            assert_eq!(super::page_settled((1 << 40) + 1, 1_300, 200, 2_000), 1, "not read again once settled");
            assert!(staging.dirty.load(Relaxed), "not marked for the next list's upload");
            assert_eq!(staging.last.lock().unwrap().asked, 0, "what the staging buffer holds was not forgotten");
        }
        // SAFETY: the allocation above, released once.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
    }

    /// A checked host callback after a texture's step sees what the check
    /// found, also when wide checks would leave it to the batch's flush
    /// (hyphennn#1314).
    #[test]
    fn a_checked_host_callback_runs_after_the_texture_check() {
        use windows_sys::Win32::System::Memory::*;
        const PAGE_RUN: usize = 0x1_0000;
        // SAFETY: a private allocation of our own, released at the end.
        let base = unsafe { VirtualAlloc(std::ptr::null(), PAGE_RUN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!base.is_null());
        let description = [3, 0, 0, 0, 4, 0, 4, 0x1_0001, 28, 1, 0, 1, 0, 0];
        let staged = crate::resource::texture_at(base as u64, &description).and_then(|resource| {
            let copy = crate::resource::host_copy(resource)?;
            super::staging(resource, &copy, 0)
        });
        let list = crate::host::host().and_then(|host| host.new_list().ok());
        // `None`: no GPU to make the host texture on.
        if let (Some(staging), Some(list)) = (staged, list) {
            let state = |s: &super::Staging| {
                let last = s.last.lock().unwrap();
                (last.hash.is_some(), last.asked)
            };
            let before = state(&staging);
            let seen = std::sync::Arc::new(std::sync::Mutex::new(None));
            list.defer_texture(staging.clone());
            {
                let (staging, seen) = (staging.clone(), seen.clone());
                list.defer_host_checked(move || *seen.lock().unwrap() = Some(state(&staging)));
            }
            list.run_steps();
            let after = state(&staging);
            assert_ne!(before, after, "the texture was not checked");
            assert_eq!(*seen.lock().unwrap(), Some(after), "the callback ran before the check");
        }
        // SAFETY: the allocation above, released once.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
    }

    /// Settling: a write again under a settling key moves its last write
    /// on; keys are taken once quiet or after the longest wait, oldest
    /// first, within the count and byte bounds (the first whatever its size).
    #[test]
    fn settling_takes_quiet_or_old_keys_oldest_first_within_bounds() {
        let mut s = super::Settling::<usize>::new();
        assert!(!s.note(1, 100, 10, || 1));
        assert!(!s.note(2, 150, 10, || 2));
        assert!(s.note(1, 250, 10, || 99));
        // Key 1 was written at 250: not quiet at 400; key 2 is.
        assert_eq!(s.settled(400, 200, 2_000, 64, 1 << 20), vec![2]);
        // Written again and again, key 1 is taken 2 s after its first write.
        assert!(s.note(1, 2_050, 10, || 99));
        assert_eq!(s.settled(2_099, 200, 2_000, 64, 1 << 20), Vec::<usize>::new());
        assert_eq!(s.settled(2_100, 200, 2_000, 64, 1 << 20), vec![1]);
        assert_eq!(s.len(), 0);
        // Bounds: two a call, then bytes; the oldest go first.
        for (key, first) in [(5, 30), (6, 10), (7, 20)] {
            s.note(key, first, 100, || key);
        }
        assert_eq!(s.settled(1_000, 200, 2_000, 2, 1 << 20), vec![6, 7]);
        assert!(!s.note(8, 40, 1_000, || 8));
        assert_eq!(s.settled(1_000, 200, 2_000, 64, 150), vec![5]);
        assert_eq!(s.settled(1_000, 200, 2_000, 64, 150), vec![8]);
    }

    /// A partly mapped texture read with the bytes its pages held before
    /// its texels landed is read again the frame after the guards see the
    /// write, not 30 frames later; with no write, every 30th frame.
    #[test]
    fn a_partly_mapped_texture_is_read_again_after_a_write() {
        use super::partial_due;
        assert!(partial_due(None, 100, false));
        assert!(partial_due(None, 100, true));
        assert!(!partial_due(Some(100), 100, true));
        assert!(partial_due(Some(100), 101, true));
        assert!(!partial_due(Some(100), 101, false));
        assert!(!partial_due(Some(100), 129, false));
        assert!(partial_due(Some(100), 130, false));
    }
}

#[cfg(test)]
mod title_memory_tests {
    use super::with_title;

    /// Whether memory is all there is answered by the runtime's views of
    /// physical pages where they lie, and by Windows where none does.
    #[test]
    fn all_there_asks_the_views_where_they_lie() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        const BLOCK: usize = 64 << 10;
        // SAFETY: a reservation of our own, its first two blocks committed.
        let at = unsafe {
            let at = VirtualAlloc(std::ptr::null(), 4 * BLOCK, MEM_RESERVE, PAGE_READWRITE);
            VirtualAlloc(at, 2 * BLOCK, MEM_COMMIT, PAGE_READWRITE);
            at as usize
        };
        let there = |len| super::all_there(at as *const u8, len);
        assert!(there(2 * BLOCK));
        assert!(!there(3 * BLOCK));
        // The runtime's views of the first two blocks answer for them.
        for block in 0..2 {
            crate::title_memory_view(at + block * BLOCK, BLOCK, 9, 0);
        }
        assert!(there(2 * BLOCK));
        assert!(!there(3 * BLOCK));
        crate::inplace::views_gone(at, 4 * BLOCK);
        // SAFETY: our reservation, released whole.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }

    /// Title memory whose tail is reserved and not committed reads through
    /// the copy as its bytes and then zeros, without a fault; so does
    /// memory a page of which is taken away between two reads, and the
    /// zeros are zeros whatever the read buffer held before.
    #[test]
    fn an_uncommitted_tail_reads_as_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const HEAD: usize = 0xb_0000;
        const SIZE: usize = 0x20_0000;
        // SAFETY: a private reservation of our own; only committed pages
        // are written in place, and nothing else reads them in place.
        unsafe {
            // The read buffer holds something else first.
            let other = vec![0x77u8; SIZE];
            with_title(other.as_ptr(), SIZE, |bytes| assert!(bytes.iter().all(|&b| b == 0x77)));
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base.cast(), HEAD, MEM_COMMIT, PAGE_READWRITE), base.cast());
            base.write_bytes(0x5a, HEAD);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..HEAD].iter().all(|&b| b == 0x5a), "what is mapped");
                assert!(bytes[HEAD..].iter().all(|&b| b == 0), "what is not: zeros");
            });
            // The title takes a page in the middle away: zeros from then
            // on, the rest as before.
            assert_ne!(VirtualFree(base.add(0x4_0000).cast(), 0x1_0000, MEM_DECOMMIT), 0);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..0x4_0000].iter().all(|&b| b == 0x5a));
                assert!(bytes[0x4_0000..0x5_0000].iter().all(|&b| b == 0));
                assert!(bytes[0x5_0000..HEAD].iter().all(|&b| b == 0x5a));
                assert!(bytes[HEAD..].iter().all(|&b| b == 0));
            });
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// The texture both device-loss runs named last (16 texels wide, 4 or 8
    /// high, format 10, layout 0x100, console flags 0x300000; its memory
    /// began 0x300 into a page in run 73 and 0x700 in run 72, and here also
    /// 0xf00, so that a page line runs through it): what the detiling is
    /// given and what it writes are the same read in place and read through
    /// the copy, and the rows it writes end where the staging buffer the
    /// host sizes for it ends (896 bytes for 4 rows of 128 bytes 256 apart,
    /// 1,920 for 8).
    #[test]
    fn the_small_texture_of_runs_72_and_73_detiles_the_same_both_ways() {
        use windows_sys::Win32::System::Memory::*;
        for (height, within) in [(4u32, 0x300usize), (8, 0x700), (4, 0xf00), (8, 0xf00)] {
            // The description as the log printed it, word for word.
            let words: [u32; 14] = [3, 0, 0, 0, 16, 0, height, 65537, 10, 1, 0, 256, 3_145_728, 0];
            // SAFETY: fourteen words are a resource description (56 bytes,
            // plain numbers).
            let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
            let layout = crate::resource::texture_layout(&desc)
                .expect("a console layout")
                .planes[0]
                .layout
                .clone();
            let size = layout.size as usize;
            let (row, pitch) = (16 * 8, 256usize);
            assert!(size >= row * height as usize, "the layout holds every texel");
            // Committed pages and a reserved one after them; the texture
            // begins `within` bytes into the first, as the title placed it.
            // SAFETY: our own reservation; the committed part is written
            // and read in place, by this thread alone.
            unsafe {
                let pages = size.div_ceil(4096) + 1;
                let base = VirtualAlloc(std::ptr::null(), (pages + 1) * 4096, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
                assert!(!base.is_null());
                assert_eq!(VirtualAlloc(base.cast(), pages * 4096, MEM_COMMIT, PAGE_READWRITE), base.cast());
                for i in 0..pages * 4096 {
                    base.add(i).write((i * 7 + i / 251) as u8);
                }
                let memory = base.add(within);
                let in_place = std::slice::from_raw_parts(memory, size);
                // The bytes: the same in place and through the copy.
                with_title(memory, size, |copied| assert_eq!(copied, in_place));
                // What the detiling writes: the same, and inside the bytes
                // the host's staging buffer for this texture has.
                let staged = (height as usize - 1) * pitch + row;
                assert_eq!(staged, if height == 4 { 896 } else { 1920 });
                let detile = |tiled: &[u8]| {
                    // The slice `Staging::detile` hands over: whole rows.
                    let mut linear = vec![0xeeu8; pitch * height as usize];
                    layout
                        .detile(tiled, 0, 0, &mut linear, pitch, pitch * height as usize)
                        .expect("detiled");
                    linear
                };
                let direct = detile(in_place);
                let through = with_title(memory, size, |copied| detile(copied));
                assert_eq!(direct, through);
                for y in 0..height as usize {
                    assert!(direct[y * pitch..][..row].iter().any(|&b| b != 0xee), "row {y} written");
                    assert!(direct[y * pitch + row..(y + 1) * pitch].iter().all(|&b| b == 0xee), "nothing past row {y}");
                }
                // A buffer of exactly the staged bytes is enough.
                let mut exact = vec![0u8; staged];
                layout
                    .detile(in_place, 0, 0, &mut exact, pitch, pitch * height as usize)
                    .expect("the staging buffer's bytes are enough");
                assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
            }
        }
    }
}

/// [`claim_upload`] (ported from Belobug/VIctory-Belobug-AMD 5d6bd84).
#[cfg(test)]
mod prelude_upload_tests {
    use super::claim_upload;
    use std::sync::atomic::{AtomicBool, Ordering::Release};

    /// The first upload is claimed, a second list of the submission skips,
    /// an unchanged texture skips, a changed one is claimed in the same
    /// submission, every time it changes.
    #[test]
    fn first_upload_clean_skip_and_immediate_same_frame_reupload() {
        let uploaded = AtomicBool::new(false);
        let dirty = AtomicBool::new(true);
        assert_eq!(claim_upload(&uploaded, &dirty), Some(true));
        assert_eq!(claim_upload(&uploaded, &dirty), None);
        assert_eq!(claim_upload(&uploaded, &dirty), None);
        dirty.store(true, Release);
        assert_eq!(claim_upload(&uploaded, &dirty), Some(false));
        dirty.store(true, Release);
        assert_eq!(claim_upload(&uploaded, &dirty), Some(false));
    }

    /// A ring of three buffers written in turn (a video's) never shows a
    /// picture older than the one its buffer holds, a held picture
    /// included.
    #[test]
    fn three_buffer_ring_never_displays_stale_frame() {
        let ring: [(AtomicBool, AtomicBool); 3] = std::array::from_fn(|_| (AtomicBool::new(false), AtomicBool::new(true)));
        let mut shown = [0u32; 3];
        let schedule = [
            (0, 1, true),
            (1, 2, true),
            (2, 3, true),
            (0, 4, true),
            (1, 5, true),
            (1, 5, false),
            (2, 6, true),
            (0, 7, true),
            (1, 8, true),
            (2, 9, true),
            (0, 10, true),
            (1, 11, true),
        ];
        for (slot, picture, changed) in schedule {
            let (uploaded, dirty) = &ring[slot];
            if changed {
                dirty.store(true, Release);
            }
            if claim_upload(uploaded, dirty).is_some() {
                shown[slot] = picture;
            }
            assert_eq!(shown[slot], picture, "slot {slot} shows picture {} instead of {picture}", shown[slot]);
        }
    }
}

/// How long this PC's card takes to make the allocations a staged texture needs, for
/// `cargo test -p victory-gpu allocation_costs -- --ignored --nocapture`.
#[cfg(test)]
mod allocation_costs {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    fn texture(size: u32) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(size),
            Height: size,
            DepthOrArraySize: 1,
            MipLevels: 0,
            Format: DXGI_FORMAT_BC7_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        }
    }

    fn buffer(size: u64) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        }
    }

    fn committed(device: &ID3D12Device, kind: D3D12_HEAP_TYPE, flags: D3D12_HEAP_FLAGS, desc: &D3D12_RESOURCE_DESC) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
        let state = if kind == D3D12_HEAP_TYPE_UPLOAD { D3D12_RESOURCE_STATE_GENERIC_READ } else { D3D12_RESOURCE_STATE_COMMON };
        let mut out = None;
        // SAFETY: plain creation.
        unsafe { device.CreateCommittedResource(&heap, flags, desc, state, None, &mut out) }.expect("committed");
        out.unwrap()
    }

    fn timed(what: &str, n: usize, threads: usize, make: &(dyn Fn(usize) -> ID3D12Resource + Sync)) {
        let began = std::time::Instant::now();
        let all: Vec<_> = std::thread::scope(|s| {
            (0..threads)
                .map(|t| s.spawn(move || (t..n).step_by(threads).map(make).collect::<Vec<_>>()))
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        let took = began.elapsed().as_secs_f64() * 1e3;
        println!("{what}, {threads} thread(s): {n} in {took:.1} ms, {:.0} us each", took * 1e3 / n as f64);
        drop(all);
    }

    /// What a look at how much of a texture is there costs: by questions to
    /// the host, and by the runtime's record of its views.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn look_costs() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const BLOCK: usize = 64 << 10;
        for (name, blocks, there) in [("350 KB, all there", 6, 6), ("22 MB, all there", 350, 350), ("22 MB, 1 MB there", 350, 16)] {
            let make = || {
                // SAFETY: a reservation of our own, of which the first part
                // is committed and touched.
                unsafe {
                    let at = VirtualAlloc(std::ptr::null(), blocks * BLOCK, MEM_RESERVE, PAGE_READWRITE);
                    VirtualAlloc(at, there * BLOCK, MEM_COMMIT, PAGE_READWRITE);
                    std::ptr::write_bytes(at.cast::<u8>(), 1, there * BLOCK);
                    at as usize
                }
            };
            let (asked, noted) = (make(), make());
            for block in 0..there {
                crate::title_memory_view(noted + block * BLOCK, BLOCK, 9, 0);
            }
            for (how, at) in [("asking the host", asked), ("by the views", noted)] {
                const N: u32 = 200;
                let began = std::time::Instant::now();
                for _ in 0..N {
                    assert_eq!(super::mapped_bytes(at as *const u8, blocks * BLOCK), (there * BLOCK) as u64);
                }
                println!("{name}, {how}: {:.1} us a look", began.elapsed().as_secs_f64() * 1e6 / f64::from(N));
            }
            crate::inplace::views_gone(noted, blocks * BLOCK);
        }
    }

    /// The digest tells every length and every changed byte apart, zeros
    /// among them.
    #[test]
    fn a_digest_tells_bytes_apart() {
        use super::digest;
        let zeros = [0u8; 400];
        let mut seen: Vec<super::Digest> = (0..400).map(|n| digest(&zeros[..n])).collect();
        for at in 0..400 {
            for value in [1u8, 0x80, 0xff] {
                let mut bytes = zeros;
                bytes[at] = value;
                seen.push(digest(&bytes));
            }
        }
        let text: Vec<u8> = (0..400u32).map(|i| ((i * 37) >> 2) as u8).collect();
        for at in 0..399 {
            let mut bytes = text.clone();
            bytes.swap(at, at + 1);
            if bytes != text {
                seen.push(digest(&bytes));
            }
        }
        seen.push(digest(&text));
        let n = seen.len();
        seen.sort_unstable_by_key(|d| d.0);
        seen.dedup();
        assert_eq!(seen.len(), n);
        assert_eq!(digest(&text), digest(&text.clone()));
    }

    /// What a texture met for the first time costs, by step, in
    /// milliseconds a megabyte on one thread: for whether a store of
    /// detiled copies on disk could be worth it (its key is a hash of the
    /// title's bytes, so reading and hashing them stay; `VICTORY_BENCH_FILE`
    /// is a big file to read past the system's cache).
    #[test]
    #[ignore = "a measurement, not a check"]
    fn cold_texture_costs() {
        let Some(host) = crate::host::host() else { return };
        let desc = [3u32, 0, 0, 0, 1024, 0, 1024, 0xb_0001, 98, 1, 0, 0x105, 0, 0];
        // SAFETY: 56 bytes in the description's layout.
        let desc = unsafe { desc.as_ptr().cast::<crate::resource::ResourceDesc>().read_unaligned() };
        let layout = crate::resource::texture_layout(&desc).unwrap().planes[0].layout.clone();
        let size = layout.size as usize;
        let title: Vec<u8> = (0..size).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let (_buffer, up) = host.upload_buffer(2 * layout.size).unwrap();
        // SAFETY: the mapping.
        let up = unsafe { std::slice::from_raw_parts_mut(up, 2 * size) };
        let mut read = vec![0u8; size];
        const N: u32 = 200;
        let per_mb = |t: std::time::Duration| t.as_secs_f64() * 1e3 / (f64::from(N) * size as f64 / 1e6);
        let timed = |what: &str, f: &mut dyn FnMut()| {
            f();
            let started = std::time::Instant::now();
            (0..N).for_each(|_| f());
            println!("{what}: {:.3} ms/MB", per_mb(started.elapsed()));
        };
        timed("read through the kernel or a safe copy", &mut || {
            std::hint::black_box(crate::host::read_into(title.as_ptr(), &mut read));
        });
        timed("blake3", &mut || {
            std::hint::black_box(blake3::hash(&title));
        });
        timed("digest", &mut || {
            std::hint::black_box(super::digest(&title));
        });
        timed("detiled into upload memory", &mut || {
            let mut at = 0;
            for (mip, m) in layout.mips.iter().enumerate() {
                let pitch = (m.width as usize * 16).next_multiple_of(256);
                let len = pitch * m.height as usize;
                layout.detile(&title, mip as u32, 0, &mut up[at..at + len], pitch, len).unwrap();
                at += len.next_multiple_of(512);
            }
        });
        timed("copied into upload memory", &mut || up[..size].copy_from_slice(&title));
        if let Some(file) = std::env::var_os("VICTORY_BENCH_FILE") {
            use std::os::windows::fs::{FileExt, OpenOptionsExt};
            const NO_BUFFERING: u32 = 0x2000_0000;
            let file = std::fs::OpenOptions::new().read(true).custom_flags(NO_BUFFERING).open(file).unwrap();
            let len = file.metadata().unwrap().len();
            let mut block = vec![0u8; size + 4096];
            let skew = block.as_ptr().align_offset(4096);
            let n = size / 4096 * 4096;
            let mut at = 0u64;
            timed("read from the disk, one request at a time", &mut || {
                at = (at + 7919 * 4096 * 64) % (len - n as u64) / 4096 * 4096;
                file.seek_read(&mut block[skew..skew + n], at).unwrap();
            });
        }
    }

    #[test]
    #[ignore = "a measurement, not a check"]
    fn allocation_costs() {
        let Some(host) = crate::host::host() else { return };
        let device = &host.device;
        let not_zeroed = D3D12_HEAP_FLAG_CREATE_NOT_ZEROED;
        const N: usize = 1000;
        for size in [128u32, 512, 1024] {
            let (t, b) = (texture(size), buffer(u64::from(size * size) * 4 / 3 + 65536));
            println!("-- BC7 {size}x{size}");
            for threads in [1, 4, 8] {
                timed("texture committed", N, threads, &|_| committed(device, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, &t));
                timed("texture committed not zeroed", N, threads, &|_| committed(device, D3D12_HEAP_TYPE_DEFAULT, not_zeroed, &t));
                let mapped = |flags| {
                    let b = &b;
                    move |_| {
                        let r = committed(device, D3D12_HEAP_TYPE_UPLOAD, flags, b);
                        let mut data = std::ptr::null_mut();
                        // SAFETY: a live upload buffer.
                        unsafe { r.Map(0, None, Some(&mut data)) }.unwrap();
                        // SAFETY: the first byte of the mapping (its first touch).
                        unsafe { data.cast::<u8>().write_volatile(1) };
                        r
                    }
                };
                timed("staging committed and mapped", N, threads, &mapped(D3D12_HEAP_FLAG_NONE));
                timed("staging committed and mapped, not zeroed", N, threads, &mapped(not_zeroed));
            }
            // SAFETY: plain query.
            let info = unsafe { device.GetResourceAllocationInfo(0, &[t]) };
            let per_heap = ((64u64 << 20) / info.SizeInBytes).max(1) as usize;
            let heaps: Vec<ID3D12Heap> = (0..N.div_ceil(per_heap))
                .map(|_| {
                    let desc = D3D12_HEAP_DESC {
                        SizeInBytes: per_heap as u64 * info.SizeInBytes,
                        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES | not_zeroed,
                        ..Default::default()
                    };
                    let mut heap = None;
                    // SAFETY: plain creation.
                    unsafe { device.CreateHeap(&desc, &mut heap) }.expect("heap");
                    heap.unwrap()
                })
                .collect();
            let place = |i: usize| {
                let mut out: Option<ID3D12Resource> = None;
                let at = (i % per_heap) as u64 * info.SizeInBytes;
                // SAFETY: a part of a live heap no other texture has.
                unsafe { device.CreatePlacedResource(&heaps[i / per_heap], at, &t, D3D12_RESOURCE_STATE_COMMON, None, &mut out) }.expect("placed");
                out.unwrap()
            };
            timed("texture placed in 64 MB heaps", N, 1, &place);
            timed("texture placed in 64 MB heaps", N, 8, &place);
        }
    }
}

#[cfg(test)]
mod hot_texture_tests {
    use super::{HOT_FRAMES, Last, VERIFY_FRAMES};

    /// A video's planes cycle through a ring of three textures, each sampled
    /// and checked every third frame, and each found changed at every check:
    /// from the second change on the texture is hot, and stays so.
    #[test]
    fn a_texture_changing_at_every_check_goes_hot() {
        let mut last = Last::default();
        last.changed(10);
        assert!(!last.hash_due(10), "the check that found the change is done");
        assert!(last.hash_due(13), "the next check verifies it");
        last.changed(13);
        assert!(last.hash_due(14), "changed twice: hot");
        last.changed(16);
        assert!(last.hash_due(16 + HOT_FRAMES - 1));
    }

    /// A texture that streams in changes once: it is hashed once more at its
    /// next checks, found as it was, and not hashed again between turns.
    #[test]
    fn a_texture_that_changed_once_is_not_hot() {
        let mut last = Last::default();
        last.changed(100);
        assert!(last.hash_due(101), "verified at the next check");
        last.unchanged(101, true);
        assert!(last.hash_due(102), "the frame after the change proves nothing");
        last.unchanged(102, true);
        assert!(!last.hash_due(103));
        assert!(!last.hash_due(100 + VERIFY_FRAMES));
        // A change long after the first is a first change again.
        last.changed(500);
        assert!(!last.hash_due(500));
        assert!(last.hash_due(501));
    }

    /// At the start of the intro movie a plane of the ring was found
    /// changed in frame 1008, as it was at a check in frame 1009, and its
    /// next changes were passed over until its turn (run intro-t1): the
    /// check in the next frame does not end the verifying any more.
    #[test]
    fn a_check_in_the_next_frame_keeps_the_verifying() {
        let mut last = Last::default();
        last.changed(1008);
        last.unchanged(1009, true);
        assert!(last.hash_due(1011));
        last.changed(1011);
        assert!(last.hash_due(1012), "changed again: hot");
        // As before (`VICTORY_TEXTURE_HOT=1`): the check ended it.
        let mut old = Last::default();
        old.changed(1008);
        old.unchanged(1009, false);
        assert!(!old.hash_due(1011));
    }

    /// Lists record the copy while the next check may hash the texture.
    #[test]
    fn due_until_covers_the_verifying_and_the_hot_frames() {
        let mut last = Last::default();
        assert_eq!(last.due_until(), 0);
        last.changed(10);
        assert_eq!(last.due_until(), 10 + VERIFY_FRAMES + 1);
        last.changed(13);
        assert_eq!(last.due_until(), (13 + HOT_FRAMES).max(13 + VERIFY_FRAMES + 1));
        last.unchanged(20, true);
        assert_eq!(last.due_until(), 13 + HOT_FRAMES);
    }

    /// `VICTORY_TEXTURE_HOT_COPY=0`: a texture that streamed in once is
    /// copied only by the list after its check, a hot one by every list.
    #[test]
    fn copy_until_leaves_out_the_verifying_unless_asked() {
        let mut last = Last::default();
        last.changed(10);
        assert_eq!(last.copy_until(false), 0);
        assert_eq!(last.copy_until(true), 10 + VERIFY_FRAMES + 1);
        last.changed(13);
        assert!(last.copy_until(false) >= 13 + HOT_FRAMES);
        assert!(last.copy_until(true) >= 13 + HOT_FRAMES);
    }

    /// A hot texture that stops changing falls back to the turns.
    #[test]
    fn a_hot_texture_cools_down() {
        let mut last = Last::default();
        last.changed(1);
        last.changed(4);
        assert!(last.hash_due(4 + HOT_FRAMES - 1));
        last.unchanged(10, true);
        assert!(!last.hash_due(4 + HOT_FRAMES + 1));
    }
}

#[cfg(test)]
mod alias_copy_tests {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    const WIDTH: u64 = 37;
    const HEIGHT: u32 = 19;
    const LEVELS: u16 = 3;
    const LAYERS: u16 = 2;
    const PARTS: usize = (LEVELS * LAYERS) as usize;

    fn texture(
        device: &ID3D12Device,
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
    ) -> ID3D12Resource {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: WIDTH,
            Height: HEIGHT,
            DepthOrArraySize: LAYERS,
            MipLevels: LEVELS,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: flags,
            ..Default::default()
        };
        let mut texture = None;
        unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut texture,
                )
                .unwrap();
        }
        texture.unwrap()
    }

    fn buffer(
        device: &ID3D12Device,
        heap: D3D12_HEAP_TYPE,
        bytes: u64,
        state: D3D12_RESOURCE_STATES,
        flags: D3D12_RESOURCE_FLAGS,
    ) -> ID3D12Resource {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut buffer = None;
        unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: heap,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    state,
                    None,
                    &mut buffer,
                )
                .unwrap();
        }
        buffer.unwrap()
    }

    fn subresource(texture: &ID3D12Resource, index: usize) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: index as u32,
            },
        }
    }

    fn placed(
        buffer: &ID3D12Resource,
        row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    ) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *row,
            },
        }
    }

    /// Every word a different one, the first few the edges of the shared
    /// exponent format (exponent 0 and 31, all mantissas set, the top bit).
    fn word(part: usize, texel: usize) -> u32 {
        const EDGES: [u32; 8] = [
            0,
            u32::MAX,
            0xf800_0000,
            0x07ff_ffff,
            0x8000_0000,
            0x0000_01ff,
            0x0003_fe00,
            0x07fc_0000,
        ];
        if part == 0 && texel < EDGES.len() {
            return EDGES[texel];
        }
        let mut x = (part as u64) << 32 | texel as u64;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (x ^ (x >> 31)) as u32
    }

    /// Fills a `written` texture with [`word`]s, copies it into a `read`
    /// one as [`super::alias`] does, straight across or through a buffer,
    /// and reads every level and layer of the result back.
    fn copied(
        written: (DXGI_FORMAT, D3D12_RESOURCE_FLAGS),
        read: (DXGI_FORMAT, D3D12_RESOURCE_FLAGS),
        direct: bool,
    ) -> Vec<Vec<u32>> {
        let host = crate::host::host().unwrap();
        let device = &host.device;
        let from = texture(device, written.0, written.1);
        let to = texture(device, read.0, read.1);
        let mut rows = [[D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); PARTS]; 2];
        let mut bytes = [0u64; 2];
        unsafe {
            for (i, texture) in [&from, &to].into_iter().enumerate() {
                device.GetCopyableFootprints(
                    &texture.GetDesc(),
                    0,
                    PARTS as u32,
                    0,
                    Some(rows[i].as_mut_ptr()),
                    None,
                    None,
                    Some(&mut bytes[i]),
                );
            }
        }
        assert_eq!(bytes[0], bytes[1]);
        for (a, b) in rows[0].iter().zip(&rows[1]) {
            assert_eq!(a.Offset, b.Offset);
            let (a, b) = (a.Footprint, b.Footprint);
            assert_eq!((a.Width, a.Height, a.RowPitch), (b.Width, b.Height, b.RowPitch));
        }
        let rows = rows[0];
        let bytes = bytes[0];
        let upload = buffer(device, D3D12_HEAP_TYPE_UPLOAD, bytes, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
        let back = buffer(device, D3D12_HEAP_TYPE_READBACK, bytes, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
        let between = buffer(device, D3D12_HEAP_TYPE_DEFAULT, bytes, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        unsafe {
            let mut data = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut data)).unwrap();
            for (part, row) in rows.iter().enumerate() {
                let f = row.Footprint;
                for y in 0..f.Height as usize {
                    let line = data.cast::<u8>().add(row.Offset as usize + y * f.RowPitch as usize).cast::<u32>();
                    for x in 0..f.Width as usize {
                        line.add(x).write_unaligned(word(part, y * f.Width as usize + x));
                    }
                }
            }
            upload.Unmap(0, None);
        }
        let list = host.new_list().unwrap();
        let cmds = &list.list;
        let (copy_dest, copy_source, common) = (
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        );
        unsafe {
            for (part, row) in rows.iter().enumerate() {
                cmds.CopyTextureRegion(&subresource(&from, part), 0, 0, 0, &placed(&upload, row), None);
            }
            cmds.ResourceBarrier(&[crate::host::transition(&from, copy_dest, copy_source)]);
            if direct {
                for part in 0..PARTS {
                    cmds.CopyTextureRegion(&subresource(&to, part), 0, 0, 0, &subresource(&from, part), None);
                }
            } else {
                cmds.ResourceBarrier(&[crate::host::transition(&between, common, copy_dest)]);
                for (part, row) in rows.iter().enumerate() {
                    cmds.CopyTextureRegion(&placed(&between, row), 0, 0, 0, &subresource(&from, part), None);
                }
                cmds.ResourceBarrier(&[crate::host::transition(&between, copy_dest, copy_source)]);
                for (part, row) in rows.iter().enumerate() {
                    cmds.CopyTextureRegion(&subresource(&to, part), 0, 0, 0, &placed(&between, row), None);
                }
            }
            cmds.ResourceBarrier(&[crate::host::transition(&to, copy_dest, copy_source)]);
            for (part, row) in rows.iter().enumerate() {
                cmds.CopyTextureRegion(&placed(&back, row), 0, 0, 0, &subresource(&to, part), None);
            }
        }
        list.close();
        host.execute(&[&list]);
        unsafe { device.GetDeviceRemovedReason().unwrap() };
        let mut parts = Vec::new();
        unsafe {
            let mut data = std::ptr::null_mut();
            back.Map(0, Some(&D3D12_RANGE { Begin: 0, End: bytes as usize }), Some(&mut data)).unwrap();
            for row in &rows {
                let f = row.Footprint;
                let mut words = Vec::new();
                for y in 0..f.Height as usize {
                    let line = data.cast::<u8>().add(row.Offset as usize + y * f.RowPitch as usize).cast::<u32>();
                    words.extend((0..f.Width as usize).map(|x| line.add(x).read_unaligned()));
                }
                parts.push(words);
            }
            back.Unmap(0, None);
        }
        parts
    }

    /// `VICTORY_ALIAS_DIRECT=1`: the copy straight across must leave the
    /// same bits as the copy through a buffer, in both directions, from a
    /// texture the host writes as a render target or with a shader too.
    #[test]
    fn a_direct_alias_copy_keeps_every_bit() {
        if crate::host::host().is_none() {
            eprintln!("skipping alias copy readback: no D3D12 device");
            return;
        }
        let none = D3D12_RESOURCE_FLAG_NONE;
        let shared = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
        let cases = [
            ((DXGI_FORMAT_R32_UINT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), (shared, none)),
            ((DXGI_FORMAT_R32_UINT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET), (shared, none)),
            ((DXGI_FORMAT_R32_SINT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), (shared, none)),
            ((shared, none), (DXGI_FORMAT_R32_UINT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)),
            ((shared, none), (DXGI_FORMAT_R32_UINT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)),
            ((shared, none), (DXGI_FORMAT_R32_SINT, none)),
        ];
        for (written, read) in cases {
            let direct = copied(written, read, true);
            let through = copied(written, read, false);
            let mut texels = 0;
            for (part, (direct, through)) in direct.iter().zip(&through).enumerate() {
                for (texel, (&d, &t)) in direct.iter().zip(through).enumerate() {
                    let want = word(part, texel);
                    assert_eq!(t, want, "through a buffer, {written:?} to {read:?}, part {part} texel {texel}");
                    assert_eq!(d, want, "straight across, {written:?} to {read:?}, part {part} texel {texel}");
                }
                texels += direct.len();
            }
            assert_eq!(texels, (37 * 19 + 18 * 9 + 9 * 4) * LAYERS as usize);
        }
    }
}

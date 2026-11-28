//! Host memory for the textures the title fills, and for their staging
//! buffers, cut from blocks made once.
//!
//! Each such texture had an allocation of its own and its staging buffer
//! another. On an RX 7900 XTX an allocation takes about 310 microseconds
//! whatever its size, on one thread (`texture::allocation_costs`), and a
//! texture cut from a block 6: the pass that makes the textures shaders
//! pick by index (`draw.rs` `heap_textures`) got through 350-500 of the
//! 24,000 queued when the world loads in each of its 250 ms turns, some
//! forty seconds of a world without its textures.
//! `VICTORY_TEXTURE_POOL=0` makes every one an allocation of its own again.
//!
//! A piece given back is used again only two seconds later (nothing the card
//! was told before that is still running: Windows resets a card that takes
//! that long over one submission), and by then no list names it (the owners' rule, as for allocations of their own): a
//! texture's piece comes back when the last reference to the host texture
//! goes (an object of ours the texture holds says so), a staging buffer's
//! when the last list that recorded its copy is reset.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{GUID, HRESULT, IUnknown, Interface};

/// Bytes of a block; pieces of more than half of one are not pooled (with
/// blocks made ahead, [`ahead`]: of more than one).
const BLOCK: u64 = 64 << 20;
/// Blocks with nothing in them that are kept.
const SPARE: usize = 2;
/// The same while the card has room ([`room`]): 4 GiB of them.
const SPARE_KEPT: usize = 64;
/// The same of blocks of staging buffers, which are the PC's memory: run
/// crash-1 held 5.2 GB of them.
const STAGING_KEPT: usize = 8;
/// How many are kept now.
static SPARES: AtomicUsize = AtomicUsize::new(SPARE);

/// Sets how many empty blocks stay, by the card's memory (budget and use):
/// [`SPARE_KEPT`] below 85 % ([`crate::host::keep_freed`]), else [`SPARE`].
/// Asked once a second. Fewer than before lets the empty texture blocks
/// past them go now ([`reclaim_on_tick`]), not when the next texture is made:
/// up to 4 GiB of them stayed on a card short of room until then.
pub(crate) fn room(memory: Option<[u64; 4]>) {
    let limit = spares(memory, crate::host::keep_freed());
    SPARES.store(texture_spares(memory, limit, spare_scale().then(ahead)), Ordering::Relaxed);
    if stats_on() {
        static ASKED: AtomicU64 = AtomicU64::new(0);
        if ASKED.fetch_add(1, Ordering::Relaxed).is_multiple_of(5) {
            stats();
        }
    }
    if !reclaim_on_tick() {
        return;
    }
    // A full pool can become idle while another subsystem needs the VRAM.
    // A busy allocator observes the new limit on its next allocation or tick.
    let tick = now();
    if let Ok(mut pool) = TEXTURES.try_lock() {
        pool.reclaim(tick, limit, true);
    }
    if let Ok(mut pool) = STAGING.try_lock() {
        pool.reclaim(tick, limit, true);
    }
}

/// Whether the once a second check also lets go of the empty blocks past the
/// limit, not only the next allocation. On by default; `VICTORY_POOL_RECLAIM=0`
/// for before.
fn reclaim_on_tick() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_POOL_RECLAIM") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_POOL_RECLAIM").map_or(true, |v| v.trim() != "0"))
}

/// Whether a pool whose block was refused waits 500 ms before asking for
/// another (`VICTORY_POOL_BACKOFF=1`; off, every miss asks again, as before).
fn backoff() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_POOL_BACKOFF").map_or(cfg!(test), |v| v.trim() == "1"))
}

/// `VICTORY_SPARES_BY_HEADROOM=1` (off by default): how many empty blocks
/// are kept follows the card's headroom below 85 % of its budget, instead
/// of the all-or-nothing 4 GiB of [`SPARE_KEPT`]. A 16 GB card 12 GB in
/// held 4 GiB of empty blocks with about a GB of its budget left, and the
/// paging that followed ran at about a frame a second (AvatarManz#94).
/// At most half of what is left is kept, so what the title makes next has
/// room; a 24 GB card 10 GB in keeps all 64 blocks, and one at 85 % or
/// past it, or asked not to keep (memory pressure, [`room`]), keeps
/// [`SPARE`] as before.
fn by_headroom() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPARES_BY_HEADROOM").is_some_and(|v| v == "1"))
}

/// `VICTORY_SPARE_SCALE=1` caps empty texture blocks by the DXGI budget.
fn spare_scale() -> bool {
    static SCALE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SCALE.get_or_init(|| std::env::var_os("VICTORY_SPARE_SCALE").is_some_and(|v| v == "1"))
}

fn texture_spares(memory: Option<[u64; 4]>, limit: usize, ripe_ahead: Option<usize>) -> usize {
    match (memory, ripe_ahead) {
        (Some([budget, ..]), Some(ripe_ahead)) => {
            // Both Ripe::now and the staging pool's rested() read SPARES.
            let scaled = (budget / 32 / BLOCK)
                .max(ripe_ahead as u64)
                .max(STAGING_KEPT as u64)
                .min(SPARE_KEPT as u64);
            limit.min(scaled as usize)
        }
        _ => limit,
    }
}

fn spares(memory: Option<[u64; 4]>, keep: bool) -> usize {
    spares_of(memory, keep, by_headroom())
}

/// [`spares`], the choice made for it: [`SPARE_KEPT`] below 85 % of the
/// budget as before, or with `by_headroom` at most half of what is left
/// below it, in blocks.
fn spares_of(memory: Option<[u64; 4]>, keep: bool, by_headroom: bool) -> usize {
    let Some([budget, used, ..]) = memory else {
        return SPARE;
    };
    if !keep {
        return SPARE;
    }
    if !by_headroom {
        return if u128::from(used) * 100 < u128::from(budget) * 85 { SPARE_KEPT } else { SPARE };
    }
    let ceiling = u128::from(budget) * 85 / 100;
    let headroom = ceiling.saturating_sub(u128::from(used));
    ((headroom / 2 / u128::from(BLOCK)) as usize).clamp(SPARE, SPARE_KEPT)
}

/// Milliseconds until a piece given back is used again. (It was three
/// frames, a tenth of a second while the title shows a loading screen at 30
/// a second, with up to 18 submissions in the air.)
const REST: u64 = 2500;

fn now() -> u64 {
    crate::host::now_ms()
}

pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_POOL").is_none_or(|v| v != "0"))
}

const DEFAULT_RIPE_BLOCKS: usize = 12;

/// How many blocks of textures are made ahead, so that one has been there
/// for a while when its first piece is taken (`VICTORY_RIPE_BLOCKS`, 12;
/// 0: none, and new blocks are used at once, as before).
///
/// A page fault of lists already on the card, in a texture made the same
/// tenth of a second as the first piece of a block made for it, lost the
/// device in runs n08 and n11 (2026-11-25): the title gives a slot that
/// those lists read by index a new texture. Why the card does not have the
/// new block's addresses yet is not known (n11 waited for `MakeResident`;
/// `indirect::tests::new_blocks_under_readers` never loses it). A block that
/// every submission begun before its making has left behind ([`Ripe`]) has
/// no such lists: pieces are cut from those first.
fn ahead() -> usize {
    static AHEAD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *AHEAD.get_or_init(|| {
        std::env::var("VICTORY_RIPE_BLOCKS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(DEFAULT_RIPE_BLOCKS)
    })
}

/// Submissions begun and finished ([`crate::host::submissions`]), and how
/// many blocks to make ahead (0: every block is ripe at once).
#[derive(Clone, Copy)]
struct Ripe {
    started: u64,
    finished: u64,
    ahead: usize,
}

impl Ripe {
    const ALL: Ripe = Ripe { started: 0, finished: 0, ahead: 0 };

    fn now() -> Ripe {
        let (started, finished) = crate::host::submissions();
        // Not more than are kept ([`room`]): they would go and be made again.
        Ripe { started, finished, ahead: ahead().min(SPARES.load(Ordering::Relaxed)) }
    }

    fn is<B>(self, block: &Block<B>) -> bool {
        self.ahead == 0 || block.made <= self.finished
    }
}

/// Pieces cut from blocks that were not ripe, for want of a ripe one.
static UNRIPE: AtomicU64 = AtomicU64::new(0);

/// Whether a texture goes to the fullest ripe block with room
/// (`VICTORY_POOL_FULLEST`, on; 0: the first, as before).
///
/// The first with room left textures spread thin: play-d held 3,237 MB of
/// textures in 7,232 MB of blocks, play-e 5,269 MB in 7,168 MB, and a block
/// is let go only when its last texture goes. Filled fullest first, the
/// emptiest get no new ones and empty as theirs are let go.
fn fullest() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_POOL_FULLEST").is_none_or(|v| v != "0"))
}

/// The free ranges of a block: start and length, never two that touch.
struct Ranges(BTreeMap<u64, u64>);

impl Ranges {
    fn new(size: u64) -> Ranges {
        Ranges(BTreeMap::from([(0, size)]))
    }

    /// The lowest place for `len` bytes at a multiple of `align`.
    fn take(&mut self, len: u64, align: u64) -> Option<u64> {
        let (start, size, at) = self.0.iter().find_map(|(&start, &size)| {
            let at = start.next_multiple_of(align);
            (at + len <= start + size).then_some((start, size, at))
        })?;
        self.0.remove(&start);
        if at > start {
            self.0.insert(start, at - start);
        }
        if at + len < start + size {
            self.0.insert(at + len, start + size - at - len);
        }
        Some(at)
    }

    fn give(&mut self, mut start: u64, mut len: u64) {
        if let Some((&before, &size)) = self.0.range(..start).next_back()
            && before + size == start
        {
            self.0.remove(&before);
            (start, len) = (before, len + size);
        }
        if let Some(size) = self.0.remove(&(start + len)) {
            len += size;
        }
        self.0.insert(start, len);
    }

    fn whole(&self, size: u64) -> bool {
        self.0.get(&0) == Some(&size)
    }
}

/// A part of a block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Piece {
    block: usize,
    start: u64,
    len: u64,
}

struct Block<B> {
    memory: B,
    free: Ranges,
    /// The end of the highest piece ever taken: above it the memory is as
    /// it was made (zeros).
    used_to: u64,
    /// Bytes of its pieces taken, the resting ones too.
    taken: u64,
    /// The newest submission begun when it was made ([`Ripe`]).
    made: u64,
    /// When that was (ms).
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    since: u64,
}

struct Pool<B> {
    /// `None`: a block let go, its place to be used again.
    blocks: Vec<Option<Block<B>>>,
    /// Pieces given back, and when each may be used again.
    resting: VecDeque<(u64, Piece)>,
    /// The empty blocks it was last told to keep, and the most it keeps.
    spare: usize,
    most: usize,
    /// No block is made before this, after one was refused.
    retry_growth_at: u64,
    /// Blocks let go so far.
    #[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
    let_go: u64,
    /// Blocks being made ahead by a thread outside the lock ([`take_from`]),
    /// counted as empty ones so that no two threads make the same one.
    making_ahead: usize,
}

impl<B: Clone> Pool<B> {
    const fn new(most: usize) -> Pool<B> {
        Pool { blocks: Vec::new(), resting: VecDeque::new(), spare: 0, most, retry_growth_at: 0, let_go: 0, making_ahead: 0 }
    }

    /// A piece of `len` bytes and the memory it lies in, and whether that
    /// part was ever used before; `make` makes a block when none has room.
    /// (`make` runs with the pool borrowed; the pools of the process are
    /// taken from with [`take_from`], which makes blocks with their lock let go.)
    #[cfg(test)]
    fn take(&mut self, len: u64, align: u64, now: u64, make: impl FnMut() -> Option<B>) -> Option<(Piece, B, bool)> {
        self.take_ripe(len, align, now, Ripe::ALL, false, make)
    }

    /// The same from a block that is ripe, if one has room (with `fullest`,
    /// the one with the most taken); `make` makes one ahead too, when few
    /// are empty.
    #[cfg(test)]
    fn take_ripe(
        &mut self,
        len: u64,
        align: u64,
        now: u64,
        ripe: Ripe,
        fullest: bool,
        make: impl FnMut() -> Option<B>,
    ) -> Option<(Piece, B, bool)> {
        self.take_ripe_with_policy(len, align, now, ripe, fullest, false, make)
    }

    /// As [`Pool::take_ripe`]; with `best_fit`, the smallest free range that
    /// fits ([`Pool::best_fit_ripe`]) in place of the fullest or first block.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn take_ripe_with_policy(
        &mut self,
        len: u64,
        align: u64,
        now: u64,
        ripe: Ripe,
        fullest: bool,
        best_fit: bool,
        mut make: impl FnMut() -> Option<B>,
    ) -> Option<(Piece, B, bool)> {
        let (i, start) = match self.cut(len, align, now, ripe, fullest, best_fit)? {
            Some(found) => found,
            None => {
                let memory = self.grow(now, &mut make)?;
                self.add_and_cut(memory, ripe.started, len, align)?
            }
        };
        if self.short_ahead(ripe)
            && let Some(memory) = self.grow(now, &mut make)
        {
            self.add(memory, ripe.started);
        }
        self.taken(i, start, len, ripe)
    }

    /// Where a piece of `len` bytes was cut from a block there is: `None`
    /// if it is never pooled, `Some(None)` if no block has room.
    fn cut(
        &mut self,
        len: u64,
        align: u64,
        now: u64,
        ripe: Ripe,
        fullest: bool,
        best_fit: bool,
    ) -> Option<Option<(usize, u64)>> {
        // With blocks made ahead, what would be an allocation of its own (and
        // new when the card first reads it) lies in one of those too.
        if len == 0 || len > if ripe.ahead > 0 { BLOCK } else { BLOCK / 2 } {
            return None;
        }
        self.rested(now);
        let mut found = None;
        if best_fit {
            found = self.best_fit_ripe(len, align, ripe);
        } else if fullest {
            let mut order: Vec<(u64, usize)> = self
                .blocks
                .iter()
                .enumerate()
                .filter_map(|(i, b)| b.as_ref().filter(|b| ripe.is(b) && BLOCK - b.taken >= len).map(|b| (b.taken, i)))
                .collect();
            order.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            found = order.into_iter().find_map(|(_, i)| Some((i, self.blocks[i].as_mut()?.free.take(len, align)?)));
        } else {
            for (i, block) in self.blocks.iter_mut().enumerate() {
                if let Some(block) = block
                    && ripe.is(block)
                    && let Some(start) = block.free.take(len, align)
                {
                    found = Some((i, start));
                    break;
                }
            }
        }
        if found.is_none() && ripe.ahead > 0 {
            // The one that has been there the longest.
            let mut unripe: Vec<(u64, usize)> =
                self.blocks.iter().enumerate().filter_map(|(i, b)| Some((b.as_ref()?.made, i))).collect();
            unripe.sort_unstable();
            found = unripe.into_iter().find_map(|(_, i)| Some((i, self.blocks[i].as_mut()?.free.take(len, align)?)));
        }
        Some(found)
    }

    /// Adds a block just made and cuts the piece from it.
    fn add_and_cut(&mut self, memory: B, made: u64, len: u64, align: u64) -> Option<(usize, u64)> {
        let i = self.add(memory, made);
        Some((i, self.blocks[i].as_mut()?.free.take(len, align)?))
    }

    /// Whether a block is to be made ahead: fewer are empty (or being made)
    /// than are wanted.
    fn short_ahead(&self, ripe: Ripe) -> bool {
        ripe.ahead > 0
            && self.blocks.iter().flatten().filter(|b| b.free.whole(BLOCK)).count() + self.making_ahead < ripe.ahead
    }

    /// The piece cut at `start` of block `i`, its memory, and whether it was
    /// used before.
    fn taken(&mut self, i: usize, start: u64, len: u64, ripe: Ripe) -> Option<(Piece, B, bool)> {
        let block = self.blocks[i].as_mut()?;
        if ripe.ahead > 0 && !ripe.is(block) {
            let n = UNRIPE.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::info!("textures cut from blocks made while lists ran that still run: {n}");
            }
        }
        let again = start < block.used_to;
        block.used_to = block.used_to.max(start + len);
        block.taken += len;
        Some((Piece { block: i, start, len }, block.memory.clone(), again))
    }

    /// A failed 64 MiB block must not trigger another driver call for each
    /// small texture. Existing free ranges remain usable during backoff.
    fn may_grow(&self, now: u64) -> bool {
        !backoff() || now >= self.retry_growth_at
    }

    fn grew(&mut self, now: u64, made: bool) {
        if backoff() && !made {
            self.retry_growth_at = now.saturating_add(500);
        }
    }

    #[cfg(test)]
    fn grow(&mut self, now: u64, make: &mut impl FnMut() -> Option<B>) -> Option<B> {
        if !self.may_grow(now) {
            return None;
        }
        let block = make();
        self.grew(now, block.is_some());
        block
    }

    /// Choose the smallest fitting free interval in a ripe block, breaking
    /// ties by alignment padding, block index and offset. An exact fit
    /// cannot be improved. Leave the unripe fallback and quarantine alone.
    fn best_fit_ripe(&mut self, len: u64, align: u64, ripe: Ripe) -> Option<(usize, u64)> {
        let mut best: Option<(u64, u64, usize, u64)> = None;
        'search: for (i, block) in self.blocks.iter().enumerate() {
            let Some(block) = block else { continue };
            if !ripe.is(block) {
                continue;
            }
            for (&start, &size) in &block.free.0 {
                // Skip ranges that cannot fit or cannot beat the best size.
                if size < len || best.is_some_and(|old| size > old.0) {
                    continue;
                }
                let at = start.next_multiple_of(align);
                if at + len <= start + size {
                    let key = (size, at - start, i, at);
                    if best.is_none_or(|old| key < old) {
                        best = Some(key);
                    }
                    if size == len && at == start {
                        break 'search;
                    }
                }
            }
        }
        let (_, _, i, at) = best?;
        let ranges = &mut self.blocks[i].as_mut()?.free.0;
        let (&start, &size) = ranges.range(..=at).next_back()?;
        debug_assert!(at + len <= start + size);
        ranges.remove(&start);
        if at > start {
            ranges.insert(start, at - start);
        }
        if at + len < start + size {
            ranges.insert(at + len, start + size - at - len);
        }
        Some((i, at))
    }

    fn add(&mut self, memory: B, made: u64) -> usize {
        let i = match self.blocks.iter().position(Option::is_none) {
            Some(i) => i,
            None => {
                self.blocks.push(None);
                self.blocks.len() - 1
            }
        };
        self.blocks[i] =
            Some(Block { memory, free: Ranges::new(BLOCK), used_to: 0, taken: 0, made, since: crate::host::now_ms() });
        i
    }

    fn give(&mut self, piece: Piece, now: u64) {
        self.resting.push_back((now + REST, piece));
    }

    /// Makes the pieces that rested long enough free, and lets go of empty
    /// blocks past [`SPARE`].
    fn rested(&mut self, now: u64) {
        self.reclaim(now, SPARES.load(Ordering::Relaxed), false);
    }

    /// [`Pool::rested`], keeping at most `kept` empty blocks.
    #[cfg(test)]
    fn rested_under(&mut self, now: u64, kept: usize) {
        self.reclaim(now, kept, false);
    }

    fn reclaim(&mut self, now: u64, limit: usize, sweep: bool) {
        let mut any = false;
        while let Some(&(ready, piece)) = self.resting.front()
            && ready <= now
        {
            self.resting.pop_front();
            if let Some(block) = self.blocks[piece.block].as_mut() {
                block.free.give(piece.start, piece.len);
                block.taken = block.taken.saturating_sub(piece.len);
                any |= block.free.whole(BLOCK);
            }
        }
        // (Without a piece given back too, once there is less room.)
        let most = limit.min(self.most);
        if sweep || any || most < self.spare {
            self.spare = most;
            let mut spare = 0;
            for block in &mut self.blocks {
                if block.as_ref().is_some_and(|b| b.free.whole(BLOCK)) {
                    spare += 1;
                    if spare > most {
                        *block = None;
                        self.let_go += 1;
                    }
                }
            }
        }
    }

    fn blocks(&self) -> usize {
        self.blocks.iter().flatten().count()
    }

    /// Bytes of its blocks no piece holds, resting ones held too; with what
    /// rested long enough made free first, so a budget check can find room
    /// again without a texture taken.
    fn reusable_bytes(&mut self, now: u64) -> u64 {
        self.rested(now);
        self.blocks.iter().flatten().map(|b| BLOCK - b.taken).sum()
    }

    /// What the blocks hold now, for [`stats`].
    fn census(&self) -> Census {
        let mut c = Census::default();
        for block in self.blocks.iter().flatten() {
            c.blocks += 1;
            c.empty += u64::from(block.free.whole(BLOCK));
            let mut free = 0;
            for &len in block.free.0.values() {
                free += len;
                c.ranges += 1;
                c.largest = c.largest.max(len);
                if len >= 1 << 20 {
                    c.wide += len;
                }
            }
            c.free += free;
            // (Pieces that rest count as taken: they are not free yet.)
            c.full[((BLOCK - free) * 4 / BLOCK).min(3) as usize] += 1;
        }
        c.resting = self.resting.iter().map(|(_, piece)| piece.len).sum();
        c
    }
}

/// A pool's blocks at one moment: how many there are and how many of them
/// are empty; the bytes free in them now, in how many ranges, the largest of
/// those, and the bytes in ranges of 1 MiB and more; the bytes of pieces
/// given back that rest; and the blocks by how full they are, in quarters.
#[derive(Default)]
struct Census {
    blocks: u64,
    empty: u64,
    free: u64,
    ranges: u64,
    largest: u64,
    wide: u64,
    resting: u64,
    full: [u64; 4],
}

/// A mapped upload buffer's memory.
#[derive(Clone)]
struct Upload(ID3D12Resource, *mut u8);
// SAFETY: the mapping is written by the owners of its pieces, each its own.
unsafe impl Send for Upload {}

static TEXTURES: Mutex<Pool<ID3D12Heap>> = Mutex::new(Pool::new(SPARE_KEPT));
static STAGING: Mutex<Pool<Upload>> = Mutex::new(Pool::new(STAGING_KEPT));
static SHARED: Mutex<Pool<ID3D12Heap>> = Mutex::new(Pool::new(SHARED_KEPT));

/// Empty blocks of shared buffers kept at most (system memory).
const SHARED_KEPT: usize = 16;

fn lock<T>(m: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// [`Pool::take_ripe_with_policy`] of a pool behind a lock, with blocks made
/// while the lock is let go. Making one is a 64 MiB heap or upload buffer,
/// made resident and waited for until it is mapped (`crate::host::resident`,
/// `crate::host::mapped`): a tenth of a millisecond to some, but over a
/// second near the card's budget (`MakeResident` waits for the system to
/// evict first). Under the lock every thread that took a texture, or gave
/// one back, waited for that.
///
/// `ripe` is asked again once a block is made: a block is ripe once the
/// submissions begun before it was made have finished, so its mark is
/// taken after the making (a mark taken before would be too low).
///
/// Two threads that both find no room may each make a block; the second
/// is a block made ahead, or a spare that [`Pool::rested`] lets go.
fn take_from<B: Clone + 'static>(
    pool: &'static Mutex<Pool<B>>,
    len: u64,
    align: u64,
    ripe: impl Fn() -> Ripe,
    fullest: bool,
    best_fit: bool,
    mut make: impl FnMut() -> Option<B>,
) -> Option<(Piece, B, bool)> {
    let first = ripe();
    let at = now();
    let mut p = lock(pool);
    let (i, start) = match p.cut(len, align, at, first, fullest, best_fit)? {
        Some(found) => found,
        None => {
            if !p.may_grow(at) {
                return None;
            }
            drop(p);
            let made = make();
            let mark = ripe().started;
            p = lock(pool);
            p.grew(at, made.is_some());
            p.add_and_cut(made?, mark, len, align)?
        }
    };
    let ahead = p.short_ahead(first) && p.may_grow(at);
    if ahead {
        p.making_ahead += 1;
    }
    let taken = p.taken(i, start, len, first);
    drop(p);
    if ahead {
        /// Counts the block as made (or not) even if `make` unwinds.
        struct Making<B: Clone + 'static>(&'static Mutex<Pool<B>>);
        impl<B: Clone + 'static> Drop for Making<B> {
            fn drop(&mut self) {
                let mut p = lock(self.0);
                p.making_ahead = p.making_ahead.saturating_sub(1);
            }
        }
        let making = Making(pool);
        let made = make();
        let mark = ripe().started;
        let mut p = lock(pool);
        // Counted off under the same lock the block is added under.
        std::mem::forget(making);
        p.making_ahead = p.making_ahead.saturating_sub(1);
        p.grew(at, made.is_some());
        if let Some(memory) = made {
            p.add(memory, mark);
        }
    }
    taken
}

/// Bytes of the pieces textures hold now.
static HELD: AtomicU64 = AtomicU64::new(0);
/// The same for shared buffers.
static SHARED_HELD: AtomicU64 = AtomicU64::new(0);

/// Bytes of the texture blocks no texture holds, pieces still resting
/// counted as held: the card's memory in use by us that new textures take
/// first. All blocks together; one texture that big may still not fit.
pub(crate) fn texture_room() -> u64 {
    lock(&TEXTURES).reusable_bytes(now())
}

/// For the memory line: blocks of textures and of staging buffers, in MiB.
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 2] {
    [lock(&TEXTURES).blocks() as u64 * (BLOCK >> 20), lock(&STAGING).blocks() as u64 * (BLOCK >> 20)]
}

/// For the memory line: texture blocks empty, under a quarter taken, under
/// half, and the rest; the MiB not taken in each of those; blocks let go so
/// far; whether new textures go to the fullest ([`fullest`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn fill() -> ([u64; 4], [u64; 4], u64, bool) {
    let pool = lock(&TEXTURES);
    let (mut blocks, mut free) = ([0u64; 4], [0u64; 4]);
    for b in pool.blocks.iter().flatten() {
        let i = match b.taken {
            0 => 0,
            t if t < BLOCK / 4 => 1,
            t if t < BLOCK / 2 => 2,
            _ => 3,
        };
        blocks[i] += 1;
        free[i] += BLOCK.saturating_sub(b.taken) >> 20;
    }
    (blocks, free, pool.let_go, fullest())
}

/// Whether the texture blocks are counted in the log every five seconds
/// (`VICTORY_POOL_STATS=1`; off by default): how full the blocks are, how
/// their free bytes lie, the textures by the size of their pieces, and what
/// the small ones would hold at multiples of 4 KiB,
/// which costs one more question of the host for each texture made.
///
/// What it showed on an RTX 4070 Laptop (8 GiB), Series S profile,
/// 2026-11-27, five minutes into a run with `VICTORY_POOL_SMALL=0`: 57
/// blocks (3,648 MB) for 2,387 MB of textures, 1,260 MB of them free in 1,655
/// ranges, 25 blocks less than half full and none empty; 13,363 of 15,836
/// textures in pieces of 64 KiB, and 13,748 that would hold 222 MB at
/// multiples of 4 KiB where they held 907 MB.
fn stats_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_POOL_STATS").is_some_and(|v| v == "1"))
}

/// The live textures by their piece's bytes (64 KiB and less, to 256 KiB, to
/// 1 MiB, to 4 MiB, more): how many, and the bytes.
static LIVE: [[AtomicU64; 2]; 5] = [const { [const { AtomicU64::new(0) }; 2] }; 5];
/// The live textures that could lie at multiples of 4 KiB and do not: how
/// many, the bytes their pieces hold, and the bytes they would hold there.
static SMALLER: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn class(len: u64) -> usize {
    match len {
        0..=0x1_0000 => 0,
        0x1_0001..=0x4_0000 => 1,
        0x4_0001..=0x10_0000 => 2,
        0x10_0001..=0x40_0000 => 3,
        _ => 4,
    }
}

/// A texture took `piece` (`made`) or gave it back; `small`: the bytes it
/// would hold at multiples of 4 KiB, 0 if it could not lie there or was not
/// asked ([`stats_on`]).
fn counted(piece: Piece, small: u64, made: bool) {
    let step = |counter: &AtomicU64, by: u64| {
        if made {
            counter.fetch_add(by, Ordering::Relaxed);
        } else {
            counter.fetch_sub(by, Ordering::Relaxed);
        }
    };
    let live = &LIVE[class(piece.len)];
    step(&live[0], 1);
    step(&live[1], piece.len);
    if small != 0 {
        step(&SMALLER[0], 1);
        step(&SMALLER[1], piece.len);
        step(&SMALLER[2], small);
    }
}

/// The log's line of [`stats_on`].
fn stats() {
    let c = lock(&TEXTURES).census();
    let live: [[u64; 2]; 5] = std::array::from_fn(|i| std::array::from_fn(|j| LIVE[i][j].load(Ordering::Relaxed)));
    let smaller: [u64; 3] = std::array::from_fn(|i| SMALLER[i].load(Ordering::Relaxed));
    let mb = |bytes: u64| bytes >> 20;
    tracing::info!(
        "texture blocks (VICTORY_POOL_STATS=1): {} of them ({} MB), {} MB held by {} textures, {} MB free now in {} ranges (the largest \
         {} KB; {} MB of it in ranges of 1 MiB and more), {} MB resting, {} blocks empty; blocks by how full: {} under a quarter, {} \
         under a half, {} under three quarters, {} above; textures by their piece: {} of 64 KiB ({} MB), {} to 256 KiB ({} MB), {} to \
         1 MiB ({} MB), {} to 4 MiB ({} MB), {} larger ({} MB); {} of them could lie at multiples of 4 KiB and would hold {} MB where \
         they hold {} MB",
        c.blocks,
        mb(c.blocks * BLOCK),
        mb(HELD.load(Ordering::Relaxed)),
        live.iter().map(|l| l[0]).sum::<u64>(),
        mb(c.free),
        c.ranges,
        c.largest >> 10,
        mb(c.wide),
        mb(c.resting),
        c.empty,
        c.full[0],
        c.full[1],
        c.full[2],
        c.full[3],
        live[0][0],
        mb(live[0][1]),
        live[1][0],
        mb(live[1][1]),
        live[2][0],
        mb(live[2][1]),
        live[3][0],
        mb(live[3][1]),
        live[4][0],
        mb(live[4][1]),
        smaller[0],
        mb(smaller[2]),
        mb(smaller[1]),
    );
}

/// Part of a block of mapped upload memory, back in the pool when dropped.
pub(crate) struct StagingPiece(Piece);

impl Drop for StagingPiece {
    fn drop(&mut self) {
        lock(&STAGING).give(self.0, now());
    }
}

/// `bytes` of upload memory, all zeros: the buffer it lies in, where in
/// that buffer, and its mapping. `None`: not pooled.
pub(crate) fn staging(host: &crate::host::Host, bytes: u64) -> Option<(StagingPiece, ID3D12Resource, u64, *mut u8)> {
    if !on() {
        return None;
    }
    // Copies from a buffer begin at multiples of 512.
    let bytes = bytes.next_multiple_of(512);
    let (piece, memory, again) = take_from(&STAGING, bytes, 512, || Ripe::ALL, false, false, || {
        let (buffer, data) = host.upload_buffer(BLOCK)?;
        crate::host::name(&buffer, || "staging of textures (a block)".into());
        Some(Upload(buffer, data))
    })?;
    let owned = StagingPiece(piece);
    // SAFETY: the piece lies in the block's mapping.
    let data = unsafe { memory.1.add(piece.start as usize) };
    if again {
        // What is not detiled into it (an unmapped texture) uploads zeros,
        // as from a buffer of its own.
        // SAFETY: the piece is ours alone.
        unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
    }
    Some((owned, memory.0, piece.start, data))
}

/// An object the host texture holds and releases when it goes: its piece
/// is free then.
#[repr(C)]
struct Reclaim {
    methods: &'static [*const (); 3],
    references: AtomicU32,
    piece: Piece,
    pool: &'static Mutex<Pool<ID3D12Heap>>,
    held: &'static AtomicU64,
    /// What [`counted`] was told of it.
    small: u64,
}

struct Methods([*const (); 3]);
// SAFETY: function addresses only.
unsafe impl Sync for Methods {}

static RECLAIM: Methods = Methods([query as *const (), add_ref as *const (), release as *const ()]);

unsafe extern "system" fn query(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    const E_NOINTERFACE: HRESULT = HRESULT(0x8000_4002_u32 as i32);
    // SAFETY: COM's arguments.
    unsafe {
        if out.is_null() {
            return HRESULT(0x8000_4003_u32 as i32);
        }
        if iid.as_ref().is_some_and(|iid| *iid == IUnknown::IID) {
            add_ref(this);
            *out = this;
            return HRESULT(0);
        }
        *out = std::ptr::null_mut();
    }
    E_NOINTERFACE
}

unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    // SAFETY: one of ours, alive while a reference to it is.
    unsafe { (*this.cast::<Reclaim>()).references.fetch_add(1, Ordering::Relaxed) + 1 }
}

unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    // SAFETY: as above; the last reference frees the box it was made as.
    unsafe {
        let left = (*this.cast::<Reclaim>()).references.fetch_sub(1, Ordering::AcqRel) - 1;
        if left == 0 {
            let gone = Box::from_raw(this.cast::<Reclaim>());
            gone.held.fetch_sub(gone.piece.len, Ordering::Relaxed);
            if std::ptr::eq(gone.pool, &TEXTURES) {
                counted(gone.piece, gone.small, false);
            }
            lock(gone.pool).give(gone.piece, now());
        }
        left
    }
}

/// Whether textures of 64 KiB and less are placed at multiples of 4 KiB in
/// their block, where the host allows it (as before; `VICTORY_POOL_SMALL=0`:
/// at multiples of 64 KiB, as every other texture).
///
/// Ported from Belobug/VIctory-Belobug-AMD c278f0d, where 64 KiB multiples
/// are the default: the two device losses of the night of 2026-11-25 with
/// the pool in that fork's report (runs steps-ab2 and start-3) faulted in
/// the same kind of texture: 256x256, block-compressed, 7 levels, placed at
/// a 4 KiB multiple that is no 64 KiB one (offsets 0x406000 and 0x3fe6000 of
/// their blocks), never used before, read by the card the moment its slot
/// was made; the descriptor's base of both lies 0xa000 into the piece, on a
/// 64 KiB multiple, and its first level of 32 KiB from there ends 0x2000
/// past the piece. Why the card faulted there is not proven; textures at
/// 64 KiB multiples have none of that, for some memory (at most 60 KiB a
/// small texture).
fn small_placement() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_POOL_SMALL") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_POOL_SMALL").is_none_or(|v| v != "0"))
}

/// Diagnostics: whether the card reads a pooled texture inside the piece it
/// was placed in. A block's address is taken as the lowest of its textures'
/// descriptor bases less their pieces' starts (a texture's first level may
/// lie past its piece's start, never before it); a texture whose first level
/// the card would read from past its piece's end is said loudly. (Ported
/// from Belobug/VIctory-Belobug-AMD c278f0d.)
#[cfg(feature = "diagnostics")]
fn where_the_card_reads(heap: &ID3D12Heap, texture: &ID3D12Resource, piece: Piece) {
    static BLOCKS: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());
    static SAID: AtomicU64 = AtomicU64::new(0);
    let Some(base) = crate::book::address(texture) else { return };
    let Some(seen) = base.checked_sub(piece.start) else { return };
    let raw = heap.as_raw() as usize;
    let block = {
        let mut blocks = BLOCKS.lock().unwrap_or_else(|e| e.into_inner());
        match blocks.iter_mut().find(|b| b.0 == raw) {
            Some(b) => {
                b.1 = b.1.min(seen);
                b.1
            }
            None => {
                if blocks.len() >= 4096 {
                    blocks.remove(0);
                }
                blocks.push((raw, seen));
                seen
            }
        }
    };
    let into = base - block - piece.start;
    // SAFETY: a live texture.
    let desc = unsafe { texture.GetDesc() };
    // The first level's bytes, as a copy would have them.
    let mut level = 0u64;
    if let Some(host) = crate::host::made_host() {
        // SAFETY: plain query of a valid description.
        unsafe { host.device.GetCopyableFootprints(&desc, 0, 1, 0, None, None, None, Some(&raw mut level)) };
    }
    if into + level > piece.len {
        let n = SAID.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 50 || n.is_power_of_two() {
            tracing::error!(
                times = n,
                base = format_args!("{base:#x}"),
                piece_start = format_args!("{:#x}", piece.start),
                piece_len = format_args!("{:#x}", piece.len),
                into = format_args!("{into:#x}"),
                first_level_bytes = format_args!("{level:#x}"),
                width = desc.Width,
                height = desc.Height,
                levels = desc.MipLevels,
                format = desc.Format.0,
                "POOL: the card reads a pooled texture's first level past the end of the piece it was placed in"
            );
        }
    }
    // Every placement offset and where the first level lies, a few of them.
    static SHOWN: AtomicU64 = AtomicU64::new(0);
    if into > 0 && SHOWN.fetch_add(1, Ordering::Relaxed) < 40 {
        tracing::info!(
            into = format_args!("{into:#x}"),
            piece_start = format_args!("{:#x}", piece.start),
            piece_len = format_args!("{:#x}", piece.len),
            width = desc.Width,
            height = desc.Height,
            levels = desc.MipLevels,
            format = desc.Format.0,
            alignment = desc.Alignment,
            "pool: a texture's first level lies past its piece's start"
        );
    }
}

/// Textures made in the last second ([`fresh`]), and when.
static RECENT: Mutex<VecDeque<(usize, u64)>> = Mutex::new(VecDeque::new());

/// Whether `texture` was made in a block in the last second.
pub(crate) fn fresh(texture: &ID3D12Resource) -> bool {
    let raw = texture.as_raw() as usize;
    let since = now().saturating_sub(1000);
    RECENT.lock().unwrap_or_else(|e| e.into_inner()).iter().rev().take_while(|r| r.1 >= since).any(|r| r.0 == raw)
}

fn made_now(texture: &ID3D12Resource) {
    let now = now();
    let raw = texture.as_raw() as usize;
    {
        let mut recent = RECENT.lock().unwrap_or_else(|e| e.into_inner());
        while recent.front().is_some_and(|r| r.1 + 1000 < now) {
            recent.pop_front();
        }
        recent.push_back((raw, now));
    }
    if !crate::descriptor::host::show_new_later() {
        return;
    }
    let begun = crate::host::submissions().0;
    NEWEST_BEGUN.fetch_max(begun, Ordering::Relaxed);
    let mut unseen = UNSEEN.lock().unwrap_or_else(|e| e.into_inner());
    let finished = crate::host::submissions().1;
    // Those every submission begun before them has left behind are no
    // one's concern any more (and a raw pointer may come again).
    unseen.retain(|_, b| *b > finished);
    unseen.insert(raw, begun);
}

/// Pooled textures made while submissions begun before them may still run:
/// the submissions begun when each was made (with
/// `crate::descriptor::host::show_new_later` only). Ported from
/// Belobug/VIctory-Belobug-AMD 4cd2d13.
static UNSEEN: std::sync::LazyLock<Mutex<std::collections::HashMap<usize, u64>>> = std::sync::LazyLock::new(Default::default);
/// The newest of those numbers (nothing to look up once it has finished).
static NEWEST_BEGUN: AtomicU64 = AtomicU64::new(0);

/// Whether a texture was made in a block while a submission begun before
/// it may still run ([`begun_before`]).
fn any_unseen() -> bool {
    NEWEST_BEGUN.load(Ordering::Relaxed) > crate::host::submissions().1
}

/// The submissions begun when `texture` was made in a block, if one of
/// those may still be on the card: lists already sent then must not find it
/// in a slot they read by index (`crate::descriptor::host::show_new_later`).
/// Every device loss with the pool in the report of
/// Belobug/VIctory-Belobug-AMD (runs n08, n11, n15, steps-ab2, start-3 and a
/// user's play-1125-2107) was a page fault at the first byte of a texture
/// made that very millisecond, its title slot changed "at once" while lists
/// sent before it read that slot by index.
pub(crate) fn begun_before(texture: &ID3D12Resource) -> Option<u64> {
    if !any_unseen() {
        return None;
    }
    let finished = crate::host::submissions().1;
    let raw = texture.as_raw() as usize;
    let unseen = UNSEEN.lock().unwrap_or_else(|e| e.into_inner());
    unseen.get(&raw).copied().filter(|&begun| begun > finished)
}

/// The private data a texture in a block carries ([`give_back_with`]).
const KEY: GUID = GUID::from_u128(0x6f0e2f0c_5a59_4b1e_9a3f_7f4d2c9be713);

/// Whether `texture` was made in a block (and so may lie where another
/// did): a texture made otherwise, such as a committed one the block
/// refused or a stand-in the PC upscaler makes, takes no aliasing barrier
/// (the host refuses a list with one on a committed resource).
pub(crate) fn pooled(texture: &ID3D12Resource) -> bool {
    let mut size = 0u32;
    // SAFETY: a size query; no data is written.
    unsafe { texture.GetPrivateData(&KEY, &mut size, None).is_ok() }
}

/// Makes `texture` give `piece` back to `pool` when the last reference to
/// it goes; `small` as for [`counted`].
fn give_back_with(
    texture: &ID3D12Resource,
    piece: Piece,
    pool: &'static Mutex<Pool<ID3D12Heap>>,
    held: &'static AtomicU64,
    small: u64,
) {
    held.fetch_add(piece.len, Ordering::Relaxed);
    let made =
        Box::into_raw(Box::new(Reclaim { methods: &RECLAIM.0, references: AtomicU32::new(1), piece, pool, held, small }));
    // SAFETY: a COM object of ours with one reference, which this takes and
    // drops at the end; the texture takes its own.
    unsafe {
        let object = IUnknown::from_raw(made.cast());
        if texture.SetPrivateDataInterface(&KEY, &object).is_err() {
            // Never given back: the piece stays taken, nothing is reused
            // under a texture that may be alive.
            std::mem::forget(object);
        }
    }
}

/// Experimental texture placement (`VICTORY_POOL_BEST_FIT=1`): the
/// smallest free range that fits, in place of the fullest block's first.
/// Staging and the default policy are unchanged.
fn texture_best_fit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_POOL_BEST_FIT").is_some_and(|v| v == "1"))
}

/// A texture of `desc` (no target, depth or unordered access) in a block;
/// `None`: not pooled, or the host refused.
///
/// # Safety
/// `desc` is a valid description.
pub(crate) unsafe fn texture(
    device: &ID3D12Device,
    desc: *const D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> Option<ID3D12Resource> {
    if !on() {
        return None;
    }
    // Small textures lie at multiples of 4 KiB where the host allows it
    // (the top mip is 64 KiB at most) and [`small_placement`] says so: 64 KiB
    // each otherwise.
    // SAFETY: the caller's description.
    let mut d = unsafe { *desc };
    d.Alignment = if small_placement() { u64::from(D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT) } else { 0 };
    // SAFETY: plain queries.
    let mut info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
    if info.SizeInBytes == u64::MAX || info.Alignment != d.Alignment {
        d.Alignment = 0;
        info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
    }
    if info.SizeInBytes == u64::MAX {
        return None;
    }
    // For the census ([`stats_on`]): what its piece would be at multiples of
    // 4 KiB, if it does not lie at those and the host would have it there.
    let small = if stats_on() && info.Alignment > u64::from(D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT) {
        // SAFETY: the caller's description.
        let mut at = unsafe { *desc };
        at.Alignment = u64::from(D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT);
        // SAFETY: a plain query.
        let there = unsafe { device.GetResourceAllocationInfo(0, &[at]) };
        if there.SizeInBytes != u64::MAX && there.Alignment == at.Alignment { there.SizeInBytes } else { 0 }
    } else {
        0
    };
    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
    let (piece, heap, again) = take_from(&TEXTURES, info.SizeInBytes, info.Alignment.max(1), Ripe::now, fullest(), texture_best_fit(), || {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: BLOCK,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES | crate::host::fresh(1),
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation.
        if let Err(e) = unsafe { device.CreateHeap(&desc, &mut heap) } {
            tracing::warn!(error = %e, "no block of texture memory");
        }
        if let Some(h) = &heap
            && let Err(e) = crate::host::resident_soon(device, h, true)
        {
            tracing::warn!(error = %e, "a block of texture memory is not resident");
            return None;
        }
        heap
    })?;
    let mut made: Option<ID3D12Resource> = None;
    // SAFETY: the piece is no one else's; the heap lives while a piece of
    // it is taken.
    let result = unsafe { device.CreatePlacedResource(&heap, piece.start, &d, state, None, &mut made) };
    match (result, made) {
        (Ok(()), Some(made)) => {
            counted(piece, small, true);
            give_back_with(&made, piece, &TEXTURES, &HELD, small);
            made_now(&made);
            #[cfg(feature = "diagnostics")]
            {
                crate::book::made(&made, || "a texture in a block".into());
                where_the_card_reads(&heap, &made, piece);
                crate::book::block(&heap, &made, piece.start, BLOCK);
                crate::book::note(made.as_raw() as usize, || {
                    let block = lock(&TEXTURES).blocks.get(piece.block).and_then(Option::as_ref).map(|b| (b.made, b.since));
                    let (submission, since) = block.unwrap_or_default();
                    format!(
                        "; piece {:#x}+{:#x} of block {} {:#x} made {:.1} s before it at submission {submission}, used before: {again}, made at submissions {:?}",
                        piece.start,
                        piece.len,
                        piece.block,
                        heap.as_raw() as usize,
                        now().saturating_sub(since) as f64 / 1e3,
                        crate::host::submissions()
                    )
                });
            }
            Some(made)
        }
        _ => {
            lock(&TEXTURES).give(piece, now());
            None
        }
    }
}

/// Whether the host copies of title buffers that shaders write are cut
/// from blocks too (`VICTORY_SHARED_POOL=1`; off: each its own allocation,
/// as before).
///
/// Each was made not resident, made resident and waited for, and its
/// mapping waited for ([`crate::host::resident`], [`crate::host::mapped`]):
/// 0.3 to 2.2 ms a buffer whatever its size. In Leafy's play run of
/// 2026-11-27 the first world frame made 4,648 of them (2,383 MB, half a
/// megabyte each) among 5,800 such waits, 3.3 s of its 4.9; a block is
/// waited for once.
fn shared_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SHARED_POOL").is_some_and(|v| v == "1"))
}

/// Shared buffers cut from blocks so far.
static SHARED_CUT: AtomicU64 = AtomicU64::new(0);

/// A buffer of `size` bytes in system memory that the CPU maps (write-back)
/// and shaders read and write, all zeros, in a block, and its mapping, as
/// [`crate::host::Host::shared_buffer`] makes; `None`: not pooled, or the
/// host refused.
pub(crate) fn shared(device: &ID3D12Device, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    if !shared_on() {
        return None;
    }
    let align = u64::from(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
    let bytes = size.max(1).next_multiple_of(align);
    let properties = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..Default::default()
    };
    let (piece, heap, again) = take_from(&SHARED, bytes, align, Ripe::now, fullest(), false, || {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: BLOCK,
            Properties: properties,
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | crate::host::not_resident(),
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation.
        if let Err(e) = unsafe { device.CreateHeap(&desc, &mut heap) } {
            tracing::warn!(error = %e, "no block of shared buffers");
        }
        if let Some(h) = &heap
            && let Err(e) = crate::host::resident(device, h).and_then(|()| crate::host::mapped(device))
        {
            tracing::warn!(error = %e, "a block of shared buffers is not resident");
            return None;
        }
        heap
    })?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut made: Option<ID3D12Resource> = None;
    let mut data = std::ptr::null_mut();
    // SAFETY: the piece is no one else's; the heap lives while a piece of
    // it is taken.
    let result = unsafe {
        device.CreatePlacedResource(&heap, piece.start, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut made).and_then(|()| {
            made.as_ref().ok_or_else(windows::core::Error::empty)?.Map(0, None, Some(&mut data))
        })
    };
    match (result, made) {
        (Ok(()), Some(made)) => {
            give_back_with(&made, piece, &SHARED, &SHARED_HELD, 0);
            if again {
                // A piece used before holds what its last buffer left.
                // SAFETY: the mapping covers the buffer, which is ours alone.
                unsafe { std::ptr::write_bytes(data.cast::<u8>(), 0, size as usize) };
            }
            let n = SHARED_CUT.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() && n >= 64 {
                tracing::info!(
                    "shared buffers cut from blocks: {n}, {} blocks, {} MiB held",
                    lock(&SHARED).blocks(),
                    SHARED_HELD.load(Ordering::Relaxed) >> 20
                );
            }
            #[cfg(feature = "diagnostics")]
            crate::book::made(&made, || "a shared buffer in a block".into());
            Some((made, data.cast()))
        }
        _ => {
            lock(&SHARED).give(piece, now());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_cut_aligned_and_join_again() {
        let mut r = Ranges::new(1024);
        assert_eq!(r.take(100, 64), Some(0));
        assert_eq!(r.take(100, 64), Some(128));
        assert_eq!(r.take(28, 4), Some(100));
        assert_eq!(r.take(2000, 4), None);
        r.give(128, 100);
        r.give(0, 100);
        assert!(!r.whole(1024));
        r.give(100, 28);
        assert!(r.whole(1024));
    }

    #[test]
    fn the_census_counts_what_is_free_resting_and_how_full_blocks_are() {
        let mut pool: Pool<u32> = Pool::new(2);
        let (kept, ..) = pool.take(BLOCK / 2, 64 << 10, 0, || Some(1)).unwrap();
        let (given, ..) = pool.take(1 << 20, 64 << 10, 0, || Some(1)).unwrap();
        assert_eq!((kept.start, given.start), (0, BLOCK / 2));
        let c = pool.census();
        assert_eq!((c.blocks, c.empty, c.ranges, c.resting), (1, 0, 1, 0));
        assert_eq!(c.free, BLOCK / 2 - (1 << 20));
        assert_eq!(c.full, [0, 0, 1, 0]);
        // A piece given back rests: neither held nor free.
        pool.give(given, 0);
        let c = pool.census();
        assert_eq!((c.resting, c.free, c.full), (1 << 20, BLOCK / 2 - (1 << 20), [0, 0, 1, 0]));
        pool.rested(REST);
        let c = pool.census();
        assert_eq!((c.resting, c.free, c.ranges, c.largest, c.wide), (0, BLOCK / 2, 1, BLOCK / 2, BLOCK / 2));
        assert_eq!(c.full, [0, 0, 1, 0]);
    }

    #[test]
    fn textures_are_counted_by_their_piece() {
        assert_eq!([0x1_0000, 0x1_0001, 0x4_0000, 0x10_0000, 0x40_0000, 0x40_0001].map(class), [0, 1, 1, 2, 3, 4]);
    }

    #[test]
    fn empty_blocks_stay_while_the_card_has_room() {
        assert_eq!(spares(Some([100, 84, 0, 0]), true), SPARE_KEPT);
        assert_eq!(spares(Some([100, 85, 0, 0]), true), SPARE);
        assert_eq!(spares(Some([100, 10, 0, 0]), false), SPARE);
        assert_eq!(spares(None, true), SPARE);

        let budgets = [
            (1u64, 8, 12),
            (8, 8, 12),
            (12, 8, 12),
            (16, 8, 12),
            (24, 12, 12),
            (256, SPARE_KEPT, SPARE_KEPT),
        ];
        for (gib, without_ripe, with_ripe) in budgets {
            let budget = gib << 30;
            for (percent, legacy) in [(84, SPARE_KEPT), (85, SPARE), (86, SPARE)] {
                let memory = Some([budget, (budget * percent).div_ceil(100), 0, 0]);
                let limit = spares(memory, true);
                for (ripe_ahead, expected) in [
                    (None, legacy),
                    (Some(0), legacy.min(without_ripe)),
                    (Some(DEFAULT_RIPE_BLOCKS), legacy.min(with_ripe)),
                    (Some(16), legacy.min(without_ripe.max(16))),
                ] {
                    assert_eq!(
                        texture_spares(memory, limit, ripe_ahead),
                        expected,
                        "{gib} GiB at {percent}%, scaled ripe floor={ripe_ahead:?}"
                    );
                }
            }
        }

        let memory = Some([8 << 30, 1 << 30, 0, 0]);
        assert_eq!(
            texture_spares(memory, spares(memory, false), Some(DEFAULT_RIPE_BLOCKS)),
            SPARE
        );
        assert_eq!(
            texture_spares(None, spares(None, true), Some(DEFAULT_RIPE_BLOCKS)),
            SPARE
        );
        assert_eq!(
            texture_spares(memory, SPARE_KEPT, Some(SPARE_KEPT + 1)),
            SPARE_KEPT
        );
    }

    /// `VICTORY_SPARES_BY_HEADROOM=1`: at most half of what is left below
    /// 85 % of the budget, in blocks, the ends kept as they are.
    #[test]
    fn spares_follow_the_cards_headroom_when_asked() {
        const GIB: u64 = 1 << 30;
        let budget_of = |gib: u64| gib * GIB;
        // A 24 GB card 10 GB in keeps all 64 blocks.
        assert_eq!(spares_of(Some([budget_of(24), budget_of(10), 0, 0]), true, true), SPARE_KEPT);
        // A 16 GB card 12 GB in: half of the ~1.6 GB left below 85 %, in
        // 64 MiB blocks, is 12 blocks, not 4 GiB of them.
        assert_eq!(spares_of(Some([budget_of(16), budget_of(12), 0, 0]), true, true), 12);
        // Past 85 %, or told not to keep, or nothing reported: as before.
        assert_eq!(spares_of(Some([budget_of(16), budget_of(15), 0, 0]), true, true), SPARE);
        assert_eq!(spares_of(Some([budget_of(16), budget_of(12), 0, 0]), false, true), SPARE);
        assert_eq!(spares_of(None, true, true), SPARE);
        // Off: the binary choice.
        assert_eq!(spares_of(Some([budget_of(16), budget_of(12), 0, 0]), true, false), SPARE_KEPT);
        assert_eq!(spares_of(Some([budget_of(16), budget_of(15), 0, 0]), true, false), SPARE);
    }

    #[test]
    fn a_failed_block_is_not_retried_for_each_small_upload() {
        let mut pool: Pool<u32> = Pool::new(2);
        let calls = std::cell::Cell::new(0);
        let refuse = || {
            calls.set(calls.get() + 1);
            None
        };
        for size in [704008, 179720, 179720, 24080, 48648, 524288, 89616, 351760] {
            assert!(pool.take(size, 512, 12, refuse).is_none());
        }
        assert_eq!(calls.get(), 1);
        assert!(pool.take(24080, 512, 1012, || Some(1)).is_some());
    }

    #[test]
    fn budget_tick_reclaims_idle_blocks_without_reusing_live_or_resting_pieces() {
        let mut pool = Pool::new(STAGING_KEPT);
        let (held, _, _) = pool.take(BLOCK / 2, 1, 0, || Some(1)).unwrap();
        let (resting, _, _) = pool.take(BLOCK / 2, 1, 0, || Some(2)).unwrap();
        pool.give(resting, 0);
        for i in 0..12 {
            pool.add(i, 0);
        }
        pool.reclaim(0, SPARE_KEPT, true);
        assert_eq!(pool.blocks(), STAGING_KEPT + 1);
        pool.reclaim(REST - 1, SPARE, true);
        assert_eq!(pool.blocks(), SPARE + 1);
        assert_eq!(pool.resting.len(), 1);
        pool.reclaim(REST, 0, true);
        assert_eq!(pool.blocks(), 1);
        assert!(pool.resting.is_empty());
        pool.give(held, REST);
        pool.reclaim(2 * REST, 0, true);
        assert_eq!(pool.blocks(), 0);
    }

    #[test]
    fn pieces_never_overlap_and_rest_before_they_are_used_again() {
        let mut pool: Pool<u32> = Pool::new(SPARE);
        let made = std::cell::Cell::new(0);
        let make = || {
            made.set(made.get() + 1);
            Some(made.get())
        };
        let mut taken: Vec<Piece> = Vec::new();
        for i in 0..500u64 {
            let (p, _, again) = pool.take(65536 * (1 + i % 7), 65536, 0, make).unwrap();
            assert!(!again);
            assert!(taken.iter().all(|q| q.block != p.block || p.start + p.len <= q.start || q.start + q.len <= p.start));
            taken.push(p);
        }
        let blocks = pool.blocks();
        assert!(blocks >= 2);
        // Given back at 10: not used again before 10 + REST.
        let first = taken[0];
        pool.give(first, 10);
        let (p, _, _) = pool.take(first.len, 65536, 10 + REST - 1, make).unwrap();
        assert_ne!((p.block, p.start), (first.block, first.start));
        let (q, _, again) = pool.take(first.len, 65536, 10 + REST, make).unwrap();
        assert_eq!((q.block, q.start, again), (first.block, first.start, true));
        // Everything back: the empty blocks past the spare ones go.
        for p in taken.drain(1..).chain([p, q]) {
            pool.give(p, 20);
        }
        pool.rested(20 + REST);
        assert_eq!(pool.blocks(), blocks.min(SPARE));
        // Too big for a block, and a block that cannot be made.
        assert!(pool.take(BLOCK, 65536, 30, make).is_none());
        let mut none: Pool<u32> = Pool::new(SPARE);
        assert!(none.take(65536, 65536, 0, || None).is_none());
    }

    /// A block is made with the pool's lock let go (making one waits for
    /// residency, up to a second near the card's budget), it is marked with
    /// the submissions begun after its making, and a block made ahead is
    /// counted off again whether it was made or not.
    #[test]
    fn blocks_are_made_with_the_pool_let_go() {
        static POOL: Mutex<Pool<u64>> = Mutex::new(Pool::new(SPARE_KEPT));
        let made = std::cell::Cell::new(0u64);
        let make = || {
            assert!(POOL.try_lock().is_ok(), "a block was made with the pool's lock held");
            made.set(made.get() + 1);
            Some(made.get())
        };
        // Ten more submissions begun for every block made: a mark taken
        // before a making would be ten lower.
        let ripe = || Ripe { started: made.get() * 10, finished: 0, ahead: 2 };
        let (piece, memory, again) = take_from(&POOL, BLOCK / 2, 65536, ripe, false, false, make).unwrap();
        assert_eq!((piece.start, memory, again), (0, 1, false));
        assert_eq!(made.get(), 2, "one more made ahead");
        {
            let p = lock(&POOL);
            let marks: Vec<u64> = p.blocks.iter().flatten().map(|b| b.made).collect();
            assert_eq!(marks, [10, 20]);
            assert_eq!(p.making_ahead, 0);
        }
        // Block 2 is empty, so one more is made ahead; then that one fails.
        assert_eq!(take_from(&POOL, BLOCK / 2, 65536, ripe, false, false, make).unwrap().1, 1);
        assert_eq!(made.get(), 3);
        let refuse = || {
            assert!(POOL.try_lock().is_ok(), "a block was made with the pool's lock held");
            None
        };
        let full = || Ripe { started: 100, finished: 100, ahead: 4 };
        assert!(take_from(&POOL, BLOCK / 2, 65536, full, false, false, refuse).is_some(), "cut from a block there is");
        assert_eq!(lock(&POOL).making_ahead, 0, "a block not made is counted off");
    }

    #[test]
    fn pieces_come_from_blocks_older_than_the_lists_that_run() {
        let mut pool: Pool<u64> = Pool::new(SPARE_KEPT);
        let made = std::cell::Cell::new(0);
        let make = || {
            made.set(made.get() + 1);
            Some(made.get())
        };
        let at = |started, finished| Ripe { started, finished, ahead: 2 };
        let take = |pool: &mut Pool<u64>, ripe| pool.take_ripe(BLOCK / 2, 65536, 0, ripe, false, make).unwrap().1;
        // Nothing runs: at once, and two are made ahead, one a time.
        assert_eq!(take(&mut pool, at(0, 0)), 1);
        assert_eq!(made.get(), 2);
        assert_eq!(take(&mut pool, at(0, 0)), 1);
        assert_eq!(made.get(), 3);
        // 4 is made while submission 5 runs.
        assert_eq!(take(&mut pool, at(5, 4)), 2);
        assert_eq!(made.get(), 4);
        assert_eq!(take(&mut pool, at(5, 4)), 2);
        assert_eq!(take(&mut pool, at(6, 4)), 3);
        assert_eq!(take(&mut pool, at(6, 4)), 3);
        // None is ripe: the oldest; it is ripe once 5 has finished.
        assert_eq!(take(&mut pool, at(7, 4)), 4);
        assert_eq!(take(&mut pool, at(7, 5)), 4);
        // A whole block's worth too.
        assert_eq!(pool.take_ripe(BLOCK, 65536, 0, at(8, 7), false, make).unwrap().1, 5);
        // Off: the first with room, whatever its age, none ahead.
        let n = made.get();
        assert_eq!(pool.take(BLOCK / 2, 65536, 0, make).unwrap().1, 6);
        assert_eq!(made.get(), n);
        assert!(pool.take(BLOCK, 65536, 0, make).is_none());
    }

    /// Milliseconds calls took, summed up: mean, median, 99th percentile,
    /// longest.
    fn spread(mut ms: Vec<f64>) -> String {
        ms.sort_by(f64::total_cmp);
        let at = |q: f64| ms[((ms.len() - 1) as f64 * q).round() as usize];
        format!(
            "{} calls: mean {:.3} ms, median {:.3}, 99th {:.3}, longest {:.3}",
            ms.len(),
            ms.iter().sum::<f64>() / ms.len() as f64,
            at(0.5),
            at(0.99),
            ms[ms.len() - 1]
        )
    }

    /// A texture of `side` x `side` RGBA8, one level: 4 x side^2 bytes.
    fn square(side: u64) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: side,
            Height: side as u32,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        }
    }

    /// Bench (`--ignored --nocapture`, with `VICTORY_RIPE_BLOCKS=0`): a
    /// thread places textures of 256 KiB, which blocks with pieces have
    /// room for, while another places textures of 24 MiB, which need a new
    /// block every second one.
    #[test]
    #[ignore]
    fn bench_textures_placed_while_another_thread_makes_blocks() {
        let Some(host) = crate::host::host() else { return };
        let (small, big) = (square(256), square(2560));
        let stop = std::sync::atomic::AtomicBool::new(false);
        // SAFETY: valid descriptions, here and below.
        let mut keep: Vec<ID3D12Resource> = (0..4).filter_map(|_| unsafe { texture(&host.device, &small, D3D12_RESOURCE_STATE_COMMON) }).collect();
        let mut ms = Vec::new();
        std::thread::scope(|s| {
            s.spawn(|| {
                let mut held = std::collections::VecDeque::new();
                while !stop.load(Ordering::Relaxed) {
                    held.extend(unsafe { texture(&host.device, &big, D3D12_RESOURCE_STATE_COMMON) });
                    if held.len() > 16 {
                        held.pop_front();
                    }
                }
            });
            std::thread::sleep(std::time::Duration::from_millis(100));
            for _ in 0..1000 {
                let began = std::time::Instant::now();
                keep.extend(unsafe { texture(&host.device, &small, D3D12_RESOURCE_STATE_COMMON) });
                ms.push(began.elapsed().as_secs_f64() * 1e3);
            }
            stop.store(true, Ordering::Relaxed);
        });
        eprintln!(
            "textures of 256 KiB placed while another thread makes blocks: {}",
            spread(ms)
        );
    }

    /// Bench (`--ignored --nocapture`): a thread stages pieces of 64 KiB
    /// while another stages pieces of 8 MiB and keeps them, so that new
    /// staging blocks are made.
    #[test]
    #[ignore]
    fn bench_staging_while_another_thread_makes_blocks() {
        let Some(host) = crate::host::host() else { return };
        let stop = std::sync::atomic::AtomicBool::new(false);
        let mut ms = Vec::new();
        let mut keep = Vec::new();
        keep.extend(staging(host, 64 << 10));
        std::thread::scope(|s| {
            s.spawn(|| {
                let mut held = std::collections::VecDeque::new();
                while !stop.load(Ordering::Relaxed) {
                    held.extend(staging(host, 8 << 20));
                    if held.len() > 64 {
                        held.pop_front();
                    }
                }
            });
            std::thread::sleep(std::time::Duration::from_millis(100));
            for _ in 0..1000 {
                let began = std::time::Instant::now();
                keep.extend(staging(host, 64 << 10));
                ms.push(began.elapsed().as_secs_f64() * 1e3);
            }
            stop.store(true, Ordering::Relaxed);
        });
        eprintln!(
            "staging of 64 KiB while another thread makes staging blocks: {}",
            spread(ms)
        );
    }

    /// Threads that place textures at once, some of them making new blocks
    /// (outside the pool's lock, [`take_from`]), get every texture,
    /// each in a piece of its own.
    #[test]
    fn threads_that_place_textures_at_once_make_blocks_and_keep_their_pieces_apart() {
        let Some(host) = crate::host::host() else { return };
        // 4 MiB each: 16 to a block, so 8 threads of 24 need new blocks
        // while others place theirs.
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1024,
            Height: 1024,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let made: Vec<ID3D12Resource> = std::thread::scope(|s| {
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    s.spawn(|| {
                        (0..24)
                            // SAFETY: a valid description.
                            .filter_map(|_| unsafe { texture(&host.device, &desc, D3D12_RESOURCE_STATE_COMMON) })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            threads.into_iter().flat_map(|t| t.join().unwrap()).collect()
        });
        assert_eq!(made.len(), 8 * 24, "every texture was placed");
        // Each texture's piece is its own: no two of a block overlap.
        let pool = lock(&TEXTURES);
        let mut taken: Vec<(usize, u64, u64)> = Vec::new();
        for (i, block) in pool.blocks.iter().enumerate() {
            let Some(block) = block else { continue };
            let mut at = 0;
            for (&start, &len) in &block.free.0 {
                if start > at {
                    taken.push((i, at, start));
                }
                at = start + len;
            }
            if at < BLOCK {
                taken.push((i, at, BLOCK));
            }
        }
        let held: u64 = taken.iter().map(|&(_, a, b)| b - a).sum();
        drop(pool);
        assert!(held >= 8 * 24 * (4 << 20), "the pieces held cover every texture");
        drop(made);
        // SAFETY: plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
    }

    /// Threads that stage at once, some making new staging blocks (outside
    /// the pool's lock), each get pieces of their own that begin as zeros.
    #[test]
    fn threads_that_stage_at_once_get_zeroed_pieces_of_their_own() {
        let Some(host) = crate::host::host() else { return };
        const BYTES: usize = 8 << 20;
        // Bytes looked at: upload memory is write-combined, and the CPU
        // reads it slowly. One in every 4 KiB, and the last.
        let looked = |data: *mut u8| (0..BYTES).step_by(4096).chain([BYTES - 1]).map(move |i| unsafe { *data.add(i) });
        std::thread::scope(|s| {
            for t in 0..8u8 {
                s.spawn(move || {
                    // 8 of 8 MiB a thread: 64 MiB each, so new blocks are made.
                    let pieces: Vec<_> = (0..8).filter_map(|_| staging(host, BYTES as u64)).collect();
                    assert_eq!(pieces.len(), 8);
                    for (_, _, _, data) in &pieces {
                        // SAFETY: the piece's mapping, ours alone.
                        unsafe {
                            assert!(looked(*data).all(|b| b == 0), "a piece is not zeros");
                            std::ptr::write_bytes(*data, t + 1, BYTES);
                        }
                    }
                    for (_, _, _, data) in &pieces {
                        assert!(looked(*data).all(|b| b == t + 1), "another thread wrote this piece");
                    }
                });
            }
        });
    }

    /// Fullest first: a new piece goes to the block with the most taken, so
    /// the one left nearly empty drains and goes.
    #[test]
    fn best_fit_prefers_an_exact_ripe_range() {
        let mut pool = Pool::new(2);
        pool.add(1u32, 0);
        pool.add(2u32, 0);
        pool.blocks[0].as_mut().unwrap().free = Ranges(BTreeMap::from([(0, 8192)]));
        pool.blocks[1].as_mut().unwrap().free = Ranges(BTreeMap::from([(4096, 4096)]));
        assert_eq!(pool.best_fit_ripe(4096, 4096, Ripe::ALL), Some((1, 4096)));
        assert_eq!(pool.blocks[0].as_ref().unwrap().free.0.get(&0), Some(&8192));
        assert!(pool.blocks[1].as_ref().unwrap().free.0.is_empty());
    }

    #[test]
    fn best_fit_splits_aligned_ranges_without_losing_padding() {
        let mut pool = Pool::new(2);
        pool.add(1u32, 0);
        let block = pool.blocks[0].as_mut().unwrap();
        block.free = Ranges(BTreeMap::from([(4097, 8192), (32768, 16384)]));
        assert_eq!(pool.best_fit_ripe(4096, 4096, Ripe::ALL), Some((0, 8192)));
        let free = &pool.blocks[0].as_ref().unwrap().free.0;
        assert_eq!(*free, BTreeMap::from([(4097, 4095), (12288, 1), (32768, 16384)]));
    }

    #[test]
    fn best_fit_does_not_prefer_an_unripe_exact_range() {
        let mut pool = Pool::new(2);
        pool.add(1u32, 0);
        pool.add(2u32, 9);
        pool.blocks[0].as_mut().unwrap().free = Ranges(BTreeMap::from([(0, 8192)]));
        pool.blocks[1].as_mut().unwrap().free = Ranges(BTreeMap::from([(0, 4096)]));
        let ripe = Ripe { started: 10, finished: 5, ahead: 1 };
        assert_eq!(pool.best_fit_ripe(4096, 4096, ripe), Some((0, 0)));
        assert_eq!(pool.blocks[1].as_ref().unwrap().free.0.get(&0), Some(&4096));
    }

    #[test]
    fn best_fit_policy_preserves_quarantine() {
        let mut pool = Pool::new(2);
        let make = || Some(1u32);
        let p = pool.take_ripe_with_policy(4096, 4096, 0, Ripe::ALL, false, true, make).unwrap().0;
        pool.give(p, 10);
        let q = pool.take_ripe_with_policy(4096, 4096, REST + 9, Ripe::ALL, false, true, make).unwrap().0;
        assert_ne!((p.block, p.start), (q.block, q.start));
        let r = pool.take_ripe_with_policy(4096, 4096, REST + 10, Ripe::ALL, false, true, make).unwrap().0;
        assert_eq!((p.block, p.start), (r.block, r.start));
    }

    #[test]
    fn pieces_go_to_the_fullest_block_with_room() {
        let mut pool: Pool<u32> = Pool::new(0);
        let made = std::cell::Cell::new(0);
        let make = || {
            made.set(made.get() + 1);
            Some(made.get())
        };
        let quarter = BLOCK / 4;
        let take = |pool: &mut Pool<u32>| pool.take_ripe(quarter, 65536, 0, Ripe::ALL, true, make).unwrap().0;
        let taken = |pool: &Pool<u32>, i: usize| pool.blocks[i].as_ref().map(|b| b.taken);
        // Two blocks full, then three of the first's pieces and one of the
        // second's back.
        let a: Vec<Piece> = (0..8).map(|_| take(&mut pool)).collect();
        assert_eq!((a[0].block, a[3].block, a[4].block, a[7].block), (0, 0, 1, 1));
        for p in [a[0], a[1], a[2], a[4]] {
            pool.give(p, 0);
        }
        pool.rested(REST);
        assert_eq!((taken(&pool, 0), taken(&pool, 1)), (Some(quarter), Some(3 * quarter)));
        // The first with room is block 0; the fullest, block 1.
        assert_eq!(take(&mut pool).block, 1);
        assert_eq!(taken(&pool, 1), Some(BLOCK));
        // Block 0's last piece back: it goes.
        pool.give(a[3], REST);
        pool.rested(2 * REST);
        assert_eq!((pool.blocks(), pool.let_go), (1, 1));
        // Next, a new block where block 0 was.
        assert_eq!(take(&mut pool).block, 0);
        assert_eq!(made.get(), 3);
    }

    /// A pooled texture's piece comes back when its last reference goes, not
    /// before; its staging piece begins as zeros.
    #[test]
    fn room_leaves_out_pieces_taken_and_resting() {
        let mut pool: Pool<u32> = Pool::new(SPARE);
        let (piece, _, _) = pool.take(BLOCK / 2, 1, 0, || Some(1)).unwrap();
        // Taken before CreatePlacedResource is asked to place a texture there.
        assert_eq!(pool.reusable_bytes(0), BLOCK / 2);
        pool.give(piece, 0);
        assert_eq!(pool.reusable_bytes(REST - 1), BLOCK / 2);
        // The check alone makes the rested piece free again.
        assert_eq!(pool.reusable_bytes(REST), BLOCK);
        // And lets the block go when none are kept.
        let mut none: Pool<u32> = Pool::new(0);
        let (piece, _, _) = none.take(BLOCK / 2, 1, 0, || Some(1)).unwrap();
        none.give(piece, 0);
        assert_eq!(none.reusable_bytes(REST), 0);
        assert_eq!(none.blocks(), 0);
    }

    #[test]
    fn room_counts_alignment_gaps_and_blocks_made_again() {
        let mut pool: Pool<u32> = Pool::new(0);
        let (first, _, _) = pool.take(100, 64, 0, || Some(1)).unwrap();
        let (second, _, _) = pool.take(100, 64, 0, || Some(1)).unwrap();
        assert_eq!((first.start, second.start), (0, 128));
        assert_eq!(pool.reusable_bytes(0), BLOCK - 200);
        // The gap left for alignment still takes a smaller piece.
        let (gap, _, _) = pool.take(28, 4, 0, || None).unwrap();
        assert_eq!(gap.start, 100);
        assert_eq!(pool.reusable_bytes(0), BLOCK - 228);
        for piece in [first, second, gap] {
            pool.give(piece, 0);
        }
        assert_eq!(pool.reusable_bytes(REST), 0);
        let (next, _, _) = pool.take(512, 512, REST, || Some(2)).unwrap();
        assert_eq!(next.block, first.block);
        assert_eq!(pool.reusable_bytes(REST), BLOCK - 512);
    }

    #[test]
    fn room_is_the_free_ranges_through_taking_and_giving_back() {
        let mut pool: Pool<u32> = Pool::new(SPARE);
        let mut held = VecDeque::new();
        for tick in 0..400u64 {
            let now = tick * 100;
            let ripe = Ripe { started: tick, finished: tick.saturating_sub(2), ahead: 2 };
            let (piece, _, _) = pool.take_ripe((tick % 7 + 1) * 65536, 65536, now, ripe, false, || Some(1)).unwrap();
            held.push_back(piece);
            if held.len() > 7 {
                pool.give(held.pop_front().unwrap(), now);
            }
            let room = pool.reusable_bytes(now);
            let free: u64 = pool.blocks.iter().flatten().flat_map(|b| b.free.0.values()).sum();
            assert_eq!(room, free, "tick {tick}");
        }
        for piece in held {
            pool.give(piece, 40000);
        }
        assert_eq!(pool.reusable_bytes(40000 + REST), pool.blocks() as u64 * BLOCK);
        assert!(pool.blocks.iter().flatten().all(|b| b.taken == 0));
    }

    #[test]
    fn a_texture_gives_its_piece_back_when_it_goes() {
        let Some(host) = crate::host::host() else { return };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        // SAFETY: a valid description.
        let Some(t) = (unsafe { texture(&host.device, &desc, D3D12_RESOURCE_STATE_COMMON) }) else { return };
        assert_eq!(crate::host::references(&t), 1, "what the texture holds is no reference to it");
        let resting = || lock(&TEXTURES).resting.len();
        let before = resting();
        let other = t.clone();
        drop(t);
        assert_eq!(resting(), before);
        drop(other);
        assert_eq!(resting(), before + 1);
        // SAFETY: the piece's mapping, 4096 bytes of it ours.
        unsafe {
            let (piece, _, at, data) = staging(host, 4096).unwrap();
            assert!(at.is_multiple_of(512));
            std::ptr::write_bytes(data, 0xAB, 4096);
            drop(piece);
            lock(&STAGING).rested(u64::MAX);
            let (_piece, _, _, data) = staging(host, 4096).unwrap();
            assert!((0..4096).all(|i| *data.add(i) == 0));
            // SAFETY: plain query.
            assert!(host.device.GetDeviceRemovedReason().is_ok());
        }
    }

    /// Empty blocks past those kept go when the card runs short of room,
    /// though no texture is made after (AvatarManz#327).
    #[test]
    fn fewer_spares_let_empty_blocks_go_without_a_texture_made() {
        let mut pool: Pool<u32> = Pool::new(SPARE_KEPT);
        let mut pieces = Vec::new();
        for n in 1u32..=5 {
            let (piece, _, _) = pool.take(BLOCK / 2, 512, 0, || Some(n)).unwrap();
            pieces.push(piece);
            // The other half taken too, so the next piece needs a block of its own.
            let (other, _, _) = pool.take(BLOCK / 2, 512, 0, || unreachable!()).unwrap();
            pool.give(other, 0);
        }
        pool.rested_under(REST + 1, SPARE_KEPT);
        assert_eq!(pool.blocks(), 5);
        for piece in pieces {
            pool.give(piece, 0);
        }
        pool.rested_under(REST + 1, SPARE);
        assert_eq!(pool.blocks(), SPARE);
    }
}

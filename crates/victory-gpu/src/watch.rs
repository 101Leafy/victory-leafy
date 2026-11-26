//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! The title shows one physical page at several addresses, and writes it
//! through one while the card reads it at another. So a block is clean, and
//! known by its bit, only while every view of its page is read-only
//! ([`cover`]): a write through any of them faults, opens them all and takes
//! clean and bit from all of them ([`open`]). A view mapped anew of a page
//! that is guarded elsewhere has the others looked at again
//! ([`views_mapped`]). What we write ourselves is told at one address and
//! counts at all ([`writing`], [`landed_beside`], [`wrote`]). Not so for a
//! page shown in more than [`MANY_VIEWS`] places (what the title shows where
//! it has nothing). `VICTORY_VIEW_GUARD=0`: each address for itself, as
//! before.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Bound, RangeBounds};
use std::sync::{Condvar, Mutex, RwLock};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// The parts of a page.
const PARTS: usize = PAGE / KERNEL_PAGE;
/// All of them.
const WHOLE: u16 = u16::MAX;

/// Whether the parts of a page that the kernel names are kept, and a refresh
/// copies those (`VICTORY_PAGE_PARTS=0`: the 64 KiB page, as before).
///
/// The kernel says which 4 KiB pages were written, and a copy took the 64 KiB
/// around each: 29 MB a frame written so were most of 114 MB copied, and as
/// much sent up and hashed again on the card (session play-1125-0203, run
/// cpu-2). What is checked is as before, a page.
fn parts_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PAGE_PARTS").is_none_or(|v| v != "0"))
}

impl Shard {
    const fn new() -> Shard {
        Shard { written: BTreeMap::new(), parts: std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::new()) }
    }

    /// Page `page` was found written by asking `now`: the parts in `parts`
    /// (a bit each). The asking that found it written before (0: none).
    ///
    /// A page's notes only go up: with the journal held alone `now` is the
    /// newest number there is, as it always was; noted under the shared lock
    /// (renderer port #7) another asking with a higher number may have
    /// noted it first.
    fn note(&mut self, page: usize, parts: u16, now: u64) -> u64 {
        let before = self.written.get(&page).copied().unwrap_or(0);
        self.written.insert(page, before.max(now));
        if parts == WHOLE || !parts_on() {
            if now >= before {
                self.parts.remove(&page);
            } else if let Some(all) = self.parts.get_mut(&page) {
                all.iter_mut().for_each(|at| *at = (*at).max(now));
            }
            return before;
        }
        let all = self.parts.entry(page).or_insert([before; PARTS]);
        for (k, at) in all.iter_mut().enumerate() {
            if parts >> k & 1 != 0 {
                *at = (*at).max(now);
            }
        }
        before
    }

    /// The parts of `page` written by an asking after `taken`.
    fn parts_since(&self, page: usize, taken: u64) -> u16 {
        match self.parts.get(&page) {
            Some(all) => all.iter().enumerate().filter(|(_, at)| **at > taken).fold(0, |m, (k, _)| m | 1 << k),
            None => WHOLE,
        }
    }
}

/// The pages of a part of the journal ([`Notes`]): pieces of
/// `1 << SHARD_SHIFT` pages (64 MiB), every [`SHARDS`]th piece in one part.
const SHARD_SHIFT: u32 = 10;
/// The parts: a bit each in a `u64` ([`shards_of`]).
const SHARDS: usize = 64;

fn shard_of(page: usize) -> usize {
    (page >> SHARD_SHIFT) % SHARDS
}

/// The parts that hold pages `[first, last)`, a bit each.
fn shards_of(first: usize, last: usize) -> u64 {
    if first >= last {
        return 0;
    }
    let (a, b) = (first >> SHARD_SHIFT, (last - 1) >> SHARD_SHIFT);
    if b - a + 1 >= SHARDS {
        return u64::MAX;
    }
    (a..=b).fold(0, |m, piece| m | 1 << (piece % SHARDS))
}

/// The parts that hold `pages`, a bit each.
fn shards_of_pages(pages: impl IntoIterator<Item = usize>) -> u64 {
    pages.into_iter().fold(0, |m, page| m | 1 << shard_of(page))
}

/// The pages' notes, in parts of their own, each under a lock of its own
/// (renderer port #7). Only reached through [`JOURNAL`]: held alone, its
/// holder changes them with no part's lock; held shared, a reader reads a
/// part at a time, and who notes or needs a number that counts holds the
/// parts it touches ([`Notes::hold`]).
struct Notes([RwLock<Shard>; SHARDS]);

/// A part of [`Notes`].
struct Shard {
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// The same for each of a page's sixteen parts ([`KERNEL_PAGE`]), where
    /// the parts are known ([`parts_on`]); a page that is not here was
    /// written whole, as far as anyone knows.
    parts: std::collections::HashMap<usize, [u64; PARTS], std::hash::BuildHasherDefault<PageHash>>,
}

impl Notes {
    const fn new() -> Notes {
        Notes([const { RwLock::new(Shard::new()) }; SHARDS])
    }

    /// The part of `page`, with the journal held alone.
    fn shard_mut(&mut self, page: usize) -> &mut Shard {
        self.0[shard_of(page)].get_mut().unwrap_or_else(|e| e.into_inner())
    }

    /// A part to read, the journal held shared or alone.
    fn read(&self, shard: usize) -> std::sync::RwLockReadGuard<'_, Shard> {
        if let Ok(s) = self.0[shard].try_read() {
            return s;
        }
        let began = std::time::Instant::now();
        let s = self.0[shard].read().unwrap_or_else(|e| e.into_inner());
        port7_waited(P7_PART_WAITS, began);
        s
    }

    /// See [`Shard::note`]; the journal held alone.
    fn note(&mut self, page: usize, parts: u16, now: u64) -> u64 {
        self.shard_mut(page).note(page, parts, now)
    }

    /// See [`Shard::parts_since`].
    fn parts_since(&self, page: usize, taken: u64) -> u16 {
        self.read(shard_of(page)).parts_since(page, taken)
    }

    /// The notes of pages `[first, last)` in page order, `each` of them
    /// until it says no: whether it never did. A part at a time.
    fn each(&self, first: usize, last: usize, mut each: impl FnMut(usize, u64) -> bool) -> bool {
        match pieces(first, last) {
            Some(pieces) => pieces.into_iter().all(|(from, to)| {
                let s = self.read(shard_of(from));
                s.written.range(from..to).all(|(&p, &w)| each(p, w))
            }),
            None => {
                let mut all: Vec<(usize, u64)> = Vec::new();
                for shard in 0..SHARDS {
                    all.extend(self.read(shard).written.range(first..last).map(|(&p, &w)| (p, w)));
                }
                all.sort_unstable();
                all.into_iter().all(|(p, w)| each(p, w))
            }
        }
    }

    /// The notes of pages `[first, last)` in page order. A part at a time.
    fn range(&self, first: usize, last: usize) -> Vec<(usize, u64)> {
        let mut out = Vec::new();
        self.each(first, last, |p, w| {
            out.push((p, w));
            true
        });
        out
    }

    /// The parts `shards` (a bit each), held until dropped: for writing
    /// (`write`) or reading. Taken in order: whoever waits for a part holds
    /// only lower ones, so no two holders wait for each other.
    fn hold(&self, shards: u64, write: bool) -> Held<'_> {
        let mut held = Held { shards, read: Vec::new(), write: Vec::new() };
        let mut m = shards;
        while m != 0 {
            let shard = m.trailing_zeros() as usize;
            m &= m - 1;
            let lock = &self.0[shard];
            if write {
                let guard = match lock.try_write() {
                    Ok(g) => g,
                    Err(_) => {
                        let began = std::time::Instant::now();
                        let g = lock.write().unwrap_or_else(|e| e.into_inner());
                        port7_waited(P7_PART_WAITS, began);
                        g
                    }
                };
                held.write.push(guard);
            } else {
                held.read.push(self.read(shard));
            }
        }
        held
    }
}

/// The pieces of pages `[first, last)` in page order, each in one part;
/// `None` for a range of more pieces than is worth walking (then each part
/// is looked at once).
fn pieces(first: usize, last: usize) -> Option<Vec<(usize, usize)>> {
    if first >= last {
        return Some(Vec::new());
    }
    let (a, b) = (first >> SHARD_SHIFT, (last - 1) >> SHARD_SHIFT);
    if b - a >= 4 * SHARDS {
        return None;
    }
    Some((a..=b).map(|piece| (first.max(piece << SHARD_SHIFT), last.min((piece + 1) << SHARD_SHIFT))).collect())
}

/// Parts of [`Notes`] held ([`Notes::hold`]).
struct Held<'a> {
    shards: u64,
    read: Vec<std::sync::RwLockReadGuard<'a, Shard>>,
    write: Vec<std::sync::RwLockWriteGuard<'a, Shard>>,
}

impl Held<'_> {
    /// The place among those held of the part of `page`.
    fn place(&self, page: usize) -> usize {
        let shard = shard_of(page);
        assert!(self.shards >> shard & 1 != 0, "page {page:#x}: its part is not held");
        (self.shards & ((1u64 << shard) - 1)).count_ones() as usize
    }

    fn shard(&self, page: usize) -> &Shard {
        let at = self.place(page);
        if self.write.is_empty() { &self.read[at] } else { &self.write[at] }
    }

    /// See [`Shard::note`]; held for writing.
    fn note(&mut self, page: usize, parts: u16, now: u64) -> u64 {
        let at = self.place(page);
        self.write[at].note(page, parts, now)
    }

    /// The notes of pages `[first, last)` in page order; their parts held.
    fn range(&self, first: usize, last: usize) -> Vec<(usize, u64)> {
        match pieces(first, last) {
            Some(pieces) => pieces
                .into_iter()
                .flat_map(|(from, to)| self.shard(from).written.range(from..to).map(|(&p, &w)| (p, w)))
                .collect(),
            None => {
                let all = if self.write.is_empty() {
                    self.read.iter().flat_map(|s| s.written.range(first..last)).map(|(&p, &w)| (p, w)).collect::<Vec<_>>()
                } else {
                    self.write.iter().flat_map(|s| s.written.range(first..last)).map(|(&p, &w)| (p, w)).collect::<Vec<_>>()
                };
                let mut all = all;
                all.sort_unstable();
                all
            }
        }
    }
}

/// Of the pages of a copy that were written after it took them (page number,
/// the asking it took the page at): the parts to take again, a bit for each
/// 4 KiB, in the same order.
pub(crate) fn parts_written(pages: &[(usize, u64)]) -> Vec<u16> {
    let j = journal();
    pages.iter().map(|&(page, taken)| j.notes.parts_since(page, taken)).collect()
}

/// Numbers the askings; 0 is "never". Counted up with the journal held
/// (shared will do: what notes an answer holds it alone).
static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn next_asking() -> u64 {
    ASKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
}

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    if views {
        (start / BLOCK..end.div_ceil(BLOCK)).for_each(|block| FINE.set(block));
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

struct Journal {
    /// Per page, the asking that last found it written, and its parts'
    /// ([`Shard`]).
    notes: Notes,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

/// Read by many threads at once (a submission's refreshes run on several),
/// written only to note an answer: the kernel is asked, and view memory
/// guarded, with the range held in [`ASKING`] and not under this lock. Under
/// one lock for all of it those threads ran one at a time: 22 of 96 looks at
/// the submitting thread, and one in six of those at the six threads beside
/// it, found it waiting for that lock (run pooloff1).
static JOURNAL: RwLock<Journal> = RwLock::new(Journal {
    notes: Notes::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
});

fn journal() -> std::sync::RwLockReadGuard<'static, Journal> {
    if let Ok(j) = JOURNAL.try_read() {
        return j;
    }
    let began = std::time::Instant::now();
    let j = waited(1, || JOURNAL.read().unwrap_or_else(|e| e.into_inner()));
    port7_waited(P7_READ_WAITS, began);
    j
}

fn journal_mut() -> std::sync::RwLockWriteGuard<'static, Journal> {
    port7_count(P7_ALONE, 1);
    if let Ok(j) = JOURNAL.try_write() {
        return j;
    }
    let began = std::time::Instant::now();
    let j = waited(2, || JOURNAL.write().unwrap_or_else(|e| e.into_inner()));
    port7_waited(P7_ALONE_WAITS, began);
    j
}

/// Renderer port #7 (`VICTORY_PORT_JOURNAL=1`, off by default; read here,
/// and A/B-able in one run with `VICTORY_AB=<s>,VICTORY_PORT_JOURNAL`): the
/// journal's lock no longer makes the threads that read it wait for those
/// that note in it, nor those for each other, unless the standing asking
/// changes. (The switch also covers `crate::port_journal`'s heaps' records.)
///
/// What it was (runs guard-4, merge-4; ports 1, 2, 5, 6 on): every note in
/// the journal ([`JOURNAL`]) took it alone: the answers of askings found
/// written (`writes`, a submission's one asking, what shaders put in beside a
/// guard), and also what has nothing to change: each title buffer copy told
/// ([`wrote`]) with no asking standing, every flush's [`ask_ahead_done`] and
/// [`ask_once_end`]. 200 such takings a frame waited, 3-3.6 ms; the readers
/// (a refresh's every look, ~10,000 a frame on seven threads) waited 480
/// times, 5.4-5.9 ms; ~2.8 ms of it was the submitting thread's.
///
/// What it is with the switch: the pages' notes are in [`SHARDS`] parts of
/// their own ([`Notes`]), each with a lock of its own. What changes the
/// standing asking (its ranges, its number, whether it stands) still takes
/// the journal alone, and so does what our steps write while it stands
/// ([`wrote`], which moves its number). Noting an answer takes the journal
/// shared and the parts of its pages for writing ([`Notes::hold`]): an
/// asking about a range (`writes`), what shaders put in beside a guard
/// ([`landed_beside`]), and the pages a submission's asking found (noted
/// before its ranges and number are set, alone and briefly). What has
/// nothing to change looks first under the shared lock.
///
/// Why nothing is lost: an answer `(number, pages)` must hold every note of
/// a lower number of those pages. A number is taken only holding the parts
/// of the pages it notes or answers for (who takes a fresh number to answer
/// with holds the parts of its range for reading, switch or not), so a note
/// of a lower number is in, or its number is higher. The standing asking's
/// number is set with the journal held alone, after every shared holder that
/// took a lower one let go. A page's note only goes up ([`Shard::note`]).
/// Parts are taken in order, the journal's lock before them and never while
/// holding them, and no holder of a part waits for anything else: no two
/// wait for each other. Tests: `port7_tests`.
pub(crate) fn port_journal() -> bool {
    #[cfg(any(feature = "diagnostics", feature = "ab"))]
    if let Some(on) = crate::ab::pick("VICTORY_PORT_JOURNAL") {
        return on;
    }
    #[cfg(test)]
    if PORT_JOURNAL_IN_TEST.get() {
        return true;
    }
    // Read here: the switch of renderer port #7.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_JOURNAL").is_ok_and(|v| v.trim() == "1"))
}

#[cfg(test)]
thread_local! {
    /// [`port_journal`], for what this thread does.
    pub(crate) static PORT_JOURNAL_IN_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

thread_local! {
    /// Whether this thread is the submitting lane ("victory-submit").
    static ON_LANE: bool = std::thread::current().name() == Some("victory-submit");
}

/// Renderer port #7's counts since the last line: takings of the journal
/// alone; of them, those that waited and nanoseconds; waits for it shared
/// and nanoseconds; the lane's waits for it (both kinds) and nanoseconds;
/// takings alone found needless and left out; notings done shared; waits for
/// a part and nanoseconds.
static PORT7: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];
const P7_ALONE: usize = 0;
/// (A count and its nanoseconds next to it, as the others below.)
const P7_ALONE_WAITS: usize = 1;
const P7_READ_WAITS: usize = 3;
const P7_LANE_WAITS: usize = 5;
const P7_SKIPPED: usize = 7;
const P7_SHARED_NOTES: usize = 8;
const P7_PART_WAITS: usize = 9;

fn port7_count(what: usize, n: u64) {
    PORT7[what].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// A wait of kind `what` (a count, its nanoseconds next) that began then.
fn port7_waited(what: usize, began: std::time::Instant) {
    let ns = began.elapsed().as_nanos() as u64;
    port7_count(what, 1);
    port7_count(what + 1, ns);
    if what != P7_PART_WAITS && ON_LANE.with(|l| *l) {
        port7_count(P7_LANE_WAITS, 1);
        port7_count(P7_LANE_WAITS + 1, ns);
    }
}

/// Renderer port #7's journal counts, taken ([`crate::port_journal::line`]).
pub(crate) struct Port7Counts([u64; 12]);

pub(crate) fn port_journal_counts() -> Port7Counts {
    Port7Counts(std::array::from_fn(|i| PORT7[i].swap(0, std::sync::atomic::Ordering::Relaxed)))
}

impl Port7Counts {
    /// For the line, per frame.
    pub(crate) fn per_frame(&self, frames: f64) -> String {
        let n = |i: usize| self.0[i] as f64 / frames;
        let ms = |i: usize| self.0[i] as f64 / 1e6 / frames;
        format!(
            "the lane waited for its lock {:.1} times, {:.2} ms; all threads waited to take it alone {:.1} times ({:.2} ms) of {:.1} \
             takings alone, shared {:.1} times ({:.2} ms); {:.1} takings alone found needless, {:.1} notings done shared, {:.1} waits \
             for a part ({:.2} ms)",
            n(P7_LANE_WAITS),
            ms(P7_LANE_WAITS + 1),
            n(P7_ALONE_WAITS),
            ms(P7_ALONE_WAITS + 1),
            n(P7_ALONE),
            n(P7_READ_WAITS),
            ms(P7_READ_WAITS + 1),
            n(P7_SKIPPED),
            n(P7_SHARED_NOTES),
            n(P7_PART_WAITS),
            ms(P7_PART_WAITS + 1),
        )
    }
}

/// Since the last line: waits and nanoseconds of waiting, of all threads, for
/// the ranges held ([`ASKING`]), the journal to read it, the journal to note
/// in it, and the guards (faults of the title's among them).
#[cfg(feature = "diagnostics")]
static WAITS: [[std::sync::atomic::AtomicU64; 2]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 4];

#[cfg(feature = "diagnostics")]
fn waited<T>(what: usize, f: impl FnOnce() -> T) -> T {
    use std::sync::atomic::Ordering::Relaxed;
    let began = std::time::Instant::now();
    let got = f();
    WAITS[what][0].fetch_add(1, Relaxed);
    WAITS[what][1].fetch_add(began.elapsed().as_nanos() as u64, Relaxed);
    got
}

#[cfg(not(feature = "diagnostics"))]
fn waited<T>(_: usize, f: impl FnOnce() -> T) -> T {
    f()
}

/// [`WAITS`], taken, for the log: per frame.
#[cfg(feature = "diagnostics")]
pub(crate) fn lock_waits(frames: f64) -> String {
    let of = |what: usize| {
        let [n, ns] = [0, 1].map(|i| WAITS[what][i].swap(0, std::sync::atomic::Ordering::Relaxed));
        format!("{:.0} {:.2} ms", n as f64 / frames, ns as f64 / 1e6 / frames)
    };
    let h: [u64; 8] = std::array::from_fn(|i| HOLDS[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    let per = |v: u64| v as f64 / frames;
    let each = |v: u64| if h[0] > 0 { v as f64 / h[0] as f64 } else { 0.0 };
    let w: [u64; 3] = std::array::from_fn(|i| REFRESH_HOLDS[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    let p: [u64; 15] = std::array::from_fn(|i| PHASES[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    let ms = |i: usize| p[i] as f64 / 1e6 / frames;
    let (long_ns, long_first, long_last) = std::mem::take(&mut *LONGEST.lock().unwrap_or_else(|e| e.into_inner()));
    format!(
        "ranges held {}, journal read {}, journal noted in {}, guards {};          whole submissions waiting {:.1} for {:.2} ms (longest {:.2} ms), {:.1} ranges held as each began, {:.1} taken while each waited;          held longest: whole {:.2} ms, a range {:.2} ms ({} KiB at {:#x}); ranges held 10 ms or more {:.1}, all held {:.2} ms (VICTORY_ASK_ALL_FIRST={});          refreshes held {:.1} ranges, {:.1} of them sent to the guards, held there {:.2} ms;          inside whole holds: ahead spans {:.2} ms, split {:.2} ms, runs {:.2} ms, hot {:.2} ms, journal {:.2} ms ({:.0} pieces); rest take {:.2} ms, loop {:.2} ms (longest {:.2} ms), hot {:.2} ms, journal {:.2} ms; rest calls {:.1}, {:.0} ranges, {:.0} kept, {:.0} askings",
        of(0),
        of(1),
        of(2),
        of(3),
        per(h[0]),
        h[1] as f64 / 1e6 / frames,
        h[2] as f64 / 1e6,
        each(h[3]),
        each(h[4]),
        h[5] as f64 / 1e6,
        long_ns as f64 / 1e6,
        (long_last - long_first) * (PAGE >> 10),
        long_first * PAGE,
        per(h[6]),
        h[7] as f64 / 1e6 / frames,
        u8::from(all_first()),
        per(w[0]),
        per(w[1]),
        w[2] as f64 / 1e6 / frames,
        ms(0),
        ms(1),
        ms(2),
        ms(3),
        ms(4),
        per(p[13]),
        ms(5),
        ms(6),
        p[14] as f64 / 1e6,
        ms(7),
        ms(8),
        per(p[12]),
        per(p[9]),
        per(p[10]),
        per(p[11]),
    )
}

/// Since the last line: whole-submission holds ([`Asking::hold_all`]) that
/// waited, their nanoseconds, the longest of them; the ranges held when each
/// began to wait and those other threads took while it waited; the longest a
/// whole submission was held; ranges held 10 ms or more; nanoseconds of all
/// ranges held, summed. [`LONGEST`] is the longest range: nanoseconds and its
/// pages.
#[cfg(feature = "diagnostics")]
static HOLDS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];
#[cfg(feature = "diagnostics")]
static LONGEST: Mutex<(u64, usize, usize)> = Mutex::new((0, 0, 0));
/// Ranges [`writes`] held, those of them it sent to the guards (the kernel
/// refused them, or `VICTORY_ASK_SPANS` left them unasked), and nanoseconds
/// at the guards with the range held.
#[cfg(feature = "diagnostics")]
static REFRESH_HOLDS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// Inside the whole-submission holds, per part, nanoseconds: [`ask_ahead`]'s
/// reservations by spans (0), by split (1), its runs (2), `note_hot` (3), the
/// journal (4); [`ask_rest`]'s taking the journal (5), its loop over the
/// ranges (6), `note_hot` (7), the journal (8). Then counts: ranges
/// `ask_rest` was given (9), ranges it held in the journal at the end (10),
/// kernel askings it made (11), its calls (12), pieces `ask_ahead` had (13),
/// and the longest loop of `ask_rest` in nanoseconds (14).
#[cfg(feature = "diagnostics")]
static PHASES: [std::sync::atomic::AtomicU64; 15] = [const { std::sync::atomic::AtomicU64::new(0) }; 15];

#[cfg(feature = "diagnostics")]
fn phase(i: usize, since: &mut std::time::Instant) {
    let now = std::time::Instant::now();
    let ns = (now - *since).as_nanos() as u64;
    PHASES[i].fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    if i == 6 {
        PHASES[14].fetch_max(ns, std::sync::atomic::Ordering::Relaxed);
    }
    *since = now;
}

#[cfg(not(feature = "diagnostics"))]
fn phase(_: usize, _: &mut std::time::Instant) {}

#[cfg(feature = "diagnostics")]
fn phase_count(i: usize, n: usize) {
    PHASES[i].fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(not(feature = "diagnostics"))]
fn phase_count(_: usize, _: usize) {}

/// Ranges held so far, for "taken while each waited".
#[cfg(feature = "diagnostics")]
static TAKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a whole submission's asking that has to wait ([`Asking::hold_all`])
/// goes before ranges asked for after it (`VICTORY_ASK_ALL_FIRST=1`; off by
/// default). It waits for no range to be held, and the refreshing threads
/// each hold one as they ask: with enough of them one is always held, and
/// the submitting thread waited 3.6 to 16 s a frame once the world loaded
/// (laptop run bp5-nowait). With the switch those ranges wait for it instead.
fn all_first() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ALL_FIRST").is_some_and(|v| v == "1"))
}

/// Whole submissions waiting to be held ([`all_first`]); changed with
/// [`ASKING`] held.
static ALL_WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The page ranges some thread is asking about: asking forgets what it
/// answers, so until the answer is in the journal no other thread may ask
/// about any of the same pages (it would be told they were not written).
/// Taken before the journal's lock, never under it.
static ASKING: (Mutex<Vec<(usize, usize)>>, Condvar) = (Mutex::new(Vec::new()), Condvar::new());

/// Pages `[first, last)` held in [`ASKING`] until dropped, and since when.
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
struct Asking(usize, usize, std::time::Instant);

impl Asking {
    fn hold(first: usize, last: usize) -> Asking {
        use std::sync::atomic::Ordering::Relaxed;
        let all = (first, last) == (0, usize::MAX);
        let blocked = |held: &Vec<(usize, usize)>| {
            held.iter().any(|&(a, b)| a < last && first < b) || (!all && ALL_WAITING.load(Relaxed) > 0)
        };
        let mut held = ASKING.0.lock().unwrap_or_else(|e| e.into_inner());
        if blocked(&held) {
            let first_in = all && all_first();
            if first_in {
                ALL_WAITING.fetch_add(1, Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            let (began, holders, taken) = (std::time::Instant::now(), held.len() as u64, TAKEN.load(Relaxed));
            held = waited(0, || {
                let mut held = held;
                while if first_in { held.iter().any(|&(a, b)| a < last && first < b) } else { blocked(&held) } {
                    held = ASKING.1.wait(held).unwrap_or_else(|e| e.into_inner());
                }
                held
            });
            if first_in {
                ALL_WAITING.fetch_sub(1, Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            if all {
                let ns = began.elapsed().as_nanos() as u64;
                HOLDS[0].fetch_add(1, Relaxed);
                HOLDS[1].fetch_add(ns, Relaxed);
                HOLDS[2].fetch_max(ns, Relaxed);
                HOLDS[3].fetch_add(holders, Relaxed);
                HOLDS[4].fetch_add(TAKEN.load(Relaxed) - taken, Relaxed);
            }
        }
        #[cfg(feature = "diagnostics")]
        TAKEN.fetch_add(1, Relaxed);
        held.push((first, last));
        Asking(first, last, std::time::Instant::now())
    }

    /// Everything: an asking for a whole submission.
    fn hold_all() -> Asking {
        Asking::hold(0, usize::MAX)
    }
}

impl Drop for Asking {
    fn drop(&mut self) {
        let mut held = ASKING.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = held.iter().position(|&r| r == (self.0, self.1)) {
            held.swap_remove(i);
        }
        drop(held);
        ASKING.1.notify_all();
        #[cfg(feature = "diagnostics")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            let ns = self.2.elapsed().as_nanos() as u64;
            HOLDS[7].fetch_add(ns, Relaxed);
            if (self.0, self.1) == (0, usize::MAX) {
                HOLDS[5].fetch_max(ns, Relaxed);
                return;
            }
            if ns >= 10_000_000 {
                HOLDS[6].fetch_add(1, Relaxed);
            }
            let mut longest = LONGEST.lock().unwrap_or_else(|e| e.into_inner());
            if ns > longest.0 {
                *longest = (ns, self.0, self.1);
            }
        }
    }
}

thread_local! {
    /// Room for the kernel's answer.
    static FOUND: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The pages of `[first, last)` the kernel says were written since it was
/// last asked, in order, each once, with their written parts; `None` if the range is not write-watched
/// memory of one reservation. The caller holds the range ([`Asking`]).
fn kernel_written(first: usize, last: usize) -> Option<Vec<(usize, u16)>> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    if refused_before(first, last) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::AskSkipped, 1);
        return None;
    }
    if ask_spans::on() && ask_spans::skip(first, last) {
        return None;
    }
    FOUND.with(|found| {
        let mut found = found.borrow_mut();
        let mut count = (last - first) * PAGE / KERNEL_PAGE;
        if found.len() < count {
            found.resize(count, 0);
        }
        let mut granularity = 0;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        // SAFETY: the answer buffer holds `count` entries.
        let failed = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        ask_spans::told(first, last, failed);
        #[cfg(feature = "diagnostics")]
        {
            tally(10, u64::from(failed));
            tally(11, started.elapsed().as_nanos() as u64);
            use crate::copystat::{C, add, again};
            if failed {
                add(C::AskRefused, 1);
                add(C::AskRefusedUs, started.elapsed().as_micros() as u64);
                if again(2, (first, last, 0)) {
                    add(C::AskRefusedAgain, 1);
                }
            } else if again(1, (first, last, 0)) {
                add(C::AskAgain, 1);
                add(C::AskAgainBytes, ((last - first) * PAGE) as u64);
            }
        }
        if failed {
            refused(first, last);
            return None;
        }
        #[cfg(feature = "diagnostics")]
        {
            tally(12, count as u64);
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
            slow(started.elapsed().as_micros() as u64);
            tally(0, 1);
            tally(1, ((last - first) * (PAGE >> 10)) as u64);
        }
        let mut pages: Vec<(usize, u16)> = Vec::new();
        for &at in &found[..count] {
            let (page, part) = (at / PAGE, 1u16 << (at % PAGE / KERNEL_PAGE));
            match pages.last_mut() {
                Some((last, parts)) if *last == page => *parts |= part,
                _ => pages.push((page, part)),
            }
        }
        Some(pages)
    })
}

/// Whether a range the kernel refused to answer for (memory that is not
/// write-watched: the title's views, its other heaps) is not asked about
/// again in the same frame (`VICTORY_REFUSED_ONCE=0`: asked every time, as
/// before).
///
/// Of 460 kernel askings a frame some 210 were refused (play-prof, gameplay):
/// the submission's one asking asked every view range its lists refresh, the
/// refresh that followed asked the same range again before it went to the
/// guards, and the next submission did both once more. A refusal stands for
/// as long as the memory stays as it is mapped, and the title maps and
/// unmaps its views between frames, not in the middle of one; a range the
/// title made write-watched in the frame is then guarded, or copied whole,
/// until the next frame asks again. Not in the tests: their frame never
/// ends, and one test's freed memory is another's write-watched memory at
/// the same address (`written_pages_are_reported_to_every_asker` refused in
/// about one run in four).
fn refused_once() -> bool {
    if cfg!(test) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REFUSED_ONCE").is_none_or(|v| v != "0"))
}

/// Ranges (first page, past the last) the kernel refused this frame, and the
/// frame.
static REFUSED: Mutex<(u64, Option<std::collections::HashSet<(usize, usize)>>)> = Mutex::new((0, None));

/// Whether the kernel refused to answer for `[first, last)` before in this
/// frame ([`refused_once`]).
fn refused_before(first: usize, last: usize) -> bool {
    if !refused_once() {
        return false;
    }
    let frame = crate::frame::current();
    let mut refused = REFUSED.lock().unwrap_or_else(|e| e.into_inner());
    if refused.0 != frame {
        *refused = (frame, None);
        return false;
    }
    refused.1.as_ref().is_some_and(|ranges| ranges.contains(&(first, last)))
}

/// The kernel refused to answer for `[first, last)`.
fn refused(first: usize, last: usize) {
    if !refused_once() {
        return;
    }
    let frame = crate::frame::current();
    let mut refused = REFUSED.lock().unwrap_or_else(|e| e.into_inner());
    if refused.0 != frame {
        *refused = (frame, None);
    }
    refused.1.get_or_insert_with(Default::default).insert((first, last));
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// [`AHEAD_GAP`], or `VICTORY_AHEAD_GAP_MB` MiB.
fn ahead_gap() -> usize {
    static GAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| {
        std::env::var("VICTORY_AHEAD_GAP_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(AHEAD_GAP, |mb| (mb << 20) / PAGE)
    })
}

/// Whether [`ask_ahead`] keeps each asking inside one reservation
/// (`VICTORY_ASK_SPLIT=1`; off by default). A run across two reservations is
/// refused, and [`ask_run`] then asks it again by halves: in Leafy's
/// gameplay frames (leafy-0314) 105 of 227 askings a frame were refused.
/// With the switch a run that is refused anyway lies in one reservation that
/// is not write-watched: it is not halved, and the reservation is not asked
/// about again in the frame. Its ranges are left to [`writes`], as a lone
/// range is.
fn ask_split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_SPLIT").is_some_and(|v| v == "1"))
}

/// Whether [`ask_ahead`] finds its pieces' reservations before it holds
/// the whole submission (`VICTORY_ASK_HOMES=1`; off by default), from
/// `VICTORY_ASK_SPANS`' regions when that is on. With `VICTORY_ASK_SPLIT`
/// the queries stood 300-380 ms a frame under the hold, and every refresh
/// waited (run lp-ask-rf).
fn ask_homes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_HOMES").is_some_and(|v| v == "1"))
}

/// The reservation `page` lies in: its allocation base, in pages.
#[cfg(test)]
fn reservation(page: usize) -> usize {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((page * PAGE) as *const _, &mut info, size_of_val(&info)) };
    if n == 0 { usize::MAX } else { info.AllocationBase as usize / PAGE }
}

/// [`reservation`] of each piece's first page, `pieces` in order: one query
/// for all the pieces that start in the region it answered for. A query
/// stood 1.4 ms (run lp-ask-ph: 3,800 pieces a frame, 5.4 s under the whole
/// submission's hold, while every refresh waited).
fn reservations(pieces: &[(usize, usize)]) -> Vec<usize> {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut homes = Vec::with_capacity(pieces.len());
    // The last region answered: its bytes and allocation base, in pages.
    let mut known: Option<(usize, usize, usize)> = None;
    for &(first, _) in pieces {
        let at = first * PAGE;
        if let Some((start, end, base)) = known
            && start <= at
            && at < end
        {
            homes.push(base);
            continue;
        }
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            known = None;
            homes.push(usize::MAX);
            continue;
        }
        let start = info.BaseAddress as usize;
        let base = info.AllocationBase as usize / PAGE;
        known = Some((start, start.saturating_add(info.RegionSize), base));
        homes.push(base);
    }
    homes
}

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        // Not view memory known by its bits: the kernel would refuse the
        // asking (and, asked together with others, theirs).
        .filter(|r| r.0 < r.1 && !published(r.0, r.1))
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if journal().kept {
        return;
    }
    let early = ask_homes();
    let held = if early { None } else { Some(Asking::hold_all()) };
    let mut t = std::time::Instant::now();
    phase_count(13, pieces.len());
    let spans = ask_spans::on();
    let bases = if spans { ask_spans::bases(&mut pieces) } else { Vec::new() };
    phase(0, &mut t);
    let split = ask_split();
    let homes: Vec<usize> = if !split {
        Vec::new()
    } else if early && spans {
        // A piece kept by ASK_SPANS lies in one reservation, found from
        // the frame's regions: its base is the piece's.
        bases.iter().map(|b| b / PAGE).collect()
    } else {
        reservations(&pieces)
    };
    phase(1, &mut t);
    let _all = held.unwrap_or_else(Asking::hold_all);
    if early {
        t = std::time::Instant::now();
    }
    let (mut asked, mut written) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + ahead_gap()
            && last.max(pieces[k].1) - first <= AHEAD_MAX
            && (!spans || bases[k] == bases[i])
            && (!split || homes[k] == homes[i])
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        if !split {
            ask_run(&pieces[i..k], false, &mut asked, &mut written);
        } else if k - i >= 2 && !refused_before(homes[i], homes[i]) {
            // (A reservation refused is noted as the range (base, base),
            // which no asking has.)
            match kernel_written(first, last) {
                Some(pages) => {
                    written.extend(pages);
                    asked.push((first, last));
                }
                None => refused(homes[i], homes[i]),
            }
        }
        i = k;
    }
    phase(2, &mut t);
    note_hot(&written);
    phase(3, &mut t);
    if port_journal() {
        // Renderer port #7: the pages noted shared, under their parts; then
        // the number and the ranges, alone (no shared holder with a lower
        // number is left by then).
        let now = note_shared(&written);
        let mut j = journal_mut();
        ahead_normalize(&mut asked);
        j.ahead_asked = now;
        j.ahead = asked;
        phase(4, &mut t);
        return;
    }
    let mut j = journal_mut();
    ahead_normalize(&mut asked);
    j.ahead_asked = next_asking();
    j.ahead = asked;
    let now = j.ahead_asked;
    for (page, parts) in written {
        j.notes.note(page, parts, now);
    }
    phase(4, &mut t);
}

/// Renderer port #7: notes `written` (pages and their parts) as found by an
/// asking with a number taken now, holding the journal shared and their
/// parts for writing; the number.
fn note_shared(written: &[(usize, u16)]) -> u64 {
    let j = journal();
    let mut held = j.notes.hold(shards_of_pages(written.iter().map(|w| w.0)), true);
    let now = next_asking();
    for &(page, parts) in written {
        held.note(page, parts, now);
    }
    port7_count(P7_SHARED_NOTES, 1);
    now
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// Whether a thread asks, between a submission's askings, about the memory
/// they found written (`VICTORY_ASK_EARLY=<ms>` between its rounds; off
/// unless given: in run cpu-2 the submissions' own askings were none the
/// cheaper for its 650 a frame, 12 to 18 ms where 9 were).
///
/// An asking's cost is the pages it finds written and resets, some 1.4
/// microseconds each, and the title writes 7,000 a frame: the first
/// submission that refreshes its 174 MB heap stood 8 to 10 ms in one call
/// (session play-1125-0203). Asked a little at a time while the title
/// records, the same pages are in the journal when the submission comes, and
/// its own asking finds those written since.
///
/// Nothing is seen later than before: a page goes into the journal as found
/// written by an asking of its own number, as from any thread that asks, and
/// the submission still asks for itself. A page the title writes again after
/// such a round faults once more.
fn early() -> Option<std::time::Duration> {
    static EVERY: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let ms = match std::env::var("VICTORY_ASK_EARLY") {
            Ok(v) => v.trim().parse::<f64>().unwrap_or(2.0),
            Err(_) => 0.0,
        };
        std::time::Duration::try_from_secs_f64(ms / 1000.0).ok().filter(|d| !d.is_zero())
    })
}

/// The pieces of memory (by [`EARLY_PIECE`] pages) that submissions' askings
/// found written lately: how many pages each, halved every round of 64.
static HOT: Mutex<BTreeMap<usize, u32>> = Mutex::new(BTreeMap::new());
/// The pieces asked in a round, at most.
const EARLY_MOST: usize = 24;
/// The pages of a piece: 2 MiB.
const EARLY_PIECE: usize = (2 << 20) / PAGE;
/// Told when a submission asked.
static HOT_MOVED: Condvar = Condvar::new();

/// Since the last frame line: pieces asked early, the microseconds of it.
static EARLY: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// [`EARLY`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn early_asks() -> [u64; 2] {
    [0, 1].map(|i| EARLY[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// A submission's asking found `written` (pages).
fn note_hot(written: &[(usize, u16)]) {
    let Some(every) = early() else {
        return;
    };
    let mut hot = HOT.lock().unwrap_or_else(|e| e.into_inner());
    for (page, _) in written {
        let n = hot.entry(page / EARLY_PIECE).or_default();
        *n = (*n + 1).min(1 << 10);
    }
    HOT_MOVED.notify_one();
    drop(hot);
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("victory-ask-early".into()).spawn(move || ask_early(every));
    });
}

fn ask_early(every: std::time::Duration) {
    use std::sync::atomic::Ordering::Relaxed;
    // So many rounds without a submission's asking: wait for one.
    const IDLE: u32 = 100;
    let mut round = 0u32;
    let mut idle = 0;
    let mut last = 0u64;
    loop {
        std::thread::sleep(every);
        let pieces: Vec<usize> = {
            let mut hot = HOT.lock().unwrap_or_else(|e| e.into_inner());
            let now = ASKED.load(Relaxed);
            idle = if now == last { idle + 1 } else { 0 };
            while idle > IDLE {
                hot = HOT_MOVED.wait(hot).unwrap_or_else(|e| e.into_inner());
                idle = 0;
            }
            round += 1;
            if round.is_multiple_of(64) {
                hot.retain(|_, n| {
                    *n /= 2;
                    *n > 0
                });
            }
            // The most written: a piece with a few pages is as dear to ask as
            // one with many.
            let mut most: Vec<(u32, usize)> = hot.iter().filter(|h| *h.1 >= 16).map(|(&p, &n)| (n, p)).collect();
            most.sort_unstable_by(|a, b| b.cmp(a));
            most.into_iter().take(EARLY_MOST).map(|(_, p)| p).collect()
        };
        let began = std::time::Instant::now();
        let mut asked = 0;
        for piece in pieces {
            // (A submission's asking stands: what it holds is answered from
            // the journal, and costs nothing here.)
            if published(piece * EARLY_PIECE * PAGE, (piece + 1) * EARLY_PIECE * PAGE) {
                continue;
            }
            asked += 1;
            let _ = writes(piece * EARLY_PIECE * PAGE, (piece + 1) * EARLY_PIECE * PAGE, false);
        }
        EARLY[0].fetch_add(asked, Relaxed);
        EARLY[1].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
        last = ASKED.load(Relaxed);
    }
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    // Renderer port #7: nothing to change is seen shared.
    if port_journal() {
        let j = journal();
        if j.kept || j.ahead.is_empty() {
            port7_count(P7_SKIPPED, 1);
            return;
        }
    }
    let mut j = journal_mut();
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    ask_once_end();
    let before = ask_spans::mark();
    ask_ahead(ranges);
    ask_rest(ranges);
    ask_spans::submitted(ranges, before);
}

/// `VICTORY_ASK_FIRST=1` (off by default; with [`once`]): a submission's one
/// asking is begun as soon as the submission has the lock, on a thread kept
/// for it ([`ask_first`]), beside the planning of the steps done on the card
/// and the look at the flights in the air; the submission waits for it where
/// it asked before, and asks for more there if it has to ([`ask_more`]).
///
/// Why: the asking is the kernel resetting the pages found written, 4 to 5
/// ms a frame on the thread a frame waits for, and it reads nothing the
/// steps before it make (session 2026-11-28 00:45: 4.3 to 5.4 ms of asking,
/// 5.8 to 18.5 ms of planning ahead of it, a frame).
///
/// What changes: the asking is made a few hundred microseconds earlier in
/// the submission. Memory the title's other threads write in that time is
/// seen by the next submission's asking, as what they write while the
/// refreshes run is today ([`once`]); what this submission's lists read was
/// written before they were submitted. What does not change: after a wait
/// for flights (what they landed is title memory written) the asking is made
/// again, whole, where it was made before; and the ranges of the list of
/// steps done on the card, which is made after the asking was begun, are
/// asked there.
pub(crate) fn first() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| once() && std::env::var_os("VICTORY_ASK_FIRST").is_some_and(|v| v == "1"))
}

/// A submission's one asking, begun by [`ask_first`]: whether it is made.
pub(crate) struct AskedFirst(std::sync::Arc<(Mutex<bool>, Condvar)>);

impl AskedFirst {
    /// Until that asking is made (or has given up: [`ask_more`] finds out).
    pub(crate) fn wait(self) {
        let done = &*self.0;
        let mut made = done.0.lock().unwrap_or_else(|e| e.into_inner());
        while !*made {
            made = done.1.wait(made).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// [`ask_once`] of `ranges`, begun on the thread kept for it ([`first`]);
/// here and now if there is no such thread.
pub(crate) fn ask_first(ranges: Vec<(usize, usize)>) -> AskedFirst {
    type Job = Box<dyn FnOnce() + Send>;
    static THREAD: std::sync::OnceLock<Option<Mutex<std::sync::mpsc::Sender<Job>>>> = std::sync::OnceLock::new();
    let done = std::sync::Arc::new((Mutex::new(false), Condvar::new()));
    let told = done.clone();
    let job: Job = Box::new(move || {
        // (However it ends, the submission that waits goes on: where no
        // asking stands it makes its own.)
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ask_once(&ranges)));
        *told.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
        told.1.notify_all();
    });
    let thread = THREAD.get_or_init(|| {
        let (send, jobs) = std::sync::mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("victory-ask".into())
            .spawn(move || {
                crate::cores::keep_on_fast_cores("victory-ask");
                jobs.into_iter().for_each(|job| job())
            })
            .ok()
            .map(|_| Mutex::new(send))
    });
    match thread {
        Some(send) => {
            if let Err(job) = send.lock().unwrap_or_else(|e| e.into_inner()).send(job) {
                (job.0)();
            }
        }
        None => job(),
    }
    AskedFirst(done)
}

/// After [`ask_first`] was waited for: asks about what of `ranges` the
/// asking that stands does not hold, and it stands on; all of them, as
/// [`ask_once`] does, if none stands (the first found nothing to hold, or
/// gave up). Whether anything was asked.
pub(crate) fn ask_more(ranges: &[(usize, usize)]) -> bool {
    if !once() {
        return false;
    }
    let held = {
        let j = journal();
        j.kept.then(|| j.ahead.clone())
    };
    let Some(held) = held else {
        ask_once(ranges);
        return true;
    };
    // (Outside the journal's lock: `published` has locks of its own.)
    let all_held = ranges.iter().all(|&(start, end)| {
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        start >= end || published(start, end) || held.iter().any(|&(a, b)| a <= first && last <= b)
    });
    if all_held {
        return false;
    }
    ask_rest(ranges);
    true
}

/// [`ask_once`] after its [`ask_ahead`]: what that left to `writes`, and the
/// asking stands.
fn ask_rest(ranges: &[(usize, usize)]) {
    let _all = Asking::hold_all();
    let mut t = std::time::Instant::now();
    let mut ahead = {
        let mut j = journal_mut();
        std::mem::take(&mut j.ahead)
    };
    phase(5, &mut t);
    phase_count(12, 1);
    phase_count(9, ranges.len());
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    let mut written = Vec::new();
    let mut added = Vec::new();
    for &(start, end) in ranges {
        if start >= end || published(start, end) {
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !ahead_holds(&ahead, first, last) && !added.iter().any(|&(a, b)| a <= first && last <= b) {
            phase_count(11, 1);
            if let Some(pages) = kernel_written(first, last) {
                written.extend(pages);
                added.push((first, last));
            }
        }
    }
    // (Added after the search: `ahead_holds` needs `ahead` in order.)
    ahead.extend(added);
    phase(6, &mut t);
    phase_count(10, ahead.len());
    note_hot(&written);
    phase(7, &mut t);
    // A number of its own, and the one this asking stands under: between
    // the two parts another thread may have asked about a range of the
    // second, copied it, and the title written it since. Under the first
    // part's number that write was older than the copy, and never copied.
    if port_journal() {
        // Renderer port #7: noted shared first (see `ask_ahead`).
        let now = note_shared(&written);
        let mut j = journal_mut();
        ahead_normalize(&mut ahead);
        j.ahead_asked = now;
        j.kept = !ahead.is_empty();
        j.ahead = ahead;
        drop(j);
        phase(8, &mut t);
        return;
    }
    let mut j = journal_mut();
    let now = next_asking();
    j.ahead_asked = now;
    for (page, parts) in written {
        j.notes.note(page, parts, now);
    }
    ahead_normalize(&mut ahead);
    j.kept = !ahead.is_empty();
    j.ahead = ahead;
    drop(j);
    phase(8, &mut t);
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    // Renderer port #7: none standing is seen shared.
    if port_journal() && !journal().kept {
        port7_count(P7_SKIPPED, 1);
        return;
    }
    let mut j = journal_mut();
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` (the title's buffer copies, done at submission): its
/// pages count as written after that asking, so the refreshes that follow copy them. Told while a
/// submission's one asking stands, or at all while [`noting`] is on.
pub(crate) fn wrote(at: usize, len: usize) {
    // Renderer port #7: with no asking standing there is nothing to note,
    // which is seen shared. (Standing, its number moves: alone, as before.)
    if port_journal() && (len == 0 || !journal().kept) {
        port7_count(P7_SKIPPED, 1);
        return;
    }
    let mut j = journal_mut();
    if (!j.kept && !noting()) || len == 0 {
        return;
    }
    let now = next_asking();
    // What is copied from now on counts as copied after these writes (fork
    // PR 410): answers older than the writes in them made every refresh
    // until the next asking copy the pages again.
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.notes.note(page, WHOLE, now);
    }
    for (_, others) in elsewhere(at, len) {
        for other in others {
            j.notes.note(other / PAGE, WHOLE, now);
        }
    }
}

/// Our writes into title memory `spans` (address, bytes), whose bytes are in, are told to the
/// journal exactly, by 4 KiB, whether or not an asking stands: a copy that took those pages before
/// is refreshed after them, and a refresh answered from the journal sees them. What shaders wrote
/// (landed by the shadow copies) and read-backs are told this way.
/// `VICTORY_NOTE_WRITES=0`: not told, as before ([`noting`]).
pub(crate) fn told(spans: &[(usize, usize)]) {
    if spans.is_empty() || !noting() {
        return;
    }
    let elsewhere: Vec<(usize, Vec<usize>)> = spans.iter().flat_map(|&(at, len)| elsewhere(at, len)).collect();
    let mut j = journal_mut();
    let now = next_asking();
    j.ahead_asked = now;
    for &(at, len) in spans.iter().filter(|s| s.1 > 0) {
        for page in at / KERNEL_PAGE..(at + len - 1) / KERNEL_PAGE + 1 {
            j.notes.note(page / PARTS, 1u16 << (page % PARTS), now);
        }
    }
    for (_, others) in &elsewhere {
        for &other in others {
            j.notes.note(other / PAGE, WHOLE, now);
        }
    }
}

/// Whether our writes into title memory are told to the journal whether or not an asking stands
/// ([`wrote`], [`told`]). On by default; `VICTORY_NOTE_WRITES=0` tells [`wrote`] only while an
/// asking stands and [`told`] nothing, as before. Each note takes the journal's lock: this switch
/// measures that cost.
pub(crate) fn noting() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var_os("VICTORY_NOTE_WRITES").is_none_or(|v| v != "0"));
    *ON
}

/// The blocks of `[at, at + len)` whose pages the title shows at other
/// addresses too, and there.
fn elsewhere(at: usize, len: usize) -> Vec<(usize, Vec<usize>)> {
    if !view_guard() || len == 0 {
        return Vec::new();
    }
    crate::inplace::shown_elsewhere(at / BLOCK * BLOCK, at.saturating_add(len))
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`]. Adds what was
/// asked to `asked` and the pages found written to `written`. With
/// [`ask_spans`] on a run lies in one reservation and is not split.
fn ask_run(pieces: &[(usize, usize)], split: bool, asked: &mut Vec<(usize, usize)>, written: &mut Vec<(usize, u16)>) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if split {
        ask_spans::split();
    }
    if let Some(pages) = kernel_written(first, last) {
        written.extend(pages);
        asked.push((first, last));
        return;
    }
    if ask_spans::on() {
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(a, true, asked, written);
    ask_run(b, true, asked, written);
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = journal();
    ahead_holds(&j.ahead, first, last).then(|| (j.ahead_asked, j.notes.range(first, last)))
}

/// Puts `ahead` in order and one range of those that touch: a lookup is a
/// binary search, not a walk of every range a submission refreshes.
fn ahead_normalize(ahead: &mut Vec<(usize, usize)>) {
    ahead.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ahead.len());
    for &(a, b) in ahead.iter() {
        match out.last_mut() {
            Some((_, e)) if a <= *e => *e = (*e).max(b),
            _ => out.push((a, b)),
        }
    }
    *ahead = out;
}

/// Whether `ahead` (in order, one range of those that touch) holds
/// `[first, last)` whole.
fn ahead_holds(ahead: &[(usize, usize)], first: usize, last: usize) -> bool {
    let i = ahead.partition_point(|&(a, _)| a <= first);
    i > 0 && ahead[i - 1].1 >= last
}

/// Whether [`writes`] of `[start, end)` would wait for another thread's
/// asking of some of it (a submission's holds everything).
pub(crate) fn would_wait(start: usize, end: usize, guarded: bool) -> bool {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let ahead = ahead_holds(&journal().ahead, first, last);
    if ahead || guarded && published(start, end) {
        return false;
    }
    ASKING.0.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|&(a, b)| a < last && first < b)
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    // Guarding asks Windows about the pages and protects them: a lazy view
    // is none to it yet.
    if guarded {
        materialize(start, end.saturating_sub(start));
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // Asked ahead with the submission's other ranges: the journal has it.
    // (The standing number changes only with the journal held alone, so its
    // notes are all in: a part at a time will do.)
    let ahead = |j: &Journal| ahead_holds(&j.ahead, first, last).then(|| (j.ahead_asked, j.notes.range(first, last)));
    // A fresh number and the notes of the range, its parts held meanwhile
    // (renderer port #7, switch or not: who notes shared takes its number
    // holding the parts it notes in).
    let fresh = || {
        let j = journal();
        let held = j.notes.hold(shards_of(first, last), false);
        (next_asking(), held.range(first, last))
    };
    if let Some(known) = ahead(&journal()) {
        return Some(known);
    }
    // Read-only since an asking noted it: the bits first, then the journal.
    if guarded && published(start, end) {
        #[cfg(feature = "diagnostics")]
        {
            tally(2, 1);
            crate::faultbook::note(crate::faultbook::What::Known, first, last, 0);
        }
        return Some(fresh());
    }
    // Asked, and the answer noted, with the range held: nothing else asks
    // about any of it in between.
    let _range = Asking::hold(first, last);
    if let Some(known) = ahead(&journal()) {
        return Some(known);
    }
    #[cfg(feature = "diagnostics")]
    REFRESH_HOLDS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut guards_asked = false;
    let mut loosely = false;
    // (View memory read-only by our hand: the kernel would refuse.)
    let refused = guarded && light() && PROTECTED.any(first, last);
    let pages = match if refused { None } else { kernel_written(first, last) } {
        Some(pages) => pages,
        None => {
            guards_asked = true;
            if !guarded {
                return None;
            }
            #[cfg(feature = "diagnostics")]
            let guarding = std::time::Instant::now();
            let told = if loose_ask() && light() { guard_parts_loose(first * PAGE, last * PAGE) } else { None };
            loosely = told.is_some();
            let written = told.or_else(|| guard_parts(first * PAGE, last * PAGE));
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                let us = guarding.elapsed().as_micros() as u64;
                add(C::GuardUs, us);
                REFRESH_HOLDS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                REFRESH_HOLDS[2].fetch_add(guarding.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                if written.is_some() {
                    add(C::Watches, 1);
                    add(C::WatchUs, us);
                    slow(us);
                }
            }
            written?
        }
    };
    if pages.is_empty() {
        // Nothing to note: the others may go on reading.
        return Some(fresh());
    }
    let answer = if port_journal() {
        // Renderer port #7: shared, the range's parts held for writing (the
        // range is held too: no other asking notes any of it meanwhile).
        let j = journal();
        let mut held = j.notes.hold(shards_of(first, last) | shards_of_pages(pages.iter().map(|p| p.0)), true);
        let now = next_asking();
        for &(page, parts) in &pages {
            held.note(page, parts, now);
        }
        port7_count(P7_SHARED_NOTES, 1);
        (now, held.range(first, last))
    } else {
        let mut j = journal_mut();
        let now = next_asking();
        for &(page, parts) in &pages {
            j.notes.note(page, parts, now);
        }
        (now, j.notes.range(first, last))
    };
    // Noted: what the guards made clean of it is known by its bits from
    // now on. (The range is still held: no other asking of any of it came
    // between.)
    if guards_asked && light() {
        let blocks: Vec<usize> = pages.iter().map(|p| p.0).collect();
        if loosely { publish_loose(&blocks) } else { publish(&blocks) }
    }
    Some(answer)
}

/// Whether nothing of `[start, end)` was written that a copy with the notes
/// `copied` (the asking after which each page from page `base` on was taken)
/// and `ranged` (the asking after which a range that holds this one was
/// taken whole, or 0) does not hold, as [`writes`] would answer with nothing
/// asked: the submission's asking covers the range, or its blocks are known
/// by their bits. `false` if something would have to be asked.
pub(crate) fn nothing_new(start: usize, end: usize, base: usize, copied: &[u64], ranged: u64) -> bool {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // The bits first, then the journal, as in `writes`.
    let known = published(start, end);
    #[cfg(feature = "diagnostics")]
    if known {
        crate::faultbook::note(crate::faultbook::What::Known, first, last, 1);
    }
    let j = journal();
    let Some(taken) = first.checked_sub(base).and_then(|from| copied.get(from..last - base)) else {
        return false;
    };
    new_in(&j, known, first, last, taken, ranged)
}

/// [`nothing_new`] past the bits, `taken` cut to pages `first..last`.
fn new_in(j: &Journal, known: bool, first: usize, last: usize, taken: &[u64], ranged: u64) -> bool {
    if !known && !ahead_holds(&j.ahead, first, last) {
        return false;
    }
    (ranged != 0 || taken.iter().all(|&at| at != 0))
        && j.notes.each(first, last, |page, write| write <= taken[page - first].max(ranged))
}

/// [`nothing_new`] of many ranges, each `(start, end, taken, ranged)` with
/// `taken` the asking after which each of its pages was taken: the bits of
/// all of them first, then the journal read once for all.
pub(crate) fn nothing_new_all(looks: &[(usize, usize, Vec<u64>, u64)]) -> Vec<bool> {
    let known: Vec<bool> = looks
        .iter()
        .map(|&(start, end, ..)| {
            let known = published(start, end);
            #[cfg(feature = "diagnostics")]
            if known {
                crate::faultbook::note(crate::faultbook::What::Known, start / PAGE, end.div_ceil(PAGE), 1);
            }
            known
        })
        .collect();
    let j = journal();
    looks
        .iter()
        .zip(known)
        .map(|((start, end, taken, ranged), known)| new_in(&j, known, start / PAGE, end.div_ceil(PAGE), taken, *ranged))
        .collect()
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut new: Vec<(usize, u64)> = Vec::new();
        for (page, write) in pages {
            let last = seen.get(&page).copied();
            if last.is_some_and(|last| last >= write) || (page * PAGE).max(from) >= ((page + 1) * PAGE).min(to) {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            new.push((page, last.unwrap_or(0)));
        }
        // Of a page taken before, the parts written since, where they are
        // known (`VICTORY_TAKE_PARTS=0`: the page).
        let parts = if crate::toggled!("VICTORY_TAKE_PARTS") { parts_written(&new) } else { vec![WHOLE; new.len()] };
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (&(page, last), &parts) in new.iter().zip(&parts) {
            let mut parts = if last == 0 { WHOLE } else { parts };
            while parts != 0 {
                let first = parts.trailing_zeros() as usize;
                let past = first + (!(parts >> first)).trailing_zeros() as usize;
                parts &= if past >= PARTS { 0 } else { u16::MAX << past };
                let (start, end) = ((page * PAGE + first * KERNEL_PAGE).max(from), (page * PAGE + past.min(PARTS) * KERNEL_PAGE).min(to));
                if start >= end {
                    continue;
                }
                match ranges.last_mut() {
                    Some((_, last)) if *last == start => *last = end,
                    _ => ranges.push((start, end)),
                }
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Whether the refreshes of host copies go the light way
/// (`VICTORY_LIGHT_REFRESH=0`: as before): view memory nothing wrote since
/// it was last asked about is known by its bits ([`PUBLISHED`]), a copy
/// nothing of whose range changed is told so before any thread is woken
/// for it (`crate::host`, `Shadow::nothing_new`), and the ends of a range,
/// pages the range holds in part, are not copied again at every refresh
/// (`Shadow::copy_stale`).
///
/// A frame asked about view memory some 1,500 times, for nearly all of it
/// to hear that nothing was written, on up to seven threads at once, each
/// asking under the guards' one lock and behind [`ASKING`]'s: 94 ms a frame
/// of all threads' time (run record-1), most of it waiting.
pub(crate) fn light() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIGHT_REFRESH").is_none_or(|v| v != "0"))
}

/// A bit per block of the address space, in pieces made when first needed
/// and never let go.
struct Bits([std::sync::atomic::AtomicPtr<BitsPiece>; Bits::PIECES]);

/// The bits of 4 GiB.
type BitsPiece = [std::sync::atomic::AtomicU64; Bits::WORDS];

impl Bits {
    const WORDS: usize = 1 << 10;
    /// 128 TiB, what a process has.
    const PIECES: usize = 1 << 15;

    const fn new() -> Bits {
        Bits([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; Bits::PIECES])
    }

    fn piece(&self, word: usize) -> Option<&BitsPiece> {
        let piece = self.0.get(word / Bits::WORDS)?.load(std::sync::atomic::Ordering::Acquire);
        // SAFETY: null, or a piece made by `set`, which is never freed.
        unsafe { piece.as_ref() }
    }

    fn set(&self, block: usize) {
        self.put(block);
    }

    /// Sets it; whether it was not set before.
    fn put(&self, block: usize) -> bool {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, SeqCst};
        let word = block / 64;
        let Some(place) = self.0.get(word / Bits::WORDS) else {
            return false;
        };
        if place.load(Acquire).is_null() {
            let piece: Box<BitsPiece> = Box::new([const { std::sync::atomic::AtomicU64::new(0) }; Bits::WORDS]);
            let piece = Box::into_raw(piece);
            if place.compare_exchange(std::ptr::null_mut(), piece, AcqRel, Acquire).is_err() {
                // SAFETY: made just now, and another thread's is there.
                drop(unsafe { Box::from_raw(piece) });
            }
        }
        let bit = 1 << (block % 64);
        self.piece(word).is_some_and(|piece| piece[word % Bits::WORDS].fetch_or(bit, SeqCst) & bit == 0)
    }

    /// Sixty-four of them.
    fn word(&self, word: usize) -> u64 {
        self.piece(word).map_or(0, |piece| piece[word % Bits::WORDS].load(std::sync::atomic::Ordering::SeqCst))
    }

    /// Whether it was set, and is no more.
    fn take(&self, block: usize) -> bool {
        self.any(block, block + 1) && {
            let (word, bit) = (block / 64, 1 << (block % 64));
            self.piece(word)
                .is_some_and(|piece| piece[word % Bits::WORDS].fetch_and(!bit, std::sync::atomic::Ordering::SeqCst) & bit != 0)
        }
    }

    fn clear(&self, block: usize) {
        let word = block / 64;
        if let Some(piece) = self.piece(word) {
            // Ahead of the protection's change, and seen by whoever the
            // writer tells of its write.
            piece[word % Bits::WORDS].fetch_and(!(1 << (block % 64)), std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Whether a bit of `[first, last)` is set.
    fn any(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            match self.piece(word) {
                Some(piece) => {
                    if piece[word % Bits::WORDS].load(std::sync::atomic::Ordering::SeqCst) & mask != 0 {
                        return true;
                    }
                    block = until;
                }
                // None of the piece's.
                None => block = (word / Bits::WORDS + 1) * Bits::WORDS * 64,
            }
        }
        false
    }

    /// Whether the bits of all of `[first, last)` are set (not so of none).
    fn all(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            let Some(piece) = self.piece(word) else {
                return false;
            };
            if piece[word % Bits::WORDS].load(std::sync::atomic::Ordering::Acquire) & mask != mask {
                return false;
            }
            block = until;
        }
        first < last
    }
}

/// The 4 KiB parts of guarded blocks a write fault opened ([`part_open`],
/// [`open_part_loose`]), a bit each, 0 for a block with none; in pieces as
/// [`Bits`]. The rest of such a block is read-only, and it is not clean.
/// `WHOLE`: told as written whole at the next asking, whatever parts are
/// open. Set under the guards' lock, or with none by a fault that holds the
/// block in [`OPENING`]; taken by who makes the block read-only again.
struct Parts([std::sync::atomic::AtomicPtr<PartsPiece>; Bits::PIECES]);

/// The parts of the blocks of 4 GiB.
type PartsPiece = [std::sync::atomic::AtomicU16; Parts::BLOCKS];

impl Parts {
    const BLOCKS: usize = Bits::WORDS * 64;

    const fn new() -> Parts {
        Parts([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; Bits::PIECES])
    }

    fn at(&self, block: usize, make: bool) -> Option<&std::sync::atomic::AtomicU16> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        let place = self.0.get(block / Parts::BLOCKS)?;
        if make && place.load(Acquire).is_null() {
            // Not on the stack: a fault's thread may have little of it.
            let layout = std::alloc::Layout::new::<PartsPiece>();
            // SAFETY: not of size 0; zeros are a piece with no parts open.
            let piece = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<PartsPiece>();
            if piece.is_null() {
                return None;
            }
            if place.compare_exchange(std::ptr::null_mut(), piece, AcqRel, Acquire).is_err() {
                // SAFETY: made just now with this layout, and another
                // thread's is there.
                unsafe { std::alloc::dealloc(piece.cast(), layout) };
            }
        }
        // SAFETY: null, or a piece made above, which is never freed.
        unsafe { place.load(Acquire).as_ref() }.map(|piece| &piece[block % Parts::BLOCKS])
    }

    fn get(&self, block: usize) -> u16 {
        self.at(block, false).map_or(0, |open| open.load(std::sync::atomic::Ordering::SeqCst))
    }

    fn or(&self, block: usize, parts: u16) {
        if let Some(open) = self.at(block, true) {
            open.fetch_or(parts, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// What was open, none now.
    fn take(&self, block: usize) -> u16 {
        self.at(block, false).map_or(0, |open| open.swap(0, std::sync::atomic::Ordering::SeqCst))
    }

    /// All parts, if a part is open.
    fn whole(&self, block: usize) {
        use std::sync::atomic::Ordering::SeqCst;
        if let Some(open) = self.at(block, false) {
            let _ = open.fetch_update(SeqCst, SeqCst, |was| (was != 0).then_some(WHOLE));
        }
    }
}

static PARTED: Parts = Parts::new();

/// The blocks that are read-only by our hand, not written since, and of
/// which the journal holds every write from before that: for those, what
/// [`writes`] answers is in the journal, and neither the kernel, nor the
/// guards' lock, nor [`ASKING`] need be asked.
///
/// Why that is so: a block's bit is set under the guards' lock while the
/// block is clean, by the asking that made it clean, after that asking noted
/// in the journal what it found ([`publish`]). It is cleared under the same
/// lock, before the block becomes writable again ([`open`]) or is forgotten
/// ([`title_memory_remapped`]). So while the bit is set nothing can have
/// written the block through this address, and who sees the bit and looks
/// at the journal after that finds every earlier write in it. A write that
/// comes while the bit is looked at is no different from one that comes just
/// after an asking: the next asking finds the block open.
///
/// So too for the blocks of placeholders that were copied as zeros
/// (`Guards::held`): nothing can be there before the runtime maps it, and
/// it tells of that ([`title_memory_remapped`]) before the title is.
static PUBLISHED: Bits = Bits::new();

/// The blocks read-only by our hand, or about to be: set under the guards'
/// lock ahead of the protection, cleared under it when the block is open
/// again. Who finds none set has no guard to open and need not wait for the
/// lock ([`writing`]): some 700 pieces a frame are landed in memory that is
/// no view.
static PROTECTED: Bits = Bits::new();

/// The blocks the title's bytes were put in beside their guards
/// ([`put_beside`]) and that no asking has told as written since. Such a
/// block is as it was else: read-only, clean, and an asking that finds its
/// bit here takes the bit and tells it as written.
///
/// The bit is set when the bytes are in, at every view of the page, and
/// [`PUBLISHED`]'s is cleared after that. So an asking either finds the bit
/// or came before the title's thread went on from its write; and who takes
/// the bit reads the bytes after.
static BESIDE: Bits = Bits::new();

/// How much was put into a block beside its guard since it was last told as
/// written (`crate::titlewrite::MOST`), by a few bits of its number.
static PUT: [std::sync::atomic::AtomicU32; 1 << 13] = [const { std::sync::atomic::AtomicU32::new(0) }; 1 << 13];

/// The blocks of memory tracked with its views ([`track`]), the descriptor
/// heaps: a write fault in one opens the 4 KiB part it is in, not the block
/// ([`part_open`]).
static FINE: Bits = Bits::new();

/// The guarded blocks a write fault opened with no lock ([`open_loose`])
/// that the guards' notes still have guarded and clean. Who holds the lock
/// and finds a block's bit here goes by the bit, not the notes ([`loose`]).
///
/// The title's threads spent 13-14% of their time in the fault handler,
/// nearly all of it waiting for the guards' lock while askings of whole
/// ranges held it (play-g, 2026-11-26).
static LOOSE: Bits = Bits::new();

/// The blocks a write fault is opening with no lock right now, one thread a
/// block: set before the block's [`LOOSE`] bit, cleared after its
/// [`PROTECTED`] bit.
static OPENING: Bits = Bits::new();

/// The blocks of `Guards::aliased`, for who takes no lock: a write through
/// one is a write of its other views as well, which only [`open`] sees to.
static ALIASED: Bits = Bits::new();

/// Whether write faults of guarded blocks with no other view and no parts
/// open them with no lock (`VICTORY_LOOSE_FAULTS=0`: every fault waits for
/// the guards' lock, as before).
fn loose_faults() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LOOSE_FAULTS").is_none_or(|v| v != "0"))
}

/// Diagnostics: write faults that opened their block with no lock, those
/// that began so and went on to the lock, and blocks so opened that a
/// holder of the lock took into the notes after; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static LOOSE_OPENS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Opens guarded `block` for a write fault in it, with no lock, if it has no
/// other view and no parts: whether it is open. The notes have it guarded
/// and clean until a holder of the lock finds its [`LOOSE`] bit.
fn open_loose(block: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE};
    if !loose_faults() || FINE.any(block, block + 1) || ALIASED.any(block, block + 1) {
        return false;
    }
    if !OPENING.put(block) {
        // Another thread's fault in it: open when that is done, or that one
        // went on to the lock.
        while OPENING.any(block, block + 1) {
            std::thread::yield_now();
        }
        return !PROTECTED.any(block, block + 1);
    }
    let mut opened = false;
    // The bit before the look at `ALIASED`, as `cover` sets that before
    // its look here.
    if PROTECTED.any(block, block + 1) && LOOSE.put(block) {
        if !ALIASED.any(block, block + 1) {
            // Before the block can be written.
            PUBLISHED.clear(block);
            let mut old = 0;
            // SAFETY: a block we made read-only.
            let done = unsafe { protect_as(6, (block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) } != 0;
            opened = done && old == PAGE_READONLY;
            if done && !matches!(old, PAGE_READONLY | PAGE_READWRITE) {
                // Not what we left there: as it was.
                // SAFETY: as above.
                unsafe { protect_as(8, (block * BLOCK) as *const _, BLOCK, old, &mut old) };
            }
        }
        if opened {
            PROTECTED.clear(block);
        } else {
            LOOSE.clear(block);
        }
    }
    OPENING.clear(block);
    #[cfg(feature = "diagnostics")]
    LOOSE_OPENS[usize::from(!opened)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    opened
}

/// For a holder of the guards' lock: whether `block` was opened by a write
/// fault with no lock ([`open_loose`]). One that is open now leaves
/// [`LOOSE`] and is noted open (written, not guarded); one a fault is
/// opening right now stays as noted, and counts as written; one guarded
/// again since it was opened leaves `LOOSE` and is as noted.
fn loose(g: &mut Guards, block: usize) -> bool {
    if !LOOSE.any(block, block + 1) {
        return false;
    }
    if OPENING.any(block, block + 1) {
        return true;
    }
    if PROTECTED.any(block, block + 1) {
        // (Set under the lock only.)
        LOOSE.clear(block);
        return false;
    }
    if LOOSE.take(block) {
        g.protected.remove(block);
        g.clean.remove(block);
        PARTED.take(block);
        g.opened(block);
        #[cfg(feature = "diagnostics")]
        LOOSE_OPENS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    true
}

/// Whether write faults of the descriptor heaps' guarded blocks open their
/// 4 KiB part with no lock (`VICTORY_LOOSE_PARTS=1`; else they wait for the
/// guards' lock, as before).
///
/// Those were the faults left taking the lock, and the title's threads
/// waited behind askings for it: 361 waits, 129 ms of all threads a frame
/// in the intro (2026-11-27).
///
/// `VICTORY_LOOSE_PARTS=2`: those, and askings about heap blocks that only
/// such faults opened ([`guard_parts_loose`]).
fn loose_parts() -> bool {
    loose_level() != 0
}

fn loose_ask() -> bool {
    loose_level() == 2
}

fn loose_level() -> u8 {
    #[cfg(test)]
    if tests::LOOSE_PARTS_HERE.get() != 0 {
        return tests::LOOSE_PARTS_HERE.get();
    }
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("VICTORY_LOOSE_PARTS").as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        _ => 0,
    })
}

/// The blocks a write fault opened a part of with no lock
/// ([`open_part_loose`]) since an asking last took their bit: not clean,
/// whatever the guards' notes say. (With `VICTORY_LOOSE_PARTS=2`, so too
/// those [`part_open`] opened a part of under the lock.)
static LOOSE_PART: Bits = Bits::new();

/// Diagnostics: heap askings answered with no lock, the blocks they told,
/// and heap askings that went on to the lock; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static LOOSE_ASKS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// [`guard_parts`] with no lock, for a range of the descriptor heaps whose
/// blocks are all guarded and known clean, but for parts that write faults
/// opened ([`LOOSE_PART`]): those are made read-only again and told.
/// `None`: ask under the lock.
///
/// Each block is held in [`OPENING`] from the look at its parts to the
/// close, as by the lock's own asking; who opens a heap block, or lets go of
/// its guard, holds it there too, or sets a bit looked at here first.
fn guard_parts_loose(start: usize, end: usize) -> Option<Vec<(usize, u16)>> {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let words = first / 64..last.div_ceil(64);
    let heap = words.clone().all(|word| {
        let these = in_word(word * 64, first, last);
        FINE.word(word) & PROTECTED.word(word) & these == these
    });
    if !heap {
        return None;
    }
    let known = words.clone().all(|word| {
        let these = in_word(word * 64, first, last);
        (ALIASED.word(word) | BESIDE.word(word) | LOOSE.word(word) | PENDING.word(word)) & these == 0
            && (PUBLISHED.word(word) | LOOSE_PART.word(word)) & these == these
    });
    if !known {
        #[cfg(feature = "diagnostics")]
        LOOSE_ASKS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    let mut told = Vec::new();
    for word in words {
        for block in BitsOf(LOOSE_PART.word(word) & in_word(word * 64, first, last)).map(|bit| word * 64 + bit) {
            if !LOOSE_PART.take(block) {
                continue;
            }
            if !OPENING.put(block) {
                // A fault is opening a part: told whole, guarded at the next
                // asking.
                LOOSE_PART.set(block);
                told.push((block, WHOLE));
                continue;
            }
            if !PROTECTED.any(block, block + 1) {
                // Let go of since the look: opened whole.
                OPENING.clear(block);
                told.push((block, WHOLE));
                continue;
            }
            let open = PARTED.take(block);
            // Read-only again before the caller reads the parts.
            let closed = open == 0 || close(block);
            if !closed {
                PARTED.or(block, WHOLE);
                LOOSE_PART.set(block);
            }
            OPENING.clear(block);
            if !closed || open == 0 || BESIDE.any(block, block + 1) {
                told.push((block, WHOLE));
                continue;
            }
            #[cfg(feature = "diagnostics")]
            if open != WHOLE {
                use crate::copystat::{C, add};
                add(C::GuardPartBlocks, 1);
                add(C::GuardPartParts, u64::from(open.count_ones()));
            }
            told.push((block, open));
        }
    }
    #[cfg(feature = "diagnostics")]
    {
        LOOSE_ASKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        LOOSE_ASKS[1].fetch_add(told.len() as u64, std::sync::atomic::Ordering::Relaxed);
    }
    Some(told)
}

/// [`publish`] for an asking [`guard_parts_loose`] answered, with no lock.
fn publish_loose(blocks: &[usize]) {
    for &block in blocks {
        let has = |bits: &Bits| bits.any(block, block + 1);
        if !has(&PROTECTED) {
            continue;
        }
        PUBLISHED.set(block);
        // (Opened, put there or let go of since: whoever did sets its bit, or
        // holds the block in `OPENING`, before it clears this one.)
        if [&BESIDE, &LOOSE, &LOOSE_PART, &PENDING, &OPENING, &ALIASED].into_iter().any(has) || !has(&PROTECTED) {
            PUBLISHED.clear(block);
        }
    }
}

/// Diagnostics: write faults that opened a part with no lock, and those that
/// went on to the lock; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static LOOSE_PARTS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// [`part_open`] with no lock, for a write fault at `at`: whether its part
/// is open. The block is held in [`OPENING`] meanwhile, and so is it by who
/// makes it read-only again, so that no part is opened between that one's
/// look at [`PARTED`] and its `close`.
fn open_part_loose(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    let block = at / BLOCK;
    let most = part_faults();
    if !loose_parts() || most == 0 || !FINE.any(block, block + 1) || ALIASED.any(block, block + 1) {
        return false;
    }
    if !OPENING.put(block) {
        // Another fault's, or an asking closing it.
        while OPENING.any(block, block + 1) {
            std::thread::yield_now();
        }
        return writable(at);
    }
    let part = 1u16 << (at % BLOCK / KERNEL_PAGE);
    let mut opened = false;
    let open = PARTED.get(block);
    if PROTECTED.any(block, block + 1) && (open & part != 0 || open.count_ones() < most) {
        // The bits before the look at `ALIASED`, as `cover_as` sets that
        // before its look at them.
        PARTED.or(block, part);
        LOOSE_PART.set(block);
        if !ALIASED.any(block, block + 1) {
            // Before the part can be written.
            PUBLISHED.clear(block);
            let page = at & !(KERNEL_PAGE - 1);
            let mut old = 0;
            // SAFETY: a block we made read-only; one kernel page of it.
            let done = unsafe { VirtualProtect(page as *const _, KERNEL_PAGE, PAGE_READWRITE, &mut old) } != 0;
            opened = done && matches!(old, PAGE_READONLY | PAGE_READWRITE);
            if done && !opened {
                // SAFETY: as above.
                unsafe { VirtualProtect(page as *const _, KERNEL_PAGE, old, &mut old) };
            }
        }
    }
    OPENING.clear(block);
    #[cfg(feature = "diagnostics")]
    {
        LOOSE_PARTS[usize::from(!opened)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if opened && open & part == 0 {
            crate::copystat::add(crate::copystat::C::GuardPartFaults, 1);
        }
    }
    opened
}

fn put_count(block: usize) -> &'static std::sync::atomic::AtomicU32 {
    &PUT[block % PUT.len()]
}

/// Whether [`put_beside`] put it there.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Beside {
    Done,
    /// The block has had its share.
    Full,
    /// Not guarded, or there is no other way to its page, or the runtime is
    /// changing a mapping.
    Cannot,
}

/// Whether a block of `[at, at + len)` is (or is about to be) read-only by
/// our hand. `len` is not 0.
#[inline]
pub(crate) fn guards_any(at: usize, len: usize) -> bool {
    PROTECTED.any(at / BLOCK, (at.saturating_add(len) - 1) / BLOCK + 1)
}

/// The title's bytes for a guarded block, put in by `put` at our own view of
/// the same page: it is given what to add to an address of the block (as by
/// `wrapping_add`). `weight`: how much of the block's share it is.
pub(crate) fn put_beside(block: usize, weight: u32, put: impl FnOnce(usize)) -> Beside {
    use std::sync::atomic::Ordering::Relaxed;
    if !light() || !view_guard() || !PROTECTED.any(block, block + 1) {
        return Beside::Cannot;
    }
    let count = put_count(block);
    if count.load(Relaxed) + weight > crate::titlewrite::MOST {
        return Beside::Full;
    }
    // No mapping changes until all is told: a view of the page that came
    // after the look at them would not be.
    let Some(_reading) = title_memory_reading_now() else {
        return Beside::Cannot;
    };
    let Some((page, shown)) = crate::inplace::beside_page(block * BLOCK) else {
        return Beside::Cannot;
    };
    put(page.wrapping_sub(block * BLOCK));
    count.fetch_add(weight, Relaxed);
    for view in shown.iter().filter(|&&view| view != 0) {
        BESIDE.set(view / BLOCK);
    }
    for view in shown.iter().filter(|&&view| view != 0) {
        PUBLISHED.clear(view / BLOCK);
    }
    Beside::Done
}

/// A faulting store of the title's leaves `block` as it is: whether it may be
/// left out (until that has been so often, since the block was taken, that
/// the faults cost more than opening it).
pub(crate) fn nothing_put(block: usize) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let count = put_count(block);
    count.load(Relaxed) < crate::titlewrite::MOST && count.fetch_add(1, Relaxed) < crate::titlewrite::MOST
}

/// The title is about to write block by block around `at`: as by a fault.
pub(crate) fn opened_for_writing(at: usize) {
    let mut g = guards();
    if open(&mut g, at, at + 1) != 0 {
        g.opened(at / BLOCK);
    }
}

/// An asking made blocks of `pages` clean and noted them in the journal.
fn publish(pages: &[usize]) {
    let g = guards();
    for &page in pages {
        if g.clean.contains(page) || g.held.contains(page) {
            PUBLISHED.set(page);
            // (Put there or opened by a fault since: whoever did clears the
            // bit after this is set, or its own bit is found here.)
            if BESIDE.any(page, page + 1)
                || LOOSE.any(page, page + 1)
                || LOOSE_PART.any(page, page + 1)
                || PENDING.any(page, page + 1)
            {
                PUBLISHED.clear(page);
            }
        }
    }
}

/// Whether placeholders are known for what they are (`VICTORY_KNOWN_HOLES=0`:
/// every hole is looked at at every asking, as before).
///
/// Most ranges asked about have holes, reservations the title maps its pages
/// into as it streams (231,000 regions, 1,125 of a frame's 1,530 asked view
/// ranges, run next1-1). Such a range was never known whole: each asking put
/// a question to Windows for each of its holes, under the guards' lock, and
/// 111 ms a frame of all threads' time went into waiting for that lock.
fn known_holes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KNOWN_HOLES").is_none_or(|v| v != "0") && light())
}

/// Whether `block` lies in a placeholder: a reservation in which pages come
/// to be only by mapping a view (or an allocation) in its place, which an
/// ordinary commit cannot do.
fn placeholder(block: usize) -> bool {
    /// `MEMORY_REGION_INFORMATION` (ntddk).
    #[repr(C)]
    #[derive(Default)]
    struct Region {
        allocation_base: usize,
        allocation_protect: u32,
        region_type: u32,
        region_size: usize,
        commit_size: usize,
        partition_id: usize,
        node_preference: usize,
    }
    #[link(name = "ntdll", kind = "raw-dylib")]
    unsafe extern "system" {
        fn NtQueryVirtualMemory(
            process: *mut std::ffi::c_void,
            at: *const std::ffi::c_void,
            class: u32,
            out: *mut std::ffi::c_void,
            len: usize,
            returned: *mut usize,
        ) -> i32;
    }
    // (Class 3 answers with the kind of memory in that word, not the flags.)
    const MEMORY_REGION_INFORMATION_EX: u32 = 7;
    const PLACEHOLDER_RESERVATION: u32 = 1 << 8;
    let mut region = Region::default();
    // SAFETY: the answer's room and its size.
    let status = unsafe {
        NtQueryVirtualMemory(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            (block * BLOCK) as *const _,
            MEMORY_REGION_INFORMATION_EX,
            (&raw mut region).cast(),
            size_of::<Region>(),
            std::ptr::null_mut(),
        )
    };
    status >= 0 && region.region_type & PLACEHOLDER_RESERVATION != 0
}

/// The runtime gave the title memory at `[at, at + len)`, allocated or
/// committed: what was known of holes there is asked again.
pub fn title_memory_allocated(at: usize, len: usize) {
    if len == 0 || !known_holes() {
        return;
    }
    let mut g = guards();
    let blocks: Vec<usize> = g.held.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).collect();
    for block in blocks {
        g.held.remove(block);
        PUBLISHED.clear(block);
    }
}

/// Once a frame: a hundredth of the known holes are looked at, in case one
/// was filled with nobody telling ([`title_memory_allocated`],
/// [`title_memory_remapped`]), which no title we know does.
///
/// The nanoseconds it waited for the guards' lock and held it, for the heap
/// pass's counters (`HolesWaitNs`, `HolesHoldNs`).
pub(crate) fn look_at_holes() -> [u64; 2] {
    const PARTS: usize = 100;
    if !known_holes() {
        return [0; 2];
    }
    let mut lock = [0u64; 2];
    static FROM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let blocks: Vec<usize> = {
        let asked = std::time::Instant::now();
        let g = guards();
        let got = std::time::Instant::now();
        let from = FROM.load(std::sync::atomic::Ordering::Relaxed);
        let part: Vec<usize> = g.held.range(from..).take(g.held.len() / PARTS + 1).collect();
        FROM.store(part.last().map_or(0, |b| b + 1), std::sync::atomic::Ordering::Relaxed);
        drop(g);
        lock[0] += (got - asked).as_nanos() as u64;
        lock[1] += got.elapsed().as_nanos() as u64;
        part
    };
    let mut until = 0;
    for block in blocks {
        if block >= until {
            until = uncommitted_end(block);
        }
        if block >= until {
            // (Or unmapped memory was mapped since the part was taken: told
            // already, and forgotten.)
            let asked = std::time::Instant::now();
            let mut g = guards();
            let got = std::time::Instant::now();
            if g.held.remove(block) {
                PUBLISHED.clear(block);
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| {
                    tracing::warn!(
                        address = format_args!("{:#x}", block * BLOCK),
                        "a placeholder of the title's was filled with nobody telling (VICTORY_KNOWN_HOLES=0 asks every time)"
                    );
                });
            }
            drop(g);
            lock[0] += (got - asked).as_nanos() as u64;
            lock[1] += got.elapsed().as_nanos() as u64;
        }
    }
    lock
}

/// Whether a block of `[start, end)` is view memory read-only by our hand:
/// the kernel keeps no written pages of it and would refuse an asking.
pub(crate) fn guarded(start: usize, end: usize) -> bool {
    PROTECTED.any(start / BLOCK, end.div_ceil(BLOCK))
}

/// Whether nothing of `[start, end)` can have been written since an asking
/// noted it in the journal ([`PUBLISHED`]).
fn published(start: usize, end: usize) -> bool {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // (In this order: who puts something in beside a guard sets the one bit,
    // then clears the other, which an asking that is publishing the block may
    // just be setting.)
    light() && PUBLISHED.all(first, last) && !BESIDE.any(first, last)
}

/// Since the last frame line: what asking about title memory was. Kernel
/// askings and the KiB they walked; view ranges answered by their bits, by
/// the guards whole clean, and block by block; blocks made read-only; blocks
/// opened by ourselves; nanoseconds waiting for the guards' lock and
/// nanoseconds of changing protections (all threads); questions to Windows
/// about holes; kernel askings refused; nanoseconds in kernel askings; the
/// 4 KiB pages they found written.
#[cfg(feature = "diagnostics")]
static ASKS: [std::sync::atomic::AtomicU64; 13] = [const { std::sync::atomic::AtomicU64::new(0) }; 13];

#[cfg(feature = "diagnostics")]
fn tally(what: usize, n: u64) {
    ASKS[what].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// [`ASKS`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn asks() -> [u64; 13] {
    std::array::from_fn(|i| ASKS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// Since the last frame line: changes of the protection of title memory by
/// who made them, each calls, 64 KiB blocks, nanoseconds, and calls of
/// 100 us or more. Under the guards' lock: 0 other views of a page made
/// read-only as a copy takes it, 3 blocks with parts open made read-only
/// again, 4 runs and 5 single blocks given write access back, 7 a 4 KiB part
/// opened for a fault, 9 view blocks an asking guards under the lock
/// ([`protect_here`]). With the lock let go: 1 the runs an asking makes
/// read-only (and their blocks one by one when a run fails), 2 other views
/// made read-only after the asking (`VICTORY_ALIAS_LATER`), 6 blocks a write
/// fault opens with no lock. 8 put back as they were (not ours to guard).
#[cfg(feature = "diagnostics")]
static PROTECTS: [[std::sync::atomic::AtomicU64; 4]; 10] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 4] }; 10];

/// `VirtualProtect`, counted in `who`'s row of [`PROTECTS`].
///
/// # Safety
///
/// As `VirtualProtect`.
unsafe fn protect_as(who: usize, at: *const std::ffi::c_void, len: usize, new: u32, old: *mut u32) -> i32 {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the caller's.
    let done = unsafe { windows_sys::Win32::System::Memory::VirtualProtect(at, len, new, old) };
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        let ns = started.elapsed().as_nanos() as u64;
        let row = &PROTECTS[who];
        row[0].fetch_add(1, Relaxed);
        row[1].fetch_add(len.div_ceil(BLOCK) as u64, Relaxed);
        row[2].fetch_add(ns, Relaxed);
        row[3].fetch_add(u64::from(ns >= 100_000), Relaxed);
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = who;
    done
}

/// [`PROTECTS`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn protects() -> [[u64; 4]; 10] {
    std::array::from_fn(|i| std::array::from_fn(|j| PROTECTS[i][j].swap(0, std::sync::atomic::Ordering::Relaxed)))
}

/// The guards' lock, the wait for it counted.
fn guards_timed() -> std::sync::MutexGuard<'static, Guards> {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let g = guards();
    #[cfg(feature = "diagnostics")]
    tally(7, started.elapsed().as_nanos() as u64);
    g
}

/// A set of blocks, a bit each: a fault, or a look at a range of thousands of
/// blocks, is a few words of it and no walk through a tree of every guarded
/// block there is.
struct Blocks(BTreeMap<usize, Box<[u64; Blocks::WORDS]>>);

impl Blocks {
    /// The words of 4 GiB.
    const WORDS: usize = 1 << 10;
    const IN_PIECE: usize = Blocks::WORDS * 64;

    const fn new() -> Blocks {
        Blocks(BTreeMap::new())
    }

    fn word(&self, word: usize) -> u64 {
        self.0.get(&(word / Blocks::WORDS)).map_or(0, |piece| piece[word % Blocks::WORDS])
    }

    fn contains(&self, block: usize) -> bool {
        self.word(block / 64) >> (block % 64) & 1 != 0
    }

    /// Whether it was not there.
    fn insert(&mut self, block: usize) -> bool {
        let piece = self.0.entry(block / Blocks::IN_PIECE).or_insert_with(|| Box::new([0; Blocks::WORDS]));
        let word = &mut piece[block % Blocks::IN_PIECE / 64];
        let was = *word;
        *word |= 1 << (block % 64);
        was != *word
    }

    /// Whether it was there.
    fn remove(&mut self, block: usize) -> bool {
        let Some(piece) = self.0.get_mut(&(block / Blocks::IN_PIECE)) else {
            return false;
        };
        let word = &mut piece[block % Blocks::IN_PIECE / 64];
        let was = *word;
        *word &= !(1 << (block % 64));
        was != *word
    }

    fn insert_all(&mut self, blocks: std::ops::Range<usize>) {
        for block in blocks {
            self.insert(block);
        }
    }

    /// Those of `range`, in order.
    fn range(&self, range: impl RangeBounds<usize>) -> impl Iterator<Item = usize> + '_ {
        let first = match range.start_bound() {
            Bound::Included(&first) => first,
            Bound::Excluded(&before) => before.saturating_add(1),
            Bound::Unbounded => 0,
        };
        // (Exclusive. No block's number is near the top.)
        let last = match range.end_bound() {
            Bound::Included(&last) => last.saturating_add(1),
            Bound::Excluded(&last) => last,
            Bound::Unbounded => usize::MAX,
        }
        .max(first);
        self.0.range(first / Blocks::IN_PIECE..=last / Blocks::IN_PIECE).flat_map(move |(&piece, words)| {
            let base = piece * Blocks::IN_PIECE;
            let from = first.max(base) - base;
            let to = last.min(base + Blocks::IN_PIECE).saturating_sub(base);
            let words = if from < to { &words[from / 64..to.div_ceil(64)] } else { &[][..] };
            words.iter().enumerate().flat_map(move |(k, &bits)| {
                let at = (from / 64 + k) * 64;
                BitsOf(bits & in_word(at, from, to)).map(move |bit| base + at + bit)
            })
        })
    }

    fn len(&self) -> usize {
        self.0.values().flat_map(|piece| piece.iter()).map(|word| word.count_ones() as usize).sum()
    }
}

/// Of the 64 blocks from `at` on, those in `[from, to)`, a bit each.
fn in_word(at: usize, from: usize, to: usize) -> u64 {
    let low = from.saturating_sub(at).min(64);
    let high = to.saturating_sub(at).min(64);
    if low >= high { 0 } else { (u64::MAX >> (64 - (high - low))) << low }
}

/// The bits of a word that are set, by number.
struct BitsOf(u64);

impl Iterator for BitsOf {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        let bit = (self.0 != 0).then(|| self.0.trailing_zeros() as usize)?;
        self.0 &= self.0 - 1;
        Some(bit)
    }
}

/// Whether a write fault opens the view that was written and leaves the
/// page's other views read-only, not clean any more
/// (`VICTORY_OPEN_ONE_VIEW=0`: opens them all, as before).
///
/// The others were opened (a change of protection each, under the guards'
/// lock, with other faults waiting) and made read-only again when the page
/// was next taken, though the title writes through one of them: 586 faults a
/// frame opened 880 blocks (run faults-1). A write through one left as it was
/// faults, and opens it.
fn open_one_view() -> bool {
    crate::toggled!("VICTORY_OPEN_ONE_VIEW")
}

/// How long a block the title has just opened by a write is left open by
/// those that take it (`VICTORY_GUARD_REST=<microseconds>`, 3,000 unless
/// given; `0`: not at all, as before).
///
/// 220 of a frame's 586 faults came within 3 ms after a refresh had taken the
/// block and made it read-only, while the title was still writing it, and
/// what a refresh takes is asked about again by the next within a millisecond
/// (run faults-1): made read-only, written, opened, up to a dozen times a
/// frame. A block that rests counts as written at each asking, so nothing is
/// missed; it is read-only at the first asking after its rest, which the
/// title's later writes are seen by.
fn rest() -> std::time::Duration {
    #[cfg(test)]
    {
        REST_IN_TEST.get()
    }
    #[cfg(not(test))]
    {
        static US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        if !crate::toggled!("VICTORY_GUARD_REST") {
            return std::time::Duration::ZERO;
        }
        let us = *US.get_or_init(|| {
            std::env::var("VICTORY_GUARD_REST").ok().and_then(|v| v.trim().parse().ok()).filter(|us| *us > 1).unwrap_or(3000)
        });
        std::time::Duration::from_micros(us)
    }
}

#[cfg(test)]
thread_local! {
    /// [`rest`], for what this thread asks.
    static REST_IN_TEST: std::cell::Cell<std::time::Duration> = const { std::cell::Cell::new(std::time::Duration::ZERO) };
    /// [`hot`], for this thread's faults.
    static HOT_IN_TEST: std::cell::Cell<std::time::Duration> = const { std::cell::Cell::new(std::time::Duration::ZERO) };
}

/// Held by every test that maps, guards or asks about views of its own: the
/// guards, the views and the askings are global, so tests beside each other
/// see each other's. A view one test maps beside another's guarded block
/// makes that block be told as written again (a range a test expects clean
/// came back with its page), and one test's asking can take a write another
/// waits for. (After TaynanGT/Vice-Tay#246.) A test that runs others as part
/// of it holds it for them: on the thread that holds it, it is not taken
/// again.
#[cfg(test)]
pub(crate) fn one_view_test() -> ViewTest {
    static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if VIEW_TEST_HELD.get() {
        return ViewTest(None);
    }
    let held = ONE.lock().unwrap_or_else(|e| e.into_inner());
    VIEW_TEST_HELD.set(true);
    ViewTest(Some(held))
}

#[cfg(test)]
thread_local! {
    /// Whether this thread holds [`one_view_test`]'s lock.
    static VIEW_TEST_HELD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// [`one_view_test`]'s hold: the lock, unless this thread had it already.
#[cfg(test)]
pub(crate) struct ViewTest(Option<std::sync::MutexGuard<'static, ()>>);

#[cfg(test)]
impl Drop for ViewTest {
    fn drop(&mut self) {
        if self.0.take().is_some() {
            VIEW_TEST_HELD.set(false);
        }
    }
}

/// How long a block the title writes again and again is left open
/// (`VICTORY_GUARD_HOT=<milliseconds>`; off unless given, 50 when only
/// `VICTORY_AB` names it, which turns it on and off): a block that a write
/// fault opens within that time after its last rest ended rests that long.
///
/// Nearly every block a frame makes read-only is written again and faults:
/// 321 to 1,017 blocks made read-only and 323 to 331 guard faults a frame
/// in play (logs coolfox-20261126-155824 and -172435), each fault a change
/// of protection on a title thread and each guarding one on ours, with the
/// guards' lock held. A block that rests counts as written at every asking,
/// as for [`rest`], so nothing is missed: it is copied at each refresh that
/// takes it while it rests, with no fault and no change of protection.
fn hot() -> Option<std::time::Duration> {
    #[cfg(test)]
    {
        let keep = HOT_IN_TEST.get();
        (!keep.is_zero()).then_some(keep)
    }
    #[cfg(not(test))]
    {
        static HOT: std::sync::OnceLock<(std::time::Duration, Option<u32>)> = std::sync::OnceLock::new();
        let &(keep, _bit) = HOT.get_or_init(|| {
            let ms = std::env::var("VICTORY_GUARD_HOT").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|ms| *ms > 0.0);
            #[cfg(feature = "diagnostics")]
            let bit = crate::ab::bit("VICTORY_GUARD_HOT");
            #[cfg(not(feature = "diagnostics"))]
            let bit = None;
            let ms = ms.or(bit.map(|_| 50.0)).unwrap_or(0.0);
            (std::time::Duration::from_secs_f64(ms.min(1e6) / 1000.0), bit)
        });
        #[cfg(feature = "diagnostics")]
        if _bit.is_some_and(|bit| !crate::ab::on(bit)) {
            return None;
        }
        (!keep.is_zero()).then_some(keep)
    }
}

struct Guards {
    /// Blocks we made read-only.
    protected: Blocks,
    /// Of those, the ones not written since a copy took them.
    clean: Blocks,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: Blocks,
    /// Of those, the ones in placeholders ([`known_holes`]).
    held: Blocks,
    /// Blocks of other memory than views in a range asked about as a view:
    /// what was absent and turned out so when something was there again, and
    /// what a question found so before a block was first guarded
    /// ([`protect_here`], [`read_only_blocks`]). Never guarded, copied every
    /// time.
    foreign: BTreeSet<usize>,
    /// Blocks (first, past the last) the kernel writes now
    /// ([`kernel_writing`]), and blocks the runtime has unmapped in a change
    /// not yet done ([`title_memory_unmapping`]). Not in the sets above, and
    /// left as they are by [`guard_parts`], which reports them written.
    kernel: Vec<(usize, usize)>,
    changing: Vec<(usize, usize)>,
    /// Blocks whose page is shown at other addresses too and that were made
    /// read-only together with those (which are in it as well): what opens
    /// one opens them all. (A write's range can be huge, `fread`'s, so only
    /// these are looked up.)
    aliased: Blocks,
    /// Blocks opened by a write fault, when ([`rest`]), and how long they
    /// rest for being written again and again ([`hot`]; zero if not).
    opened: std::collections::HashMap<usize, (std::time::Instant, std::time::Duration), std::hash::BuildHasherDefault<PageHash>>,
}

impl Guards {
    fn left_alone(&self, block: usize) -> bool {
        self.kernel.iter().chain(&self.changing).any(|&(first, last)| (first..last).contains(&block))
    }

    /// Whether `block` is left open for now: the title has just written it.
    fn rests(&self, block: usize, rest: std::time::Duration) -> bool {
        self.opened.get(&block).is_some_and(|&(at, hot)| at.elapsed() < rest.max(hot))
    }

    /// A write fault opened `block`.
    fn opened(&mut self, block: usize) {
        let (at, hot) = (std::time::Instant::now(), hot());
        // (Mapped views rest too with `map_rest`: thousands of blocks a frame.)
        let many: usize = if map_rest() { 1 << 16 } else { 1 << 13 };
        if self.opened.len() >= many {
            self.opened.retain(|_, (at, hot)| at.elapsed() < (*hot).max(std::time::Duration::from_millis(50)));
            if self.opened.len() >= many {
                self.opened.clear();
            }
        }
        put_count(block).store(0, std::sync::atomic::Ordering::Relaxed);
        // Opened again within that time after its last rest: it rests that
        // long now.
        let hot = hot
            .filter(|&keep| {
                self.opened.get(&block).is_some_and(|&(before, rested)| at.saturating_duration_since(before) < rested.max(rest()) + keep)
            })
            .unwrap_or_default();
        self.opened.insert(block, (at, hot));
        #[cfg(feature = "diagnostics")]
        if guard_frames::on() {
            guard_frames::opened(block);
        }
    }
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: Blocks::new(),
    clean: Blocks::new(),
    absent: Blocks::new(),
    held: Blocks::new(),
    foreign: BTreeSet::new(),
    kernel: Vec::new(),
    changing: Vec::new(),
    aliased: Blocks::new(),
    opened: std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::new()),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    if let Ok(g) = GUARDS.try_lock() {
        return g;
    }
    waited(3, || GUARDS.lock().unwrap_or_else(|e| e.into_inner()))
}

/// The 4 KiB pages of title memory that what shaders wrote was put into,
/// and the frame it last was: in pieces made when first needed and never let
/// go. Some 20,000 pages a frame are noted by the relay and the landing
/// threads: in a map behind one lock that was 309 waits for it and 280 looks
/// at its table in the 40 s of a profile, as many as all else of the landings'
/// notes (run faults-5).
struct Landed([std::sync::atomic::AtomicPtr<LandedPiece>; Landed::PIECES]);

/// Of 256 MiB: the frame (its low 31 bits and one), 0 for never.
type LandedPiece = [std::sync::atomic::AtomicU32; Landed::PAGES];

impl Landed {
    const PAGES: usize = 1 << 16;
    /// 128 TiB, what a process has.
    const PIECES: usize = 1 << 19;

    fn stamp(frame: u64) -> u32 {
        (frame as u32 & 0x7fff_ffff) + 1
    }

    fn note(&self, page: usize, frame: u64) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed};
        let Some(place) = self.0.get(page / Landed::PAGES) else {
            return;
        };
        let mut piece = place.load(Acquire);
        if piece.is_null() {
            let made: Box<LandedPiece> = Box::new([const { std::sync::atomic::AtomicU32::new(0) }; Landed::PAGES]);
            let made = Box::into_raw(made);
            piece = match place.compare_exchange(std::ptr::null_mut(), made, AcqRel, Acquire) {
                Ok(_) => made,
                Err(there) => {
                    // SAFETY: made just now, and another thread's is there.
                    drop(unsafe { Box::from_raw(made) });
                    there
                }
            };
        }
        // SAFETY: a piece, which is never freed.
        let at = unsafe { &(&*piece)[page % Landed::PAGES] };
        let stamp = Landed::stamp(frame);
        // (Most are noted several times a frame: no write then.)
        if at.load(Relaxed) != stamp {
            at.store(stamp, Relaxed);
        }
    }

    /// Whether `page` was noted in a frame from `since` on, up to `now`.
    fn since(&self, page: usize, since: u64, now: u64) -> bool {
        let Some(place) = self.0.get(page / Landed::PAGES) else {
            return false;
        };
        // SAFETY: null, or a piece, which is never freed.
        let Some(piece) = (unsafe { place.load(std::sync::atomic::Ordering::Acquire).as_ref() }) else {
            return false;
        };
        let stamp = piece[page % Landed::PAGES].load(std::sync::atomic::Ordering::Relaxed);
        stamp != 0 && u64::from(Landed::stamp(now).wrapping_sub(stamp) & 0x7fff_ffff) <= now.saturating_sub(since)
    }
}

static LANDED: Landed = Landed([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; Landed::PIECES]);

/// A page's number, spread.
#[derive(Default)]
struct PageHash(u64);

impl std::hash::Hasher for PageHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(u64::from(byte));
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0 ^ n).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(26).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

/// What shaders wrote is about to go into title memory at `[at, at + len)`.
pub(crate) fn landing(at: usize, len: usize) {
    {
        let frame = crate::frame::current();
        let (first, last) = (at / KERNEL_PAGE, (at + len.max(1) - 1) / KERNEL_PAGE);
        (first..=last).for_each(|page| LANDED.note(page, frame));
        for (block, others) in elsewhere(at, len) {
            let block = block / KERNEL_PAGE;
            let here = first.max(block)..=last.min(block + BLOCK / KERNEL_PAGE - 1);
            for other in others {
                here.clone().for_each(|page| LANDED.note(page - block + other / KERNEL_PAGE, frame));
            }
        }
    }
    writing(at, len);
}


/// What shaders wrote went into title memory through another view of the
/// same pages (`crate::inplace::beside`): `pages` (of 4 KiB, by number)
/// hold some of it, all of them in `spans` (address, bytes).
///
/// Nothing faulted and no guard was opened. A guarded block stays read-only,
/// clean, and known by its bit ([`PUBLISHED`]), all of which say that the
/// title wrote nothing there: our write is an entry of the journal, which
/// every answer about the block comes from, the quick ones and the guards'
/// alike (a block whose guard is off they report anyway). So neither the
/// guards' lock nor a protection is touched here, or by the next refresh.
/// Noted after the bytes are in, never before: a copy that read the block
/// before they were did so with the number of an asking made before that,
/// which is under the one given here, and takes it again; noted before, a
/// copy could read the older bytes with a number above the note's.
///
/// Returns the note's number, and for each 64 KiB block noted the number of
/// the block's last write before (0: none is known).
pub(crate) fn landed_beside(pages: &[usize], spans: &[(usize, usize)]) -> (u64, Vec<(usize, u64)>) {
    let mut before = Vec::with_capacity(spans.len());
    let noted;
    // The same pages at other addresses: copies of the memory there have not
    // what was put in either.
    let elsewhere: Vec<(usize, Vec<usize>)> = spans.iter().flat_map(|&(at, len)| elsewhere(at, len)).collect();
    {
        let frame = crate::frame::current();
        for &page in pages {
            LANDED.note(page, frame);
        }
        for (block, others) in &elsewhere {
            let first = block / KERNEL_PAGE;
            for &page in pages.iter().filter(|&&page| (first..first + BLOCK / KERNEL_PAGE).contains(&page)) {
                others.iter().for_each(|other| LANDED.note(page - first + other / KERNEL_PAGE, frame));
            }
        }
    }
    {
        let mut blocks: Vec<(usize, u16)> = Vec::with_capacity(spans.len());
        for &page in pages {
            let (block, part) = (page / PARTS, 1u16 << (page % PARTS));
            match blocks.last_mut() {
                Some((last, parts)) if *last == block => *parts |= part,
                _ => blocks.push((block, part)),
            }
        }
        let others = || elsewhere.iter().flat_map(|(_, others)| others.iter().map(|other| other / PAGE));
        if port_journal() {
            // Renderer port #7: shared, the parts of its pages held.
            let j = journal();
            let mut held = j.notes.hold(shards_of_pages(blocks.iter().map(|b| b.0).chain(others())), true);
            let now = next_asking();
            noted = now;
            for (block, parts) in blocks {
                before.push((block, held.note(block, parts, now)));
            }
            for other in others() {
                held.note(other, WHOLE, now);
            }
            port7_count(P7_SHARED_NOTES, 1);
        } else {
            let mut j = journal_mut();
            let now = next_asking();
            noted = now;
            for (block, parts) in blocks {
                before.push((block, j.notes.note(block, parts, now)));
            }
            for other in others() {
                j.notes.note(other, WHOLE, now);
            }
        }
    }
    for &(at, len) in spans {
        // A placed texture there changes with no descriptor written.
        crate::descriptor::host::remapped(at, len);
    }
    for other in elsewhere.into_iter().flat_map(|(_, others)| others) {
        crate::descriptor::host::remapped(other, BLOCK);
    }
    (noted, before)
}

/// Whether every write of the title's to 64 KiB block `block` so far is in
/// the journal ([`PUBLISHED`]).
pub(crate) fn block_published(block: usize) -> bool {
    published(block * BLOCK, (block + 1) * BLOCK)
}

/// Whether shaders' writes went into a page of `[at, at + len)` in the last
/// `frames` frames.
pub(crate) fn landed_lately(at: usize, len: usize, frames: u64) -> bool {
    let now = crate::frame::current();
    (at / 4096..=(at + len.max(1) - 1) / 4096).any(|page| LANDED.since(page, now.saturating_sub(frames), now))
}

/// Whether every block of `[at, at + len)` is view memory written since a
/// host copy last took it: guarded then, and the guard is off. (Written by
/// the title, or by us putting a flight's shader writes into it, if not
/// beside the guard, [`landed_beside`].) Not for
/// blocks no copy ever took, which were never guarded: the caller's range
/// lies in a copy a flight was given, which took all of it. (Nor for blocks
/// read-only by way of another view of their page: whether they were written
/// since is not known.)
pub(crate) fn written_since_taken(at: usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }
    let g = guards();
    (at / BLOCK..=(at + len - 1) / BLOCK)
        .all(|b| {
            LOOSE.any(b, b + 1)
                || PENDING.any(b, b + 1)
                || !g.clean.contains(b) && !g.absent.contains(b) && !g.protected.contains(b)
        })
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
#[cfg(test)]
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    guard_parts(start, end).map(|told| told.into_iter().map(|(block, _)| block).collect())
}

/// [`guard`], with the parts of each block written (a bit for each 4 KiB):
/// those a write fault opened ([`part_open`]), else all of them.
fn guard_parts(start: usize, end: usize) -> Option<Vec<(usize, u16)>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let known = !*ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"))
        && (PROTECTED.any(first, first + 1) || {
            let g = guards_timed();
            g.protected.contains(first) || g.absent.contains(first)
        });
    if !known && !is_view(start, end) {
        return None;
    }
    install();
    // The holes that an ordinary commit may have filled, with no word of the
    // runtime's. Windows is asked with the lock not held (the range is, no
    // other thread asks about it: `ASKING`); what the runtime tells meanwhile
    // takes the block out of the set, and it is looked at as any other.
    let mut g = guards_timed();
    let clean = (first / 64..last.div_ceil(64)).all(|word| {
        let these = in_word(word * 64, first, last);
        g.clean.word(word) & these == these
    });
    if clean
        && !BESIDE.any(first, last)
        && !LOOSE.any(first, last)
        && !LOOSE_PART.any(first, last)
        && !PENDING.any(first, last)
    {
        // Nothing written, mapped or new since the range was last taken.
        #[cfg(feature = "diagnostics")]
        tally(3, 1);
        return Some(Vec::new());
    }
    let unsure: Vec<usize> = g.absent.range(first..last).filter(|block| !g.held.contains(*block)).collect();
    let mut came: Vec<(usize, bool)> = Vec::new();
    // VICTORY_KEEP_GUARD_LOCK=1: with no block to ask Windows about, the lock
    // is kept rather than let go and taken again.
    if !unsure.is_empty() || !crate::titlewrite::asked_for!("VICTORY_KEEP_GUARD_LOCK") {
        drop(g);
        let mut uncommitted_until = first;
        for block in unsure {
            // One question for the blocks of a region.
            if block >= uncommitted_until {
                uncommitted_until = uncommitted_end(block);
                #[cfg(feature = "diagnostics")]
                tally(9, 1);
            }
            if block >= uncommitted_until {
                // The hole may have been given up, and its address be the
                // title's heap's, or code's, now: not ours to guard.
                came.push((block, is_view(block * BLOCK, (block + 1) * BLOCK)));
            }
        }
        g = guards_timed();
    }
    let rest = rest();
    #[cfg(feature = "diagnostics")]
    tally(4, 1);
    for &(block, view) in &came {
        if g.absent.remove(block) && !view {
            g.foreign.insert(block);
        }
    }
    // The range's blocks that are neither clean nor known to be holes. A
    // frame asks about some 10 GB of view memory, 150,000 blocks, nearly all
    // clean.
    let mut written = Vec::new();
    for word in first / 64..last.div_ceil(64) {
        let known = g.clean.word(word)
            & !BESIDE.word(word)
            & !LOOSE.word(word)
            & !LOOSE_PART.word(word)
            & !PENDING.word(word)
            | g.absent.word(word);
        written.extend(BitsOf(!known & in_word(word * 64, first, last)).map(|bit| word * 64 + bit));
    }
    let mut parts = vec![WHOLE; written.len()];
    let mut run: Option<(usize, usize)> = None;
    let mut later = Vec::new();
    let mut aliases = Aliases::default();
    for (i, &block) in written.iter().enumerate() {
        if let Some((from, to)) = run
            && to != block
        {
            pend(&mut g, from, to, &mut later);
            run = None;
            if !protect_later() {
                // Faults of the title's wait for the lock: each block is done
                // on its own, and looked at anew below.
                drop(g);
                g = guards_timed();
            }
        }
        // Told as written by this, and read by the caller after.
        let beside = BESIDE.take(block);
        if beside {
            put_count(block).store(0, std::sync::atomic::Ordering::Relaxed);
        }
        if LOOSE_PART.take(block) {
            g.clean.remove(block);
        }
        if loose(&mut g, block) && g.protected.contains(block) {
            // A fault is opening it right now: guarded at the next asking.
            if let Some((from, to)) = run.take() {
                pend(&mut g, from, to, &mut later);
            }
            continue;
        }
        if g.protected.contains(block) {
            // Read-only by way of another view of its page, or not clean
            // because one could not be made so. (Or clean, and bytes were put
            // in beside the guard: it stays.)
            if let Some((from, to)) = run.take() {
                pend(&mut g, from, to, &mut later);
            }
            if g.clean.contains(block) {
                continue;
            }
            if alias_later() && PENDING.any(block, block + 1) {
                // Another asking's: another view of it is not read-only yet.
                continue;
            }
            // No part opened with no lock between the look and the close.
            let claimed = loose_parts() && FINE.any(block, block + 1);
            if claimed && !OPENING.put(block) {
                // A fault is opening a part: told whole, guarded at the next
                // asking.
                LOOSE_PART.set(block);
                continue;
            }
            let open = PARTED.take(block);
            // Read-only again before the caller reads the parts.
            let closed = open == 0 || close(block);
            if !closed {
                PARTED.or(block, WHOLE);
            }
            if claimed {
                OPENING.clear(block);
            }
            if !closed {
                continue;
            }
            if open != 0 && !beside {
                parts[i] = open;
            }
            #[cfg(feature = "diagnostics")]
            if open != 0 && open != WHOLE {
                use crate::copystat::{C, add};
                add(C::GuardPartBlocks, 1);
                add(C::GuardPartParts, u64::from(open.count_ones()));
            }
            if !alias_later() {
                if cover(&mut g, block, block + 1) {
                    g.clean.insert(block);
                }
                continue;
            }
            let relied = aliases.relies.len();
            if cover_as(&mut g, block, block + 1, Some(&mut aliases)) {
                if aliases.relies.len() == relied {
                    g.clean.insert(block);
                } else {
                    aliases.clean.push(block);
                }
            }
        } else if !g.foreign.contains(&block)
            && !g.left_alone(block)
            && !g.rests(block, rest)
            // Opened since another asking's run took it, which is not done
            // with it: written still, and guarded at the next asking.
            && !PENDING.any(block, block + 1)
        {
            run = Some((run.map_or(block, |r| r.0), block + 1));
        } else if let Some((from, to)) = run.take() {
            pend(&mut g, from, to, &mut later);
        }
    }
    if let Some((from, to)) = run {
        pend(&mut g, from, to, &mut later);
    }
    drop(g);
    if !later.is_empty() || !aliases.blocks.is_empty() {
        let fared: Vec<Fared> = later
            .iter()
            .map(|&(from, to)| {
                if all_views(from, to) && read_only(from, to, 1) {
                    Fared::Whole
                } else if fallback_later() {
                    Fared::Blocks(read_only_blocks(from, to))
                } else {
                    Fared::Not
                }
            })
            .collect();
        let mut g = guards_timed();
        for (&(from, to), fared) in later.iter().zip(fared) {
            protected_later_as(&mut g, from, to, fared, alias_later().then_some(&mut aliases));
        }
        if !aliases.blocks.is_empty() {
            drop(g);
            let done: Vec<bool> = aliases.blocks.iter().map(|&block| read_only(block, block + 1, 2)).collect();
            aliases_done(&mut guards_timed(), &aliases, done);
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::faultbook::note(crate::faultbook::What::Asked, first, last, written.len() as u64);
    Some(written.into_iter().zip(parts).collect())
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: &mut Guards, from: usize, to: usize) {
    #[cfg(feature = "diagnostics")]
    taken(from, to);
    protect_here(g, from, to);
    cover(g, from, to);
}

#[cfg(feature = "diagnostics")]
fn taken(from: usize, to: usize) {
    crate::faultbook::note(crate::faultbook::What::Taken, from, to, 0);
    let again = (from..to).filter(|&block| crate::copystat::again(4, (block, 0, 0))).count();
    crate::copystat::add(crate::copystat::C::GuardAgain, again as u64);
    if guard_frames::on() {
        guard_frames::guarded(from, to);
    }
}

/// Write watch askings cut at reservation ends (`VICTORY_ASK_SPANS=1`), and
/// the counts of its log line, kept with the switch off too.
///
/// `ask_ahead` joined a submission's ranges into runs without knowing where
/// one reservation ends: a run across two, or holding a view, was refused
/// whole and asked again by halves, each half paying the 30-70 microseconds
/// of a call again (leafy-play: 205 askings a frame, 115 of them refused).
/// With the switch on a run stays in one reservation (VirtualQuery,
/// remembered for the frame, as the title maps and unmaps between frames),
/// views are not asked (they go to the guards, as a refusal sends them), a
/// reservation the kernel refused is not asked again that frame, and
/// nothing is split.
pub(crate) mod ask_spans {
    use std::cell::Cell;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::{Mutex, MutexGuard, OnceLock};

    use super::PAGE;

    pub(crate) fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_SPANS").is_some_and(|v| v == "1"))
    }

    const ANSWERED: usize = 0;
    const REFUSED_ACROSS: usize = 1;
    const REFUSED_ONE: usize = 2;
    const REFUSED_VIEW: usize = 3;
    const NOT_ACROSS: usize = 4;
    const NOT_VIEW: usize = 5;
    const NOT_REFUSED: usize = 6;
    const SPLITS: usize = 7;
    const SUBMISSIONS: usize = 8;
    const RESERVATIONS: usize = 9;
    const ASKINGS: usize = 10;
    const ASKED: usize = 11;
    const RANGES: usize = 12;
    const QUERIES: usize = 13;
    static COUNTS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

    /// This frame's regions by start (end, reservation base, view), and the
    /// reservations the kernel refused.
    struct Frame {
        frame: u64,
        regions: BTreeMap<usize, (usize, usize, bool)>,
        refused: BTreeSet<usize>,
    }
    static FRAME: Mutex<Frame> = Mutex::new(Frame { frame: 0, regions: BTreeMap::new(), refused: BTreeSet::new() });

    thread_local! {
        /// Kernel askings made on this thread, and the bytes they covered.
        static CALLS: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
    }

    fn frame() -> MutexGuard<'static, Frame> {
        let now = crate::frame::current();
        let mut f = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if f.frame != now {
            f.frame = now;
            f.regions.clear();
            f.refused.clear();
        }
        f
    }

    /// The region holding `at`: start, end, reservation base, view.
    fn region(f: &mut Frame, at: usize) -> Option<(usize, usize, usize, bool)> {
        if let Some((&start, &(end, base, view))) = f.regions.range(..=at).next_back()
            && at < end
        {
            return Some((start, end, base, view));
        }
        use windows_sys::Win32::System::Memory::{MEM_FREE, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        COUNTS[QUERIES].fetch_add(1, Relaxed);
        if n == 0 || info.State == MEM_FREE {
            return None;
        }
        let start = info.BaseAddress as usize;
        let end = start.saturating_add(info.RegionSize);
        let base = info.AllocationBase as usize;
        let view = info.Type == MEM_MAPPED;
        f.regions.insert(start, (end, base, view));
        Some((start, end, base, view))
    }

    /// The reservation pages `[first, last)` lie in, by its base, and
    /// whether any of them is a view; `None` across reservations or into
    /// free memory.
    fn reservation(f: &mut Frame, first: usize, last: usize) -> Option<(usize, bool)> {
        let end = last * PAGE;
        let (_, mut to, base, mut view) = region(f, first * PAGE)?;
        while to < end {
            let (_, next, b, v) = region(f, to)?;
            if b != base {
                return None;
            }
            view |= v;
            to = next;
        }
        Some((base, view))
    }

    /// Whether pages `[first, last)` are left unasked: across reservations,
    /// holding a view, or in a reservation refused this frame.
    pub(super) fn skip(first: usize, last: usize) -> bool {
        let mut f = frame();
        let why = match reservation(&mut f, first, last) {
            None => NOT_ACROSS,
            Some((_, true)) => NOT_VIEW,
            Some((base, false)) if f.refused.contains(&base) => NOT_REFUSED,
            Some(_) => return false,
        };
        COUNTS[why].fetch_add(1, Relaxed);
        true
    }

    /// The kernel answered or refused an asking of pages `[first, last)`.
    pub(super) fn told(first: usize, last: usize, failed: bool) {
        CALLS.with(|c| {
            let (n, bytes) = c.get();
            c.set((n + 1, bytes + ((last - first) * PAGE) as u64));
        });
        if !failed {
            COUNTS[ANSWERED].fetch_add(1, Relaxed);
            return;
        }
        let mut f = frame();
        let slot = match reservation(&mut f, first, last) {
            None => REFUSED_ACROSS,
            Some((_, true)) => REFUSED_VIEW,
            Some((base, false)) => {
                if on() {
                    f.refused.insert(base);
                }
                REFUSED_ONE
            }
        };
        COUNTS[slot].fetch_add(1, Relaxed);
    }

    /// An asking made by splitting a refused run.
    pub(super) fn split() {
        COUNTS[SPLITS].fetch_add(1, Relaxed);
    }

    /// Keeps the pieces (page ranges in order) that lie in one reservation,
    /// holding no view, not refused this frame, and gives each one's base.
    pub(super) fn bases(pieces: &mut Vec<(usize, usize)>) -> Vec<usize> {
        let mut f = frame();
        let mut bases = Vec::with_capacity(pieces.len());
        pieces.retain(|&(first, last)| {
            let why = match reservation(&mut f, first, last) {
                None => NOT_ACROSS,
                Some((_, true)) => NOT_VIEW,
                Some((base, false)) if f.refused.contains(&base) => NOT_REFUSED,
                Some((base, false)) => {
                    bases.push(base);
                    return true;
                }
            };
            COUNTS[why].fetch_add(1, Relaxed);
            false
        });
        let distinct: BTreeSet<usize> = bases.iter().copied().collect();
        COUNTS[RESERVATIONS].fetch_add(distinct.len() as u64, Relaxed);
        bases
    }

    /// This thread's askings so far, for [`submitted`].
    pub(super) fn mark() -> (u64, u64) {
        CALLS.with(Cell::get)
    }

    /// A submission's asking ([`super::ask_once`]) is done: its askings,
    /// the bytes they covered, and the bytes of its ranges themselves.
    pub(super) fn submitted(ranges: &[(usize, usize)], before: (u64, u64)) {
        let (n, bytes) = mark();
        let mut pages: Vec<(usize, usize)> = ranges
            .iter()
            .filter(|r| r.0 < r.1)
            .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
            .collect();
        pages.sort_unstable();
        let (mut sum, mut to) = (0, 0);
        for (first, last) in pages {
            let first = first.max(to);
            if last > first {
                sum += last - first;
                to = last;
            }
        }
        COUNTS[SUBMISSIONS].fetch_add(1, Relaxed);
        COUNTS[ASKINGS].fetch_add(n - before.0, Relaxed);
        COUNTS[ASKED].fetch_add(bytes - before.1, Relaxed);
        COUNTS[RANGES].fetch_add((sum * PAGE) as u64, Relaxed);
    }

    pub(crate) fn line(frames: f64) -> String {
        let c: [f64; 14] = std::array::from_fn(|i| COUNTS[i].swap(0, Relaxed) as f64 / frames);
        let each = |v: f64| if c[SUBMISSIONS] > 0.0 { v / c[SUBMISSIONS] } else { 0.0 };
        format!(
            "{:.0} answered, {:.0} refused across reservations, {:.0} refused in one reservation, {:.0} refused on views; \
             not asked: {:.0} across reservations, {:.0} on views, {:.0} in reservations refused before; \
             {:.0} askings made by splitting; {:.1} submissions asked ahead, {:.1} reservations each (switch on only), {:.1} askings each; \
             {:.0} MB asked over ranges of {:.0} MB; {:.0} VirtualQuery calls",
            c[ANSWERED],
            c[REFUSED_ACROSS],
            c[REFUSED_ONE],
            c[REFUSED_VIEW],
            c[NOT_ACROSS],
            c[NOT_VIEW],
            c[NOT_REFUSED],
            c[SPLITS],
            c[SUBMISSIONS],
            each(c[RESERVATIONS]),
            each(c[ASKINGS]),
            c[ASKED] / 1048576.0,
            c[RANGES] / 1048576.0,
            c[QUERIES]
        )
    }
}

/// `VICTORY_GUARD_FRAMES=1`, for the log: whether the blocks made read-only
/// are the same frame after frame, and how many guard faults land on those.
/// In Leafy's run (leafy-0314) 400 blocks a frame were made read-only, with
/// 282 guard faults and 264 waits for the guards' lock (91 ms) over all
/// threads. A block guarded and written every frame costs a fault and two
/// protection changes each time for nothing.
#[cfg(feature = "diagnostics")]
pub(crate) mod guard_frames {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::{Mutex, OnceLock};

    /// Frames running from which a block counts as guarded every frame.
    const RUNNING: u32 = 8;

    pub(crate) fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_FRAMES").is_some_and(|v| v == "1"))
    }

    /// Per block: the last frame it was made read-only, the frames running
    /// it was, and the guard faults on it in that run.
    type Seen = HashMap<usize, (u64, u32, u32), std::hash::BuildHasherDefault<super::PageHash>>;
    static SEEN: Mutex<Option<Seen>> = Mutex::new(None);

    /// Blocks made read-only; of them, made read-only the frame before too;
    /// [`RUNNING`] frames running or more; guard faults; of them, on blocks
    /// made read-only the frame before or this one; on blocks [`RUNNING`]
    /// frames running.
    static COUNTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

    pub(super) fn guarded(from: usize, to: usize) {
        let frame = crate::frame::current();
        let (mut before, mut running) = (0, 0);
        {
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            let seen = seen.get_or_insert_with(Default::default);
            if seen.len() >= 1 << 20 {
                seen.retain(|_, &mut (at, _, _)| at + 2 >= frame);
            }
            for block in from..to {
                let e = seen.entry(block).or_insert((0, 0, 0));
                if e.0 == frame && e.1 > 0 {
                    continue;
                }
                if e.1 > 0 && e.0 + 1 == frame {
                    e.1 += 1;
                    before += 1;
                } else {
                    *e = (0, 1, 0);
                }
                e.0 = frame;
                if e.1 >= RUNNING {
                    running += 1;
                }
            }
        }
        COUNTS[0].fetch_add((to - from) as u64, Relaxed);
        COUNTS[1].fetch_add(before, Relaxed);
        COUNTS[2].fetch_add(running, Relaxed);
    }

    /// A guard fault on `block`.
    pub(super) fn opened(block: usize) {
        let frame = crate::frame::current();
        COUNTS[3].fetch_add(1, Relaxed);
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let Some(e) = seen.as_mut().and_then(|seen| seen.get_mut(&block)) else {
            return;
        };
        if e.0 + 1 >= frame {
            COUNTS[4].fetch_add(1, Relaxed);
            e.2 += 1;
            if e.1 >= RUNNING {
                COUNTS[5].fetch_add(1, Relaxed);
            }
        }
    }

    /// The counts per frame, and the longest runs of neighbouring blocks
    /// made read-only every frame: start, length, frames running (the least
    /// of the run), guard faults per frame in it.
    pub(crate) fn line(frames: f64) -> String {
        let c = [0, 1, 2, 3, 4, 5].map(|i| COUNTS[i].swap(0, Relaxed) as f64 / frames);
        let frame = crate::frame::current();
        let mut blocks: Vec<(usize, u32, u32)> = {
            let seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            seen.iter()
                .flatten()
                .filter(|&(_, &(at, run, _))| at + 1 >= frame && run >= RUNNING)
                .map(|(&block, &(_, run, faults))| (block, run, faults))
                .collect()
        };
        blocks.sort_unstable();
        let mut runs: Vec<(usize, usize, u32, f64)> = Vec::new();
        for (block, run, faults) in blocks {
            match runs.last_mut() {
                Some(r) if r.0 + r.1 == block => {
                    r.1 += 1;
                    r.2 = r.2.min(run);
                    r.3 += f64::from(faults) / f64::from(run);
                }
                _ => runs.push((block, 1, run, f64::from(faults) / f64::from(run))),
            }
        }
        let every: usize = runs.iter().map(|r| r.1).sum();
        runs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        let top: Vec<String> = runs
            .iter()
            .take(8)
            .map(|&(block, n, run, faults)| {
                format!("{:#x}+{:#x} ({run}+ frames, {faults:.1} faults a frame)", block * super::BLOCK, n * super::BLOCK)
            })
            .collect();
        format!(
            "{:.0} blocks made read-only, {:.0} of them the frame before too, {:.0} {RUNNING}+ frames running; \
             {:.0} guard faults, {:.0} on blocks made read-only that frame or the one before, {:.0} on blocks {RUNNING}+ frames running; \
             {every} blocks made read-only every frame now, in {} runs, the longest: {}",
            c[0],
            c[1],
            c[2],
            c[3],
            c[4],
            c[5],
            runs.len(),
            top.join(", ")
        )
    }
}

/// Blocks [`guard_parts`] has noted guarded and clean but not made
/// read-only yet ([`pend`]). Another asking tells them as written: a write
/// can still land there unseen until then, and it is that asking's caller
/// that reads it. One opened meanwhile can be read-only with no note
/// until [`protected_later`] opens it again: a write fault there waits for
/// that, and so does [`kernel_writing`].
static PENDING: Bits = Bits::new();

/// Whether [`guard_parts`] makes its blocks read-only with the guards' lock
/// let go (`VICTORY_PROTECT_LATER=0`: under it, as before).
///
/// The lock was held 105-149 ms a frame over all threads, much of it over
/// the `VirtualProtect` calls of these askings (play-g, 2026-11-26).
fn protect_later() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PROTECT_LATER").is_none_or(|v| v != "0"))
}

/// Whether a run of [`pend`]'s that [`read_only`] could not make read-only
/// in one call (it spans views) is done block by block with the lock let
/// go too (`VICTORY_GUARD_FALLBACK_LATER=1`; off: under the lock, as
/// before).
///
/// 32% of these runs failed, some 210 `VirtualProtect` calls a frame made
/// under the lock after, 11.5 us each (a5c442eec).
fn fallback_later() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GUARD_FALLBACK_LATER"))
}

/// Whether [`guard_parts`] makes the other views of its pages read-only
/// with the lock let go too (`VICTORY_ALIAS_LATER=1`; off: one
/// `VirtualProtect` each under the lock, [`protect_alias`], as before).
///
/// Some 1000 such calls a frame were made under the lock (a5c442eec).
fn alias_later() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_ALIAS_LATER"))
}

/// Whether the blocks a mapped view opens rest as those a write fault
/// opened do ([`Guards::rests`], `VICTORY_MAP_REST=1`; off: guarded again at
/// the next asking, as before).
///
/// The spikes follow `views_mapped` storms: about 1967 blocks made
/// read-only again a frame, most of them opened by a mapping just before
/// (a5c442eec).
fn map_rest() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_MAP_REST"))
}

/// Diagnostics: runs of guarded blocks made read-only with the guards' lock
/// let go ([`pend`]), the blocks in them, blocks opened meanwhile and made
/// writable again, and runs that failed and were guarded under the lock
/// after; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static LATER: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Graphics blocks of the title's allocator that got write-watched memory of
/// their own (`XMemAllocDefault`), since the start: blocks, bytes, and
/// blocks given back.
pub(crate) static GRAPHICS_BLOCKS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Told by the title's allocator of a graphics block given write-watched
/// memory of its own, `len` bytes, or of one given back.
pub fn title_graphics_block(len: usize, freed: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if freed {
        GRAPHICS_BLOCKS[2].fetch_add(1, Relaxed);
    } else {
        GRAPHICS_BLOCKS[0].fetch_add(1, Relaxed);
        GRAPHICS_BLOCKS[1].fetch_add(len as u64, Relaxed);
    }
}

/// [`protect`] for [`guard_parts`], with the `VirtualProtect` left for when
/// the lock is let go ([`protect_later`]): `[from, to)` is noted guarded,
/// clean and [`PENDING`] now, and goes in `later` for [`read_only`] and
/// [`protected_later`].
fn pend(g: &mut Guards, from: usize, to: usize, later: &mut Vec<(usize, usize)>) {
    if !protect_later() {
        protect(g, from, to);
        return;
    }
    #[cfg(feature = "diagnostics")]
    taken(from, to);
    for block in from..to {
        // Ahead of the protection, as in `protect_here`.
        PROTECTED.set(block);
        PENDING.set(block);
    }
    g.protected.insert_all(from..to);
    g.clean.insert_all(from..to);
    later.push((from, to));
}

/// How [`read_only`] fared with a run of [`pend`]'s.
enum Fared {
    Whole,
    /// Failed, nothing changed: guarded block by block under the lock.
    Not,
    /// Failed, and done block by block with the lock let go
    /// ([`fallback_later`]).
    Blocks(Vec<Took>),
}

/// How [`read_only_blocks`] fared with one block.
enum Took {
    ReadOnly,
    /// Not committed (unmapped); `held` if a placeholder holds it.
    Absent { held: bool },
    /// Other memory than a view ([`There::Other`]): left as it is.
    Foreign,
    Not,
}

/// Whether all of `[from, to)` is view memory, by what Windows says of each
/// region ([`there`]): only then is a run of [`pend`]'s made read-only in one
/// call. Otherwise it goes region by region ([`read_only_blocks`], or
/// [`protect_here`] under the lock).
fn all_views(from: usize, to: usize) -> bool {
    let mut block = from;
    while block < to {
        let (what, whole, ..) = there(block);
        if what != There::View {
            return false;
        }
        block = whole.max(block + 1);
    }
    true
}

/// [`protect_here`] for a run, with no notes: those are made under the lock
/// after ([`protected_later_as`]). Only view memory is made read-only, as
/// there.
fn read_only_blocks(from: usize, to: usize) -> Vec<Took> {
    let mut took = Vec::with_capacity(to - from);
    let mut block = from;
    while block < to {
        let (what, whole, kind, protection) = there(block);
        let whole = whole.min(to);
        let until = whole.max(block + 1);
        match what {
            There::View if read_only(block, until, 1) => took.extend((block..until).map(|_| Took::ReadOnly)),
            There::View => took.extend(
                (block..until).map(|block| if read_only(block, block + 1, 1) { Took::ReadOnly } else { Took::Not }),
            ),
            There::Hole => {
                took.extend((block..whole).map(|block| Took::Absent { held: known_holes() && placeholder(block) }));
                took.extend((whole..until).map(|_| Took::Not));
                #[cfg(feature = "diagnostics")]
                tally(9, 1);
            }
            There::Other => {
                took.extend((block..until).map(|_| Took::Foreign));
                not_a_view(block, until, kind, protection);
            }
        }
        block = until;
    }
    took
}

/// After [`read_only`] of [`pend`]'s run `[from, to)`, under the lock again.
/// A block [`open`] opened in between (notes gone, `PROTECTED` cleared) is
/// made writable again; a run that failed (`!done`, nothing changed) is
/// guarded block by block as before. Then the other views of its pages.
#[cfg(test)]
fn protected_later(g: &mut Guards, from: usize, to: usize, done: bool) {
    protected_later_as(g, from, to, if done { Fared::Whole } else { Fared::Not }, None);
}

/// [`protected_later`], with the run done block by block already if it
/// failed ([`Fared::Blocks`]), and the other views of its pages left in
/// `plan` to make read-only with the lock let go ([`alias_later`]).
fn protected_later_as(g: &mut Guards, from: usize, to: usize, fared: Fared, mut plan: Option<&mut Aliases>) {
    for block in from..to {
        PENDING.clear(block);
    }
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        LATER[0].fetch_add(1, Relaxed);
        LATER[1].fetch_add((to - from) as u64, Relaxed);
    }
    let done = match fared {
        Fared::Whole => true,
        Fared::Not => false,
        Fared::Blocks(took) => {
            for (block, took) in (from..to).zip(took) {
                let ours = PROTECTED.any(block, block + 1) && g.protected.contains(block);
                match took {
                    Took::ReadOnly => {
                        // (As below, for a run done whole.)
                        if !PROTECTED.any(block, block + 1) && !g.protected.contains(block) {
                            open_block(block);
                            #[cfg(feature = "diagnostics")]
                            LATER[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    // Those opened or let go meanwhile stay as that left them.
                    _ if !ours => {}
                    took => {
                        g.protected.remove(block);
                        g.clean.remove(block);
                        PROTECTED.clear(block);
                        match took {
                            Took::Absent { held } => {
                                g.absent.insert(block);
                                if held {
                                    g.held.insert(block);
                                }
                            }
                            Took::Foreign => {
                                g.foreign.insert(block);
                            }
                            _ => {}
                        }
                    }
                }
            }
            cover_as(g, from, to, plan);
            #[cfg(feature = "diagnostics")]
            LATER[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return;
        }
    };
    if !done {
        // Those opened or let go meanwhile stay as that left them (a range
        // the kernel writes now among them).
        let ours: Vec<usize> = (from..to).filter(|&b| PROTECTED.any(b, b + 1) && g.protected.contains(b)).collect();
        for run in ours.chunk_by(|a, b| a + 1 == *b) {
            let (from, to) = (run[0], run[run.len() - 1] + 1);
            for block in from..to {
                g.protected.remove(block);
                g.clean.remove(block);
                PROTECTED.clear(block);
            }
            protect_here(g, from, to);
            cover_as(g, from, to, plan.as_deref_mut());
        }
        #[cfg(feature = "diagnostics")]
        LATER[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return;
    }
    for block in from..to {
        // (One a fault opened with no lock is still in the notes.)
        if !PROTECTED.any(block, block + 1) && !g.protected.contains(block) {
            open_block(block);
            #[cfg(feature = "diagnostics")]
            LATER[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    cover_as(g, from, to, plan);
}

/// Other views of guarded pages left by [`cover_as`] for [`guard_parts`] to
/// make read-only with the lock let go ([`alias_later`]), and
/// [`aliases_done`] to note after. Each is noted guarded and [`PENDING`]
/// meanwhile, and so is each block that relies on one (in `relies`): who
/// asks about those tells them as written until then.
#[derive(Default)]
struct Aliases {
    blocks: Vec<usize>,
    planned: std::collections::HashSet<usize>,
    /// (block, other view of its page)
    relies: Vec<(usize, usize)>,
    /// Blocks to note clean if all they rely on is made read-only.
    clean: Vec<usize>,
}

/// Diagnostics: other views of guarded pages made read-only with the
/// guards' lock let go ([`alias_later`]), and those that failed or were
/// opened meanwhile; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIAS_LATER: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// After [`read_only`] of each of `aliases.blocks` (`done`), under the lock
/// again. One opened in between is made writable again, as in
/// [`protected_later`]; one that failed is not guarded. A block that relied
/// on either is not clean.
fn aliases_done(g: &mut Guards, aliases: &Aliases, done: Vec<bool>) {
    let mut failed = std::collections::HashSet::new();
    for (&block, done) in aliases.blocks.iter().zip(done) {
        PENDING.clear(block);
        let ours = PROTECTED.any(block, block + 1) && g.protected.contains(block);
        if done && ours {
            #[cfg(feature = "diagnostics")]
            crate::faultbook::note(crate::faultbook::What::Alias, block, 0, 0);
            continue;
        }
        if done {
            // (One a fault opened with no lock is still in the notes.)
            if !PROTECTED.any(block, block + 1) && !g.protected.contains(block) {
                open_block(block);
            }
        } else if ours {
            g.protected.remove(block);
            PROTECTED.clear(block);
        }
        failed.insert(block);
    }
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        ALIAS_LATER[0].fetch_add(aliases.blocks.len() as u64, Relaxed);
        ALIAS_LATER[1].fetch_add(failed.len() as u64, Relaxed);
    }
    let mut short = std::collections::HashSet::new();
    for &(block, other) in &aliases.relies {
        PENDING.clear(block);
        if failed.contains(&other) {
            short.insert(block);
            g.clean.remove(block);
        }
    }
    for &block in &aliases.clean {
        if !short.contains(&block)
            && g.protected.contains(block)
            && PROTECTED.any(block, block + 1)
            && !LOOSE.any(block, block + 1)
            && !LOOSE_PART.any(block, block + 1)
        {
            g.clean.insert(block);
        }
    }
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Diagnostics: runs of guarded blocks opened with one call ([`open_run`]),
/// the blocks in them, and blocks opened one at a time; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static OPENS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Of the blocks `[from, to)`, which are read-only by our hand where they
/// can be: every other view of their pages is made read-only too. Where one
/// cannot be (the kernel writes through it now, or the runtime is changing
/// it), a write through it would not be seen: that block is not clean, and
/// told as written at every asking, until it can. Whether all are covered.
fn cover(g: &mut Guards, from: usize, to: usize) -> bool {
    cover_as(g, from, to, None)
}

/// [`cover`], with the views to make read-only left in `plan`
/// ([`alias_later`]): they count as covered, and the block is
/// [`PENDING`] until they are done.
fn cover_as(g: &mut Guards, from: usize, to: usize, mut plan: Option<&mut Aliases>) -> bool {
    let mut all = true;
    for (page, others) in elsewhere(from * BLOCK, (to - from) * BLOCK) {
        let block = page / BLOCK;
        if !g.protected.contains(block) {
            continue; // not there, or not ours to guard
        }
        #[cfg(feature = "diagnostics")]
        ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        g.aliased.insert(block);
        ALIASED.set(block);
        if loose(g, block)
            || loose_parts()
                && FINE.any(block, block + 1)
                && (LOOSE_PART.any(block, block + 1) || OPENING.any(block, block + 1))
        {
            // Opened by a fault since it was guarded.
            all = false;
            g.clean.remove(block);
            continue;
        }
        let mut covered = true;
        let relied = plan.as_ref().map_or(0, |a| a.relies.len());
        for other in others {
            covered &= protect_alias(g, other / BLOCK, plan.as_deref_mut().map(|a| (a, block)));
        }
        if plan.as_ref().is_some_and(|a| a.relies.len() != relied) {
            PENDING.set(block);
        }
        if !covered {
            all = false;
            g.clean.remove(block);
        }
    }
    all
}

/// Makes `block` read-only: another view of a page a copy is about to take
/// through its own, so that a write through this one faults too. It is not
/// noted clean: the copies of title memory at this view have not taken the
/// page ([`guard_parts`] tells such a block as written, then notes it clean).
/// One that is clean stays so: had the page been written since its copies
/// took it, [`open`] would have opened it. Whether it is read-only now.
///
/// With `plan` ([`alias_later`]) the `VirtualProtect` is left there for
/// [`guard_parts`] to make with the lock let go: `block` is noted guarded
/// and [`PENDING`] now, the block whose page it is a view of relies on it,
/// and it counts as read-only.
fn protect_alias(g: &mut Guards, block: usize, plan: Option<(&mut Aliases, usize)>) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE};
    g.aliased.insert(block);
    ALIASED.set(block);
    if alias_later() && PENDING.any(block, block + 1) {
        // Not read-only yet: covered only once this asking makes it so.
        return match plan {
            Some((aliases, of)) if aliases.planned.contains(&block) => {
                aliases.relies.push((of, block));
                true
            }
            _ => false,
        };
    }
    if loose(g, block) && g.protected.contains(block) {
        // A fault is opening it.
        return false;
    }
    if g.protected.contains(block) {
        // With parts open, a write through those would not be seen here.
        let claimed = loose_parts() && FINE.any(block, block + 1);
        if claimed && !OPENING.put(block) {
            // A fault is opening a part.
            return false;
        }
        let closed = PARTED.get(block) == 0 || close(block);
        if closed {
            PARTED.whole(block);
        }
        if claimed {
            OPENING.clear(block);
        }
        return closed;
    }
    if g.left_alone(block) || g.rests(block, rest()) {
        return false;
    }
    // Ahead of the protection.
    PROTECTED.set(block);
    if let Some((aliases, of)) = plan {
        // Noted guarded now, so that one who opens it meanwhile opens it
        // as any other (`aliases_done` makes it writable again after).
        PENDING.set(block);
        g.protected.insert(block);
        aliases.planned.insert(block);
        aliases.blocks.push(block);
        aliases.relies.push((of, block));
        return true;
    }
    let mut old = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: title view memory; only its protection changes.
    let done = unsafe { protect_as(0, (block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0;
    #[cfg(feature = "diagnostics")]
    {
        tally(5, 1);
        tally(8, started.elapsed().as_nanos() as u64);
    }
    if done && matches!(old, PAGE_READWRITE | PAGE_READONLY) {
        g.protected.insert(block);
        #[cfg(feature = "diagnostics")]
        crate::faultbook::note(crate::faultbook::What::Alias, block, 0, 0);
        return true;
    }
    if done {
        // Not for us to guard: as it was.
        // SAFETY: as above.
        unsafe { protect_as(8, (block * BLOCK) as *const _, BLOCK, old, &mut old) };
    }
    PROTECTED.clear(block);
    false
}

/// Makes blocks `[from, to)` read-only in one call: whether it was done.
/// Memory that was neither read-write nor read-only stays as it was. `who`
/// is a row of [`PROTECTS`].
fn read_only(from: usize, to: usize, who: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE};
    let mut old = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: title view memory; only its protection changes.
    let mut done = unsafe {
        protect_as(
            who,
            (from * BLOCK) as *const _,
            (to - from) * BLOCK,
            PAGE_READONLY,
            &mut old,
        ) != 0
    };
    if done && !matches!(old, PAGE_READWRITE | PAGE_READONLY) {
        // What is there is not for us to guard (code, or memory the
        // title protects itself): as it was.
        // SAFETY: as above.
        unsafe { protect_as(8, (from * BLOCK) as *const _, (to - from) * BLOCK, old, &mut old) };
        done = false;
    }
    #[cfg(feature = "diagnostics")]
    {
        tally(5, (to - from) as u64);
        tally(8, started.elapsed().as_nanos() as u64);
    }
    done
}

/// What Windows says is at a block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum There {
    /// A view: committed, mapped from a section.
    View,
    /// Nothing: a reservation (a placeholder), or free.
    Hole,
    /// Other memory, committed: the title's own heaps, the runtime's, a
    /// library's, a thread's stack.
    Other,
}

/// What is at `block` ([`There`]), by its first page, and the end
/// (exclusive) of the whole blocks from `block` on that it holds for (`block`
/// itself if the region ends within it), with what Windows calls its kind and
/// protection: one question for the blocks of a region.
fn there(block: usize) -> (There, usize, u32, u32) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_MAPPED, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n == 0 {
        // Nothing Windows knows of: not ours to guard, whatever it is.
        return (There::Other, block, 0, 0);
    }
    let end = ((info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK).max(block);
    let what = match info.State {
        MEM_COMMIT if info.Type == MEM_MAPPED => There::View,
        MEM_RESERVE | MEM_FREE => There::Hole,
        _ => There::Other,
    };
    (what, end, info.Type, info.Protect)
}

/// [`protect`] for one range, without the other views of its pages.
///
/// Only view memory is made read-only. A range asked about as a view is one
/// by where it begins ([`is_view`]), and what follows in it need not be the
/// title's at all: a window of title memory is cut on a grid (4 MiB), and the
/// one over the end of a reservation of the title's runs on into whatever
/// lies behind it. Under Wine that was a thread's stack for its calls into
/// the system, put into the gap before the title's next reservation (which
/// begins on a 2 MiB line): private memory, committed and read-write, as a
/// view is for all a change of protection tells. Made read-only it took the
/// thread with it, and the process: the thread's own data for the C library
/// and the kernel lie at that stack's top, a fault in a call into the system
/// is not one our handler is given, and what the kernel could not write
/// there (the thread's restartable-sequence area) it ended the process for,
/// with a segmentation fault that names no address.
///
/// So Windows is asked what is at a block before it is guarded ([`there`]),
/// a region at a time, every time: a view is guarded; a hole is noted absent,
/// as before; other memory is left as it is and noted foreign, which is told
/// as written at every asking and so copied every time. The answer is not
/// kept from one guarding to the next: memory that is not the runtime's (a
/// library's file mapping behind a reservation, say) goes and comes with
/// nobody telling, and an answer kept would guard what came in its place.
/// What that costs is one question for each region guarded, where a block is
/// guarded again only after a write fault opened it: a few hundred a frame,
/// each beside a change of protection that costs more. The same goes for a
/// run guarded with the lock let go ([`all_views`], [`read_only_blocks`]).
fn protect_here(g: &mut Guards, from: usize, to: usize) {
    let mut block = from;
    while block < to {
        let (what, whole, kind, protection) = there(block);
        let whole = whole.min(to);
        // An allocation begins on a block's line: a block whose first page
        // is of one has nothing of another's.
        let until = whole.max(block + 1);
        match what {
            There::View => protect_views(g, block, until),
            There::Hole => {
                // (A block that only begins as a hole is left for the next
                // asking, as before.)
                for block in block..whole {
                    g.absent.insert(block);
                    if known_holes() && placeholder(block) {
                        g.held.insert(block);
                    }
                }
                #[cfg(feature = "diagnostics")]
                tally(9, 1);
            }
            There::Other => {
                g.foreign.extend(block..until);
                not_a_view(block, until, kind, protection);
            }
        }
        block = until;
    }
}

/// Says that the guards met memory that is no view in a range asked about as
/// one, `[from, to)` in blocks, the first few times: where the title's views
/// end and another's memory begins is worth knowing of a run.
fn not_a_view(from: usize, to: usize, kind: u32, protection: u32) {
    static TOLD: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if TOLD.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
        tracing::info!(
            at = format_args!("{:#x}", from * BLOCK),
            bytes = (to - from) * BLOCK,
            kind = format_args!("{kind:#x}"),
            protection = format_args!("{protection:#x}"),
            "memory in a range asked about as a view is no view (private memory, an image, a thread's stack): the guards leave it as it is, and it is copied every time"
        );
    }
}

/// [`protect_here`] for blocks Windows said were view memory just now
/// ([`there`]). Called for nothing else.
fn protect_views(g: &mut Guards, from: usize, to: usize) {
    let set = |from, to| read_only(from, to, 9);
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.insert_all(from..to);
        g.clean.insert_all(from..to);
    };
    // Ahead of the protection (`PROTECTED`).
    (from..to).for_each(|block| PROTECTED.set(block));
    if set(from, to) {
        note(g, from, to);
        return;
    }
    let mut uncommitted_until = from;
    for block in from..to {
        // (Not one question a block of a region that is not there.)
        if block >= uncommitted_until && set(block, block + 1) {
            note(g, block, block + 1);
            continue;
        }
        PROTECTED.clear(block);
        if block >= uncommitted_until {
            uncommitted_until = uncommitted_end(block);
            #[cfg(feature = "diagnostics")]
            tally(9, 1);
        }
        if block < uncommitted_until {
            g.absent.insert(block);
            if known_holes() && placeholder(block) {
                g.held.insert(block);
            }
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; how many there were. The same physical pages at
/// every other view of them are written by that too (the write that faulted,
/// or is ours, `crate::host::copy_mapped`, goes through one view): they are
/// clean no longer, whether they are opened as well ([`open_one_view`]) or
/// not.
fn open(g: &mut Guards, start: usize, end: usize) -> usize {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    if first >= last {
        return 0;
    }
    let mut blocks: Vec<usize> = g.protected.range(first..last).collect();
    let here = blocks.len();
    let aliased: Vec<usize> = g.aliased.range(first..last).collect();
    if !aliased.is_empty() {
        let one = open_one_view();
        for block in aliased {
            g.aliased.remove(block);
            ALIASED.clear(block);
            for other in crate::inplace::pages_of(block * BLOCK) {
                let other = other / BLOCK;
                if one {
                    // Before the first can be written.
                    g.clean.remove(other);
                    PUBLISHED.clear(other);
                    PARTED.whole(other);
                    continue;
                }
                g.aliased.remove(other);
                ALIASED.clear(other);
                if !(first..last).contains(&other) && g.protected.contains(other) {
                    blocks.push(other);
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        ALIASES[1].fetch_add((blocks.len() - here) as u64, std::sync::atomic::Ordering::Relaxed);
    }
    unguard(g, &blocks);
    here
}

/// Gives write access back to `blocks`, guarded ones: one `VirtualProtect`
/// a run of blocks next to each other where it can ([`open_run`]), else one
/// a block.
fn unguard(g: &mut Guards, blocks: &[usize]) {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let mut sorted = blocks.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let held = hold_heap(&sorted);
    // All of them before the first can be written: a write through one is a
    // write of the others' bytes.
    for &block in blocks {
        g.protected.remove(block);
        g.clean.remove(block);
        PARTED.take(block);
        PUBLISHED.clear(block);
    }
    for run in sorted.chunk_by(|a, b| a + 1 == *b) {
        if run.len() == 1 || !open_run(run[0], run[run.len() - 1] + 1) {
            run.iter().for_each(|&block| open_block(block));
        }
        run.iter().for_each(|&block| PROTECTED.clear(block));
    }
    held.iter().for_each(|&block| OPENING.clear(block));
    #[cfg(feature = "diagnostics")]
    if !blocks.is_empty() {
        tally(8, started.elapsed().as_nanos() as u64);
    }
}

/// With `VICTORY_LOOSE_PARTS=2`, holds the heap blocks of `blocks` (no two
/// the same) in [`OPENING`] for a guard let go of under the lock: no asking
/// with no lock looks at them meanwhile. The caller lets go of those
/// returned after `PROTECTED` is cleared.
fn hold_heap(blocks: &[usize]) -> Vec<usize> {
    if !loose_ask() {
        return Vec::new();
    }
    let held: Vec<usize> = blocks.iter().copied().filter(|&block| FINE.any(block, block + 1)).collect();
    for &block in &held {
        // (Who holds one with no lock waits for nothing meanwhile.)
        while !OPENING.put(block) {
            std::thread::yield_now();
        }
    }
    held
}

#[cfg(test)]
thread_local! {
    /// `VirtualProtect` calls the test thread made to open guarded blocks.
    static TEST_OPEN_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Gives write access back to the guarded blocks `[first, last)` with one
/// `VirtualProtect`, when they are one region of one allocation, committed
/// and read-only (as we left them); whether it did.
///
/// A title write over a range and a range mapped again open long runs, and
/// one call a block was most of the guard lock's time (upstream issue #400:
/// about 3,800 calls and 600 ms a frame). Over separate allocations (views,
/// placeholders) one call fails, and over other memory among ours it would
/// make that writable: those are left to [`open_block`].
fn open_run(first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((first * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n == 0
        || info.State != MEM_COMMIT
        || info.Protect != PAGE_READONLY
        || (info.BaseAddress as usize).saturating_add(info.RegionSize) < last * BLOCK
    {
        return false;
    }
    let mut old = 0;
    // SAFETY: blocks we made read-only, all of one region.
    let done =
        unsafe { protect_as(4, (first * BLOCK) as *const _, (last - first) * BLOCK, PAGE_READWRITE, &mut old) } != 0;
    #[cfg(test)]
    TEST_OPEN_CALLS.with(|c| c.set(c.get() + 1));
    #[cfg(feature = "diagnostics")]
    if done {
        OPENS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        OPENS[1].fetch_add((last - first) as u64, std::sync::atomic::Ordering::Relaxed);
    }
    done
}

/// Gives write access back to guarded block `block`.
fn open_block(block: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE};
    let mut old = 0;
    // SAFETY: a block we made read-only.
    unsafe {
        if protect_as(5, (block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) != 0
            && !matches!(old, PAGE_READONLY | PAGE_READWRITE)
        {
            // Not what we left there: the address is other memory's
            // (code's, say), which keeps its protection.
            protect_as(8, (block * BLOCK) as *const _, BLOCK, old, &mut old);
        }
    }
    #[cfg(test)]
    TEST_OPEN_CALLS.with(|c| c.set(c.get() + 1));
    #[cfg(feature = "diagnostics")]
    OPENS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// How many 4 KiB parts of a block of the descriptor heaps ([`FINE`]) write
/// faults open one by one; the next opens the whole block
/// (`VICTORY_GUARD_PARTS`, 4; 0 opens whole blocks, as before).
///
/// An opened block was told as written whole, and the heap copy read and
/// compared all of it: 399,360 descriptor slots looked at a frame for about
/// 57,000 changed, 18.4 MB read, 9.7 ms reading and comparing, and 17 ms
/// asking (play-master, 2026-11-26 04:42). The title writes a few
/// descriptors at a time.
fn part_faults() -> u32 {
    static MOST: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_GUARD_PARTS").ok().and_then(|v| v.parse().ok()).unwrap_or(4).min(PARTS as u32)
    })
}

/// A write fault at `at`: opens only the 4 KiB part it is in, if the block
/// is a guarded one of the descriptor heaps ([`FINE`]), and notes the part
/// ([`PARTED`]); [`guard_parts`] tells those parts as written.
/// Whether it did, or the part was open already. `false`: open the whole
/// block. So too for a block with other views (a write through one of those
/// would not fault at this one's parts), or with [`part_faults`] parts open.
fn part_open(g: &mut Guards, at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE};
    let block = at / BLOCK;
    let most = part_faults();
    if most == 0 || !FINE.any(block, block + 1) || !g.protected.contains(block) {
        return false;
    }
    if g.aliased.contains(block) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardPartAliased, 1);
        return false;
    }
    let part = 1u16 << (at % BLOCK / KERNEL_PAGE);
    // No asking with no lock between the look at the parts and the part
    // opened (`guard_parts_loose`).
    let held = loose_ask();
    while held && !OPENING.put(block) {
        std::thread::yield_now();
    }
    let done = |opened: bool| {
        if held {
            OPENING.clear(block);
        }
        opened
    };
    let open = PARTED.get(block);
    // (A part opened with no lock may have been let be read-only again.)
    if open & part != 0 && !loose_parts() {
        return true;
    }
    if open & part == 0 && open.count_ones() >= most {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardPartWhole, 1);
        return done(false);
    }
    // Told as written before the part can be.
    if held {
        LOOSE_PART.set(block);
    }
    g.clean.remove(block);
    PUBLISHED.clear(block);
    let page = at & !(KERNEL_PAGE - 1);
    let mut old = 0;
    // SAFETY: a block we made read-only; one kernel page of it.
    if unsafe { protect_as(7, page as *const _, KERNEL_PAGE, PAGE_READWRITE, &mut old) } == 0 {
        return done(false);
    }
    if !matches!(old, PAGE_READONLY | PAGE_READWRITE) {
        // SAFETY: as above.
        unsafe { protect_as(8, page as *const _, KERNEL_PAGE, old, &mut old) };
        return done(false);
    }
    PARTED.or(block, part);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::GuardPartFaults, 1);
    done(true)
}

/// Makes all of a guarded `block` read-only again, the parts [`part_open`]
/// opened too.
fn close(block: usize) -> bool {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;
    let mut old = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: title view memory; only its protection changes.
    let done = unsafe { protect_as(3, (block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0;
    #[cfg(feature = "diagnostics")]
    {
        tally(5, 1);
        tally(8, started.elapsed().as_nanos() as u64);
    }
    done
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    let last = at.saturating_add(len).div_ceil(BLOCK);
    if !light() || PROTECTED.any(at / BLOCK, last) {
        let mut g = guards_timed();
        let opened = open(&mut g, at, at.saturating_add(len));
        #[cfg(feature = "diagnostics")]
        {
            tally(6, opened as u64);
            crate::faultbook::note(crate::faultbook::What::Ours, at / BLOCK, last, opened as u64);
        }
        let _ = opened;
    }
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// The kernel writes title memory `[at, at + len)` until what this returns
/// is dropped ([`kernel_writing`]).
pub struct KernelWrite(usize, usize);

/// The runtime is about to have the kernel write title memory
/// `[at, at + len)` (a file read): its guarded blocks are opened, as by
/// [`writing`], and no refresh guards them again before what this returns is
/// dropped, at the end of that call. One that did would make the kernel's
/// write fail (the kernel does not fault into our handler): a read cut
/// short, which the title takes for what the file holds.
pub fn kernel_writing(at: usize, len: usize) -> KernelWrite {
    // The kernel's write does not fault into the handler.
    materialize(at, len);
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    {
        let mut g = guards_timed();
        open(&mut g, at, at.saturating_add(len));
        if len != 0 {
            // Nothing known of them stands while they are left alone.
            let absent: Vec<usize> = g.absent.range(first..last).collect();
            for block in absent {
                g.absent.remove(block);
                if g.held.remove(block) {
                    PUBLISHED.clear(block);
                }
            }
            g.kernel.push((first, last));
        }
    }
    // A block opened above may be made read-only yet by an asking that took
    // it before (`pend`): opened again when that one has the lock.
    while PENDING.any(first, last) {
        std::thread::yield_now();
    }
    crate::descriptor::host::remapped(at, len);
    KernelWrite(first, if len == 0 { first } else { last })
}

impl Drop for KernelWrite {
    fn drop(&mut self) {
        if self.0 == self.1 {
            return;
        }
        let mut g = guards();
        if let Some(i) = g.kernel.iter().position(|&r| r == (self.0, self.1)) {
            g.kernel.swap_remove(i);
        }
    }
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `guards`: bit 1 alone. `0`: nothing, counted only (the default
/// until run 199). `1`: bits 1 and 2. `slots`: bit 2 alone. `all`:
/// everything.
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 makes texture slots due and is the one that flooded the
/// queue in an earlier form: still asked for by name.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ => 1,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    {
        let mut g = guards();
        let guarded = g.protected.range(at / BLOCK..end.div_ceil(BLOCK)).count() as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&mut g, at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again - and are, the pages noted written, which also has the next check
/// guard the new view ([`cover`]).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    // (A block is noted in `PROTECTED` ahead of the look at its page's views
    // that did not find this one yet.)
    if !ranges.iter().any(|&(at, n)| PROTECTED.any(at / BLOCK, (at + n).div_ceil(BLOCK))) {
        return;
    }
    // All of them, once one is guarded: another may have been made clean
    // since that look, relying on those that are opened here. And all
    // together: none may be writable while another is clean (who asks about
    // that one by its bits takes no lock).
    let mut g = guards();
    let blocks: Vec<usize> =
        ranges.iter().flat_map(|&(at, n)| g.protected.range(at / BLOCK..(at + n).div_ceil(BLOCK)).collect::<Vec<_>>()).collect();
    #[cfg(feature = "diagnostics")]
    ALIASES[2].fetch_add(blocks.len() as u64, std::sync::atomic::Ordering::Relaxed);
    unguard(&mut g, &blocks);
    if map_rest() {
        for block in blocks {
            g.opened(block);
        }
    }
}

/// The runtime is about to unmap a view of which it maps a part again at
/// once ([`title_memory_unmapping`] of all of it comes next): the title's
/// threads may write through that part before we are told of it
/// ([`views_mapped`]). What is guarded of its pages at other views is opened,
/// and not guarded again before the change is done ([`Remapping`]).
pub fn title_memory_splitting(at: usize, len: usize) {
    let others: Vec<usize> = elsewhere(at, len).into_iter().flat_map(|(_, others)| others).collect();
    if others.is_empty() {
        return;
    }
    let mut g = guards();
    // (The view itself too: the part that stays is read-only until then. And
    // all of them told as written before the first is writable.)
    for &other in &others {
        g.clean.remove(other / BLOCK);
        PUBLISHED.clear(other / BLOCK);
    }
    open(&mut g, at, at.saturating_add(len));
    for other in others {
        open(&mut g, other, other + BLOCK);
        if map_rest() {
            g.opened(other / BLOCK);
        }
        if CHANGING.get().0 {
            CHANGING.set((true, true));
            g.changing.push((other / BLOCK, other / BLOCK + 1));
        }
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

impl Drop for Remapping {
    fn drop(&mut self) {
        // The change is done: what is there now may be guarded.
        if CHANGING.replace((false, false)).1 {
            guards().changing.clear();
        }
        CHANGES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// How often the runtime began or ended a change of what title memory
/// shows: odd while one is under way.
static CHANGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

thread_local! {
    /// This thread holds [`Remapping`]; and put blocks in
    /// [`Guards::changing`].
    static CHANGING: std::cell::Cell<(bool, bool)> = const { std::cell::Cell::new((false, false)) };
}

/// For [`again_after_change`]: to be asked before what may fail.
pub(crate) fn changes() -> u64 {
    CHANGES.load(std::sync::atomic::Ordering::SeqCst)
}

/// `f` failed to read or write title memory, at some time since [`changes`]
/// said `since`. If the runtime changed a mapping in that time, the pages
/// may have been away for that moment only ([`MAPPING`]): again, when the
/// change is done. Whether `f` did it.
pub(crate) fn again_after_change(since: u64, mut f: impl FnMut() -> bool) -> bool {
    let mut since = since;
    for _ in 0..4 {
        let now = change_done();
        if now == since {
            return false;
        }
        since = now;
        if f() {
            return true;
        }
    }
    false
}

/// Returns when the change of a mapping that is under way, if one is, is
/// over (or has been going on for 2 ms): pages it took away for its moment
/// are back. Another may have begun.
pub(crate) fn change_over() {
    let now = changes();
    if now.is_multiple_of(2) {
        return;
    }
    let started = std::time::Instant::now();
    while changes() == now && started.elapsed().as_millis() < 2 {
        for _ in 0..50 {
            std::hint::spin_loop();
        }
    }
}

/// [`changes`], when no change is under way (or one has been for a second).
fn change_done() -> u64 {
    let mut tries = 0;
    let mut started = None;
    loop {
        let now = changes();
        if now.is_multiple_of(2) {
            return now;
        }
        tries += 1;
        if tries < 100 {
            std::hint::spin_loop();
            continue;
        }
        if started.get_or_insert_with(std::time::Instant::now).elapsed().as_secs() >= 1 {
            return now;
        }
        // SAFETY: gives the processor to another thread.
        unsafe { windows_sys::Win32::System::Threading::Sleep(u32::from(tries > 200)) };
    }
}

/// Faults of title threads that met a change of mapping and went on after
/// it ([`came_back`]).
static MET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A title thread's access to `at` (`write`: a write) faulted. Whether it
/// can be made now: if a change of mapping is under way, after it.
///
/// The console changes one page's mapping and no other's. Here the rest of
/// a view is unmapped and mapped again ([`MAPPING`]), and a title thread that
/// uses a page of it just then, which the title never gave up, faulted
/// ("reserved, never committed", or read-only for the moment between our
/// forgetting a guarded block and its unmapping). It waits here instead.
/// Memory that is not there with no change under way is not waited for.
fn came_back(at: usize, write: bool) -> bool {
    let there = || {
        if !write {
            return readable_view(at);
        }
        // (A refresh may have guarded the page that came back.)
        open(&mut guards(), at, at + 1) != 0 || writable(at)
    };
    let before = changes();
    if there() || again_after_change(before, there) {
        MET.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return true;
    }
    false
}

/// Whether `at` is in a committed page of a view that can be read.
fn readable_view(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == MEM_COMMIT
        && info.Type == MEM_MAPPED
        && matches!(info.Protect, PAGE_READONLY | PAGE_READWRITE)
}

/// Title memory that could not be written with what was to go there (what a
/// shader wrote, a title's copy): counted, and said at the 1st, 2nd, 4th, 8th
/// and so on. A page the title unmapped loses nothing by it; a page that is
/// there does.
pub(crate) fn lost(what: &'static str, at: usize, len: usize) {
    static LOST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = LOST.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        let there = writable(at);
        tracing::warn!(n, what, at = format_args!("{at:#x}"), len, there, "title memory not written");
    }
}


/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    // For the title's own threads ([`came_back`]).
    install();
    let met = MET.load(std::sync::atomic::Ordering::Relaxed);
    if met != 0 {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if met >= 2 * SAID.load(std::sync::atomic::Ordering::Relaxed) {
            SAID.store(met, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(met, "accesses of title threads that waited for a change of mapping");
        }
    }
    let changing = MAPPING.write().unwrap_or_else(|e| e.into_inner());
    CHANGES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    CHANGING.set((true, false));
    Remapping(changing)
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    match MAPPING.try_read() {
        Ok(guard) => return Reading(guard),
        Err(std::sync::TryLockError::Poisoned(e)) => return Reading(e.into_inner()),
        Err(std::sync::TryLockError::WouldBlock) => {}
    }
    let started = std::time::Instant::now();
    let guard = MAPPING.read().unwrap_or_else(|e| e.into_inner());
    mapping_cost("wait", started.elapsed().as_nanos() as u64, 0);
    Reading(guard)
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    match MAPPING.try_read() {
        Ok(guard) => Some(Reading(guard)),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(e.into_inner())),
        Err(std::sync::TryLockError::WouldBlock) => {
            mapping_cost("miss", 0, 0);
            None
        }
    }
}

/// By kind and thread: how often, nanoseconds in all, the longest, bytes.
/// Kinds: `wait`, a read of title memory in place that waited for a change
/// of mapping; `miss`, one that would have waited and went another way;
/// `copy`, the title's CopyPageMappings (bytes mapped and unmapped).
static MAPPING_COSTS: Mutex<BTreeMap<(&'static str, String), [u64; 4]>> = Mutex::new(BTreeMap::new());

pub(crate) fn mapping_cost(kind: &'static str, ns: u64, bytes: u64) {
    let thread = std::thread::current();
    let name = thread.name().map_or_else(|| format!("{:?}", thread.id()), str::to_owned);
    let mut costs = MAPPING_COSTS.lock().unwrap_or_else(|e| e.into_inner());
    let n = costs.entry((kind, name)).or_default();
    n[0] += 1;
    n[1] += ns;
    n[2] = n[2].max(ns);
    n[3] += bytes;
}

/// What [`mapping_cost`] counted since the last call, one tab-separated
/// entry each: kind, count, microseconds in all, the longest, bytes, thread.
pub fn mapping_costs() -> Option<String> {
    let costs = std::mem::take(&mut *MAPPING_COSTS.lock().unwrap_or_else(|e| e.into_inner()));
    (!costs.is_empty()).then(|| {
        let all: Vec<String> = costs
            .iter()
            .map(|((kind, thread), n)| format!("{kind},{},{},{},{},{thread}", n[0], n[1] / 1000, n[2] / 1000, n[3]))
            .collect();
        all.join("\t")
    })
}

/// The runtime mapped title memory `[at, at + len)`: what it shows is new,
/// so it is no longer guarded (nor known unmapped), and read-write.
///
/// The view is there before we are told of it, and a copy's refresh on
/// another thread may have guarded its blocks in between (it finds blocks
/// it knew absent committed, and holds [`GUARDS`], which we wait for).
/// Forgetting such a block and leaving it read-only would make the title's
/// next write to it an access violation nobody takes (`on_write` finds it
/// neither guarded nor writable) and a kernel read into it fail, so the
/// write access is given back.
pub fn title_memory_remapped(at: usize, len: usize) {
    crate::texback::remapped(at, len);
    open(&mut title_memory_changed(at, len), at, at.saturating_add(len));
}

/// The runtime is about to unmap title memory `[at, at + len)`: it is
/// nothing of ours any more. Its protection goes with it. Until the change
/// is done ([`Remapping`]) no refresh guards what is there: one that did
/// before the unmapping would leave the address guarded, clean and known by
/// its bits ([`PUBLISHED`]) with no view behind it.
pub fn title_memory_unmapping(at: usize, len: usize) {
    crate::texback::remapped(at, len);
    let mut g = title_memory_changed(at, len);
    if CHANGING.get().0 && len != 0 {
        CHANGING.set((true, true));
        let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
        match g.changing.last_mut() {
            Some(before) if before.1 == first => before.1 = last,
            _ => g.changing.push((first, last)),
        }
    }
    let blocks: Vec<usize> = g.protected.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).collect();
    let held = hold_heap(&blocks);
    for block in blocks {
        g.protected.remove(block);
        g.clean.remove(block);
        PARTED.take(block);
        PUBLISHED.clear(block);
        PROTECTED.clear(block);
    }
    held.iter().for_each(|&block| OPENING.clear(block));
}

/// The title is about to free or decommit memory of its own,
/// `[at, at + len)`, or with no length the whole allocation at `at`: what
/// was noted of its blocks (guarded, clean, absent) would be taken for true
/// of the next memory at the address.
pub fn title_memory_freed(at: usize, len: usize) {
    crate::shadow::write_combined_freed(at, len);
    let first = at / BLOCK;
    if len != 0 {
        forget_beside(first, at.saturating_add(len).div_ceil(BLOCK));
    } else if BESIDE.any(first, first + Bits::WORDS * 64) {
        forget_beside(first, allocation_end(at, at).div_ceil(BLOCK).max(first + 1));
    }
    let mut g = guards();
    let noted = [g.protected.range(first..).next(), g.absent.range(first..).next(), g.foreign.range(first..).next().copied()];
    let Some(noted) = noted.into_iter().flatten().min() else {
        return;
    };
    let end = if len != 0 {
        at.saturating_add(len).div_ceil(BLOCK)
    } else {
        allocation_end(at, noted * BLOCK).div_ceil(BLOCK)
    };
    if noted >= end {
        return;
    }
    let absent: Vec<usize> = g.absent.range(first..end).collect();
    for block in absent {
        g.absent.remove(block);
        if g.held.remove(block) {
            PUBLISHED.clear(block);
        }
    }
    let foreign: Vec<usize> = g.foreign.range(first..end).copied().collect();
    for block in foreign {
        g.foreign.remove(&block);
    }
    // (Memory that is still a view is not freed by that call, which fails:
    // it must not stay read-only.)
    open(&mut g, at, end * BLOCK);
}

/// What was put in beside the guards of `[first, last)` ([`BESIDE`], and the
/// blocks' share of it) is forgotten: it was for memory no longer there, and
/// the next memory at the address would take the bit as its own and be told
/// as written by an asking that comes before any write.
fn forget_beside(first: usize, last: usize) {
    if !BESIDE.any(first, last) {
        return;
    }
    let mut block = first;
    while block < last {
        let until = ((block / 64 + 1) * 64).min(last);
        if BESIDE.any(block, until) {
            for block in block..until {
                if BESIDE.take(block) {
                    put_count(block).store(0, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        block = until;
    }
}

/// The end of the allocation at `at`, if `inside` lies in it; else `at`.
fn allocation_end(at: usize, inside: usize) -> usize {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let ask = |address: usize| {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(address as *const _, &mut info, size_of_val(&info)) };
        (n != 0 && info.AllocationBase as usize == at)
            .then(|| (info.BaseAddress as usize).saturating_add(info.RegionSize))
    };
    let mut end = at;
    let mut next = ask(inside.max(at));
    while let Some(to) = next {
        end = to;
        next = ask(to);
    }
    end
}

/// Tells what keeps track of what title memory shows; the guards, with no
/// block of the range known unmapped.
fn title_memory_changed(at: usize, len: usize) -> std::sync::MutexGuard<'static, Guards> {
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let mut g = guards();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    forget_beside(range.start, range.end);
    let absent: Vec<usize> = g.absent.range(range.clone()).collect();
    for block in absent {
        g.absent.remove(block);
        if g.held.remove(block) {
            PUBLISHED.clear(block);
        }
    }
    let foreign: Vec<usize> = g.foreign.range(range.clone()).copied().collect();
    for block in foreign {
        g.foreign.remove(&block);
    }
    let aliased: Vec<usize> = g.aliased.range(range.clone()).collect();
    for block in aliased {
        g.aliased.remove(block);
        ALIASED.clear(block);
    }
    // What the title wrote again and again there was another page ([`hot`]).
    if hot().is_some() || cfg!(test) {
        g.opened.retain(|block, (_, hot)| hot.is_zero() || !range.contains(block));
    }
    g
}

/// The runtime's way to make title memory it mapped only in its books
/// real (`xmem`, `VICTORY_LAZY_VIEWS=1`): `(address, bytes)` makes every
/// such view that overlaps the range a mapping of the host's, and says
/// whether there was one (and all of them could be made).
static LAZY_VIEWS: std::sync::OnceLock<fn(usize, usize) -> bool> = std::sync::OnceLock::new();

/// The runtime maps title memory lazily: views it tells of (as mapped, by
/// [`title_memory_remapped`] and `title_memory_view`) may be no mapping of
/// the host's until something touches them. `materialize` makes them one.
pub fn use_lazy_views(materialize: fn(usize, usize) -> bool) {
    let _ = LAZY_VIEWS.set(materialize);
}

/// Makes title memory `[at, at + len)` that the runtime mapped lazily a
/// mapping of the host's ([`use_lazy_views`]); whether any of it was lazy.
/// Called before every touch of title memory the fault handler below does
/// not see: reads and writes through the kernel, protection changes, and
/// questions to Windows about the pages. (A fault in title memory, the
/// title's or ours, comes to [`install`]'s handler, which asks first.)
pub(crate) fn materialize(at: usize, len: usize) -> bool {
    LAZY_VIEWS.get().is_some_and(|f| f(at, len.max(1)))
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
pub(crate) fn install() {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, NTSTATUS, STATUS_GUARD_PAGE_VIOLATION,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        let _fault = crate::timeline::span("a fault looked at");
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const READ: usize = 0;
        const WRITE: usize = 1;
        // A fault of our own copies (`crate::safecopy`): whatever it is, if
        // it is not a block we guard, the copy fails and goes on from there.
        let ours = crate::safecopy::failing_exit(r.ExceptionAddress as usize);
        // SAFETY: as above.
        let context = unsafe { (*info).ContextRecord };
        let fail = move |exit: usize| {
            // SAFETY: the context the system passes, to go on from.
            unsafe { (*context).Rip = exit as u64 };
            EXCEPTION_CONTINUE_EXECUTION
        };
        // Memory the runtime mapped lazily: mapped now, and the access is
        // made again (any access, the title's or ours).
        if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            && r.NumberParameters >= 2
            && materialize(r.ExceptionInformation[1], 1)
        {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return match ours {
                Some(exit) => {
                    // A guard page of title memory fails the copy as the kernel's read would:
                    // the fault cleared its guard flag, so it is set again (or counted, if it
                    // cannot be).
                    if r.ExceptionCode == STATUS_GUARD_PAGE_VIOLATION {
                        rearm_guard((r.NumberParameters >= 2).then(|| r.ExceptionInformation[1]));
                    }
                    fail(exit)
                }
                None if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
                    && r.NumberParameters >= 2
                    && r.ExceptionInformation[0] == READ
                    && came_back(r.ExceptionInformation[1], false) =>
                {
                    EXCEPTION_CONTINUE_EXECUTION
                }
                None => EXCEPTION_CONTINUE_SEARCH,
            };
        }
        let at = r.ExceptionInformation[1];
        let block = at / BLOCK;
        if ours.is_none()
            && PROTECTED.any(block, block + 1)
            // SAFETY: the context the system passes, to go on from.
            && crate::titlewrite::store(unsafe { &mut *context }, at)
        {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // Another thread may have opened the block since this one faulted
        // (then nothing waits for the lock), and a copy may have guarded it
        // again since that (then it is ours again, not read-only for some
        // other reason).
        for _ in 0..3 {
            // Read-only by an asking's late protection after it was opened
            // (`pend`), with no note: opened again when that one has the
            // lock.
            while PENDING.any(block, block + 1) && !PROTECTED.any(block, block + 1) {
                std::thread::yield_now();
            }
            if !PROTECTED.any(block, block + 1) && writable(at) {
                #[cfg(feature = "diagnostics")]
                crate::faultbook::note(crate::faultbook::What::Open, block, 0, 0);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            if open_loose(block) {
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GuardFaults, 1);
                    crate::faultbook::note(crate::faultbook::What::Fault, block, 1 << 32, 0);
                    crate::faultbook::note(crate::faultbook::What::Site, r.ExceptionAddress as usize, block, 0);
                }
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            if open_part_loose(at) {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            #[cfg(feature = "diagnostics")]
            let began = std::time::Instant::now();
            // (Let go again before the next look: the runtime wants the
            // guards to end a change.)
            let mut g = guards();
            #[cfg(feature = "diagnostics")]
            let waited = began.elapsed();
            if part_open(&mut g, at) {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            let opened = open(&mut g, at, at + 1);
            if opened != 0 {
                g.opened(block);
                drop(g);
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GuardFaults, 1);
                    let ns = waited.as_nanos().min(u128::from(u32::MAX)) as usize;
                    let held = (began.elapsed() - waited).as_nanos() as u64;
                    crate::faultbook::note(crate::faultbook::What::Fault, block, opened << 32 | ns, held);
                    crate::faultbook::note(crate::faultbook::What::Site, r.ExceptionAddress as usize, block, 0);
                }
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        if writable(at) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        match ours {
            Some(exit) => fail(exit),
            None if came_back(at, true) => EXCEPTION_CONTINUE_EXECUTION,
            None => EXCEPTION_CONTINUE_SEARCH,
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Sets the guard flag again on the page holding `at` (the address the fault
/// reported, if it did), with the protection the page has now: the fault
/// cleared the flag. False, and counted by [`not_rearmed`], if the address is
/// not known, the page is not committed any more, or its protection cannot be
/// set.
fn rearm_guard(at: Option<usize>) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, VirtualProtect, VirtualQuery,
    };
    let done = at.is_some_and(|at| {
        let page = at / KERNEL_PAGE * KERNEL_PAGE;
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let mut old = 0;
        // SAFETY: valid out pointers of the right size; the protection change is of one page, and
        // only if that page is committed.
        unsafe {
            VirtualQuery(page as *const _, &mut info, size_of_val(&info)) != 0
                && info.State == MEM_COMMIT
                && VirtualProtect(
                    page as *const _,
                    KERNEL_PAGE,
                    info.Protect | PAGE_GUARD,
                    &mut old,
                ) != 0
        }
    });
    if !done {
        not_rearmed(at);
    }
    done
}

/// Guard pages that a copy stopped at and that could not be guarded again.
static NOT_REARMED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Counts a guard page that could not be guarded again; the first is said
/// (the title may then not grow its stack, or catch that guard, at the page).
fn not_rearmed(at: Option<usize>) {
    if NOT_REARMED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
        tracing::warn!(
            at = format_args!("{:#x}", at.unwrap_or(0)),
            "guard not re-armed: a copy of title memory took a guard page and it could not be \
             set again (0: no address)"
        );
    }
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod parts_tests {
    use super::*;

    #[test]
    fn the_parts_of_a_page_written_since_it_was_taken() {
        let mut j = Shard::new();
        assert_eq!(j.parts_since(7, 0), WHOLE);
        assert_eq!(j.note(7, WHOLE, 3), 0);
        // Parts 0 and 5 at 4, part 5 again at 6: at 2 the whole page is due.
        assert_eq!(j.note(7, 0b10_0001, 4), 3);
        assert_eq!(j.note(7, 0b10_0000, 6), 4);
        assert_eq!(j.parts_since(7, 2), WHOLE);
        assert_eq!(j.parts_since(7, 3), 0b10_0001);
        assert_eq!(j.parts_since(7, 4), 0b10_0000);
        assert_eq!(j.parts_since(7, 5), 0b10_0000);
        assert_eq!(j.parts_since(7, 6), 0);
        // Written whole: all of it, whoever took it before.
        assert_eq!(j.note(7, WHOLE, 8), 6);
        assert_eq!(j.parts_since(7, 7), WHOLE);
        // A page never written before: what was not named was never written.
        assert_eq!(j.note(9, 0b100, 9), 0);
        assert_eq!(j.parts_since(9, 5), 0b100);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        let _one = crate::watch::one_view_test();
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        // The journal may hold pages of another test's memory that was here.
        let (none, before) = writes(start, end, false).expect("watched");
        assert!(before.iter().all(|&(_, w)| w < none), "{before:?}");
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        let new = |w: Vec<(usize, u64)>| w.into_iter().filter(|w| w.1 > none).collect::<Vec<_>>();
        assert_eq!(new(after), [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(new(again), [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn reservation_is_the_allocation_base() {
        // SAFETY: two fresh reservations, the first committed in part, freed at the end.
        let (a, b) = unsafe {
            let a = VirtualAlloc(std::ptr::null(), 8 * PAGE, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            VirtualAlloc((a + 2 * PAGE) as _, PAGE, MEM_COMMIT, PAGE_READWRITE);
            (a, VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_READWRITE) as usize)
        };
        assert!(a != 0 && b != 0);
        assert_eq!(reservation(a / PAGE + 2), a / PAGE);
        assert_eq!(reservation(a / PAGE + 7), a / PAGE);
        assert_eq!(reservation(b / PAGE), b / PAGE);
        let mut pieces = [(a / PAGE, a / PAGE + 1), (a / PAGE + 2, a / PAGE + 3), (a / PAGE + 7, a / PAGE + 8), (b / PAGE, b / PAGE + 1)];
        pieces.sort_unstable();
        let one: Vec<usize> = pieces.iter().map(|p| reservation(p.0)).collect();
        assert_eq!(reservations(&pieces), one);
        // SAFETY: our reservations.
        unsafe {
            VirtualFree(a as *mut _, 0, MEM_RELEASE);
            VirtualFree(b as *mut _, 0, MEM_RELEASE);
        }
    }

    /// Our write into title memory after the asking that found it has ended is in the journal: the
    /// next submission's asking, answered from the journal, reports it. (`wrote` told the journal
    /// only while an asking stood.)
    #[test]
    fn our_write_between_two_askings_is_copied() {
        const MIB: usize = 1 << 20;
        // SAFETY: a fresh write-watched allocation, kept.
        let p = unsafe { VirtualAlloc(std::ptr::null(), 2 * MIB, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        let a = p.next_multiple_of(MIB);
        let (start, end) = (a, a + MIB);
        ask_once(&[(start, end)]);
        let (asked, _) = writes(start, end, false).expect("watched");
        ask_once_end();
        // A Copy step: our bytes go into title memory, and then the journal is told.
        // SAFETY: a committed page of ours.
        unsafe { ((a + 3 * PAGE + 8) as *mut u8).write(1) };
        wrote(a + 3 * PAGE, PAGE);
        // The next submission's asking: the write is in the journal.
        ask_once(&[(start, end)]);
        let (_, after) = writes(start, end, false).expect("watched");
        ask_once_end();
        // Not freed: the asking made the piece hot, and the early asker may
        // still ask it; a refusal there is kept for the frame, and a later
        // test's memory at this address would be refused.
        let page = (a + 3 * PAGE) / PAGE;
        assert!(after.iter().any(|&(p, w)| p == page && w > asked), "our write is not in the journal: {after:?} (asked {asked})");
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        let _one = crate::watch::one_view_test();
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // Before the view goes, as the runtime does: the address may be
        // another test's the moment it is free.
        title_memory_unmapping(view, 2 * BLOCK);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
    }

    /// A write in a guarded block of the descriptor heaps opens its 4 KiB
    /// part only, and the guard tells that part; past `part_faults` parts the
    /// block opens whole.
    #[test]
    fn heap_blocks_open_by_the_part() {
        heap_parts(0);
    }

    /// So too with the parts opened with no lock.
    #[test]
    fn heap_blocks_open_by_the_part_with_no_lock() {
        LOOSE_PARTS_HERE.set(1);
        heap_parts(1);
        LOOSE_PARTS_HERE.set(0);
    }

    /// And with the blocks asked about with no lock.
    #[test]
    fn heap_blocks_open_by_the_part_asked_with_no_lock() {
        LOOSE_PARTS_HERE.set(2);
        heap_parts(2);
        LOOSE_PARTS_HERE.set(0);
    }

    thread_local! {
        /// `VICTORY_LOOSE_PARTS` for this thread's faults and askings.
        pub(super) static LOOSE_PARTS_HERE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    }

    fn heap_parts(level: u8) {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let block = view / BLOCK;
        (block..block + 2).for_each(|b| FINE.set(b));
        // As `writes` asks: with no lock first at level 2, the blocks told
        // known by their bits after. Whether it was with no lock.
        let ask = || {
            let (start, end) = (view, view + 2 * BLOCK);
            if level == 2 {
                if let Some(told) = guard_parts_loose(start, end) {
                    publish_loose(&told.iter().map(|t| t.0).collect::<Vec<_>>());
                    return (Some(told), true);
                }
                let told = guard_parts(start, end);
                publish(&told.iter().flatten().map(|t| t.0).collect::<Vec<_>>());
                return (told, false);
            }
            (guard_parts(start, end), false)
        };
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        if level == 2 {
            publish(&[block, block + 1]);
        }
        // SAFETY: our view; a write to a guarded part faults into the handler.
        let poke = |part: usize| unsafe { ((view + part * KERNEL_PAGE + 8) as *mut u8).write_volatile(1) };
        poke(3);
        assert!(writable(view + 3 * KERNEL_PAGE));
        assert!(!writable(view + 4 * KERNEL_PAGE));
        assert_eq!(LOOSE_PART.any(block, block + 1), level != 0);
        poke(3);
        assert_eq!(ask(), (Some(vec![(block, 1 << 3)]), level == 2));
        assert!(!writable(view + 3 * KERNEL_PAGE));
        assert_eq!(ask(), (Some(Vec::new()), level == 2));
        (0..=part_faults() as usize).for_each(poke);
        assert!(writable(view + 15 * KERNEL_PAGE));
        assert_eq!(ask(), (Some(vec![(block, WHOLE)]), false));
        title_memory_unmapping(view, 2 * BLOCK);
        (block..block + 2).for_each(|b| FINE.clear(b));
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
    }

    /// A placeholder of `blocks` blocks, each its own.
    fn placeholder(blocks: usize) -> usize {
        use windows_sys::Win32::System::Memory::{
            MEM_PRESERVE_PLACEHOLDER, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder, split.
        unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                blocks * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            for block in 0..blocks - 1 {
                assert_ne!(
                    VirtualFree((p + block * BLOCK) as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER),
                    0
                );
            }
            p
        }
    }

    /// Private memory of `protect` in place of the placeholder of one block.
    fn private_at(at: usize, protect: u32) {
        use windows_sys::Win32::System::Memory::{MEM_REPLACE_PLACEHOLDER, VirtualAlloc2};
        // SAFETY: our placeholder.
        let got = unsafe {
            VirtualAlloc2(
                windows_sys::Win32::System::Threading::GetCurrentProcess(),
                at as *const _,
                BLOCK,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                protect,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(got as usize, at);
    }

    fn protection(at: usize) -> u32 {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data the query fills in.
        unsafe {
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            VirtualQuery(at as *const _, &mut info, size_of_val(&info));
            info.Protect
        }
    }

    /// A refresh between the opening of a kernel read's destination and the
    /// read itself leaves it open: the read lands, and is seen.
    #[test]
    fn a_refresh_leaves_open_what_the_kernel_is_writing() {
        let _one = crate::watch::one_view_test();
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK)
        };
        let at = view.Value as usize;
        assert_ne!(at, 0);
        let pages = |block: usize| at / PAGE + block..at / PAGE + block + 1;
        assert!(writes(at, at + 2 * BLOCK, true).is_some());
        assert!(published(at, at + 2 * BLOCK) || !light());
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        file.rewind().unwrap();
        let writing = kernel_writing(at + 100, 13);
        // The refresh: the block being written is new to it every time, and
        // stays open; the other is as it was.
        for _ in 0..2 {
            assert_eq!(guard(at, at + 2 * BLOCK), Some(pages(0).collect()));
            let (now, found) = writes(at, at + BLOCK, true).expect("a view");
            assert!(found.contains(&(at / PAGE, now)));
            assert!(writable(at) && !writable(at + BLOCK));
            assert!(!published(at, at + BLOCK));
        }
        // SAFETY: 13 bytes of our view, nothing else refers to them.
        let into = unsafe { std::slice::from_raw_parts_mut((at + 100) as *mut u8, 13) };
        assert_eq!(file.read(into).expect("the kernel's write lands"), 13);
        assert_eq!(into, b"from the disk");
        drop(writing);
        // Seen by the next refresh, and guarded again.
        assert_eq!(guard(at, at + 2 * BLOCK), Some(pages(0).collect()));
        assert!(!writable(at));
        assert_eq!(guard(at, at + 2 * BLOCK), Some(Vec::new()));
        title_memory_unmapping(at, 2 * BLOCK);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(view) };
    }

    /// A hole the kernel is to write is no known hole meanwhile, nor is one
    /// the runtime has begun to change; after that it is again.
    #[test]
    fn what_is_left_alone_is_no_known_hole() {
        let _one = crate::watch::one_view_test();
        let p = placeholder(3);
        let known = |block: usize| published(p + block * BLOCK, p + (block + 1) * BLOCK);
        let ask = || writes(p, p + 3 * BLOCK, true).expect("as a view");
        ask();
        let holes = known_holes();
        assert!((0..3).all(known) == holes);
        let writing = kernel_writing(p + BLOCK + 5, 100);
        for _ in 0..2 {
            assert!(known(0) == holes && !known(1) && known(2) == holes);
            let (now, written) = ask();
            assert!(written.contains(&(p / PAGE + 1, now)));
            assert!(written.iter().all(|w| w.0 == p / PAGE + 1 || w.1 < now));
        }
        assert!(!guards().held.contains(p / BLOCK + 1));
        drop(writing);
        ask();
        assert!((0..3).all(known) == holes);
        {
            let _changing = title_memory_remapping();
            title_memory_unmapping(p, BLOCK);
            ask();
            assert!(!known(0) && !guards().held.contains(p / BLOCK));
        }
        ask();
        assert!((0..3).all(known) == holes);
        title_memory_freed(p, BLOCK);
        assert!(!known(0) && !guards().held.contains(p / BLOCK));
        title_memory_unmapping(p, 3 * BLOCK);
        // SAFETY: our placeholders.
        unsafe {
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
        }
    }

    /// A refresh during a change of mapping guards nothing the runtime has
    /// begun to unmap: no note of it stays behind.
    #[test]
    fn nothing_is_guarded_between_our_forgetting_it_and_its_unmapping() {
        let _one = crate::watch::one_view_test();
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};
        // SAFETY: a pagefile section of a block, mapped read-write.
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK)
        };
        let at = view.Value as usize;
        assert_ne!(at, 0);
        assert!(writes(at, at + BLOCK, true).is_some());
        let changing = title_memory_remapping();
        title_memory_unmapping(at, BLOCK);
        assert!(writes(at, at + BLOCK, true).is_some());
        {
            let g = guards();
            assert!(!g.protected.contains(at / BLOCK) && !g.clean.contains(at / BLOCK));
        }
        assert!(!published(at, at + BLOCK));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(view) };
        drop(changing);
        assert!(!guards().left_alone(at / BLOCK));
    }

    /// Memory at an address where blocks were guarded or absent is not
    /// taken for them: code stays code, private memory is not guarded.
    #[test]
    fn other_memory_at_a_guarded_address_keeps_its_protection() {
        let _one = crate::watch::one_view_test();
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MapViewOfFile3, PAGE_EXECUTE_READ,
            PAGE_EXECUTE_READWRITE, UnmapViewOfFile2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let p = placeholder(3);
        // SAFETY: a section's block in place of the first placeholder.
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile3(
                section,
                GetCurrentProcess(),
                p as *const _,
                0,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(view.Value as usize, p);
        // One block guarded, two absent.
        assert_eq!(guard(p, p + 3 * BLOCK).map(|w| w.len()), Some(3));
        // Not told of what comes: the view goes, code is there; and private
        // memory in the holes.
        // SAFETY: our view.
        assert_ne!(unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) }, 0);
        private_at(p, PAGE_EXECUTE_READ);
        private_at(p + BLOCK, PAGE_READWRITE);
        private_at(p + 2 * BLOCK, PAGE_EXECUTE_READWRITE);
        // (The runtime's allocations it tells of, or the holes stay known.)
        title_memory_allocated(p + BLOCK, 2 * BLOCK);
        let as_made = || {
            assert_eq!(protection(p), PAGE_EXECUTE_READ);
            assert_eq!(protection(p + BLOCK), PAGE_READWRITE);
            assert_eq!(protection(p + 2 * BLOCK), PAGE_EXECUTE_READWRITE);
        };
        let holes = [p / PAGE + 1, p / PAGE + 2];
        for _ in 0..2 {
            let written = guard(p, p + 3 * BLOCK).expect("as a view");
            assert!(holes.iter().all(|page| written.contains(page)), "copied every time");
            as_made();
            let g = guards();
            assert!(holes.iter().all(|&block| !g.held.contains(block) && !g.absent.contains(block)));
        }
        writing(p, BLOCK);
        as_made();
        title_memory_unmapping(p, 3 * BLOCK);
        // SAFETY: our allocations.
        unsafe {
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
        }
    }

    /// A range asked about as a view (it begins in one) that runs on into
    /// memory that is no view, there before the first asking and never told
    /// of: the view is guarded, the other memory keeps its protection and is
    /// copied every time. (A window of title memory over the end of the
    /// title's reservation ran on into a thread's stack under Wine: made
    /// read-only, it ended the process.)
    #[test]
    fn memory_past_a_view_that_is_no_view_is_not_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MapViewOfFile3, PAGE_READONLY,
            UnmapViewOfFile2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let p = placeholder(4);
        // SAFETY: a section's block in place of the first placeholder.
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile3(
                section,
                GetCurrentProcess(),
                p as *const _,
                0,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(view.Value as usize, p);
        // Another's memory behind it, two blocks (a stack with its page
        // that is not written at the bottom), and a hole.
        private_at(p + BLOCK, PAGE_READONLY);
        private_at(p + 2 * BLOCK, PAGE_READWRITE);
        let block = |k: usize| p / BLOCK + k;
        let as_made = || {
            assert_eq!(protection(p + BLOCK), PAGE_READONLY);
            assert_eq!(protection(p + 2 * BLOCK), PAGE_READWRITE);
            assert!(!PROTECTED.any(block(1), block(3)));
            let g = guards();
            for k in 1..3 {
                assert!(!g.protected.contains(block(k)) && !g.clean.contains(block(k)) && !g.absent.contains(block(k)));
            }
        };
        // The first asking: all of it written, as far as anyone knows.
        assert_eq!(guard(p, p + 4 * BLOCK).map(|w| w.len()), Some(4));
        assert_eq!(protection(p), PAGE_READONLY, "the view is guarded");
        as_made();
        // A write there is nobody's fault.
        // SAFETY: our allocation, read-write.
        unsafe { ((p + 2 * BLOCK + 8) as *mut u8).write(1) };
        for _ in 0..2 {
            let written = guard(p, p + 4 * BLOCK).expect("as a view");
            assert_eq!(written, vec![p / PAGE + 1, p / PAGE + 2], "copied every time, and nothing else");
            as_made();
        }
        // The view is written and guarded again, as ever.
        // SAFETY: the view, opened by the handler.
        unsafe { ((p + 8) as *mut u8).write(1) };
        assert_eq!(guard(p, p + 4 * BLOCK), Some(vec![p / PAGE, p / PAGE + 1, p / PAGE + 2]));
        assert_eq!(protection(p), PAGE_READONLY);
        as_made();
        // Told of a change there, nothing is known of the blocks any more.
        {
            let g = guards();
            assert!(g.foreign.contains(&block(1)) && g.foreign.contains(&block(2)));
        }
        title_memory_unmapping(p, 4 * BLOCK);
        {
            let g = guards();
            assert!(!g.foreign.contains(&block(1)) && !g.foreign.contains(&block(2)));
        }
        // SAFETY: our view and our allocations.
        unsafe {
            assert_ne!(UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER), 0);
            for k in 0..4 {
                VirtualFree((p + k * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
        }
    }

    /// A view of a section of its own, one block, in place of the
    /// placeholder of one block at `at`, of which nobody is told.
    fn section_view_at(at: usize) -> windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, MEM_REPLACE_PLACEHOLDER, MapViewOfFile3};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a section's block in place of our placeholder; the view
        // keeps the section, whose handle is closed.
        unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            let view = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                at as *const _,
                0,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(view.Value as usize, at);
            windows_sys::Win32::Foundation::CloseHandle(section);
            view
        }
    }

    /// What Windows said of a block once is not kept: a guarded view that
    /// goes with nobody telling, and private memory in its place (a library's
    /// mapping behind the title's reservation, and then a thread's stack), is
    /// asked about again before it is guarded, and left as it is.
    #[test]
    fn what_comes_in_place_of_a_guarded_view_with_nobody_telling_is_not_guarded() {
        use windows_sys::Win32::System::Memory::{MEM_PRESERVE_PLACEHOLDER, PAGE_READONLY, UnmapViewOfFile2};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let p = placeholder(2);
        let (one, two) = (section_view_at(p), section_view_at(p + BLOCK));
        let other = p / BLOCK + 1;
        assert_eq!(guard(p, p + 2 * BLOCK).map(|w| w.len()), Some(2));
        assert_eq!([protection(p), protection(p + BLOCK)], [PAGE_READONLY; 2]);
        // The second view goes and private memory comes, untold.
        // SAFETY: our view.
        assert_ne!(unsafe { UnmapViewOfFile2(GetCurrentProcess(), two, MEM_PRESERVE_PLACEHOLDER) }, 0);
        private_at(p + BLOCK, PAGE_READWRITE);
        // What was noted guarded there is opened (our own write through the
        // kernel, say), so the next asking would guard it again.
        writing(p + BLOCK, BLOCK);
        assert!(!guards().protected.contains(other));
        for _ in 0..2 {
            assert_eq!(guard(p, p + 2 * BLOCK), Some(vec![p / PAGE + 1]), "copied every time");
            assert_eq!(protection(p + BLOCK), PAGE_READWRITE, "asked about again: no view, and left as it is");
            assert!(!PROTECTED.any(other, other + 1));
            let g = guards();
            assert!(g.foreign.contains(&other) && !g.protected.contains(other));
        }
        assert_eq!(protection(p), PAGE_READONLY, "the view that stayed is guarded still");
        title_memory_unmapping(p, 2 * BLOCK);
        // SAFETY: our view and our allocations.
        unsafe {
            assert_ne!(UnmapViewOfFile2(GetCurrentProcess(), one, MEM_PRESERVE_PLACEHOLDER), 0);
            for k in 0..2 {
                VirtualFree((p + k * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
        }
    }

    /// Memory the title frees leaves no note of its blocks, whether it names
    /// its length or the allocation.
    #[test]
    fn freed_memory_leaves_no_notes() {
        let _one = crate::watch::one_view_test();
        let p = placeholder(3);
        let noted = |block: usize| {
            let g = guards();
            let block = p / BLOCK + block;
            g.protected.contains(block) || g.clean.contains(block) || g.absent.contains(block)
        };
        assert!(guard(p, p + 3 * BLOCK).is_some());
        assert!(noted(0) && noted(1) && noted(2));
        // The middle allocation.
        title_memory_freed(p + BLOCK, 0);
        assert!(noted(0) && !noted(1) && noted(2));
        title_memory_freed(p, BLOCK);
        assert!(!noted(0) && noted(2));
        // Nothing noted there: nothing to do, and nothing beside it.
        title_memory_freed(p + BLOCK, 0);
        assert!(noted(2));
        title_memory_freed(p + 2 * BLOCK, 0);
        assert!(!noted(2));
        // SAFETY: our placeholders.
        unsafe {
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
        }
    }

    /// A write between another thread's asking about a range and the second
    /// part of a submission's one asking is later than what that thread
    /// copied.
    #[test]
    fn the_rest_of_the_one_asking_has_a_number_of_its_own() {
        let _one = crate::watch::one_view_test();
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        // One range: `ask_ahead` leaves it.
        let ranges = [(p, p + 4 * PAGE)];
        ask_ahead(&ranges);
        let (copied, _) = writes(p, p + 4 * PAGE, false).expect("watched");
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE) as *mut u8).write_volatile(1) };
        ask_rest(&ranges);
        let last = journal().notes.range(p / PAGE + 1, p / PAGE + 2).first().map(|n| n.1);
        ask_once_end();
        assert!(last.is_some_and(|last| last > copied), "{last:?} is not after {copied}");
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// An asking begun on the thread kept for it is made when it has been
    /// waited for, and a range it did not hold is asked for afterwards.
    /// (Other tests end the asking that stands at any time: what is looked at
    /// here is what the askings noted, which stays.)
    #[test]
    fn an_asking_begun_first_is_made_and_more_is_asked_for() {
        let fresh = || {
            // SAFETY: a fresh write-watched allocation, freed at the end.
            let p = unsafe {
                VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
            } as usize;
            assert_ne!(p, 0);
            p
        };
        let (p, q) = (fresh(), fresh());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE) as *mut u8).write_volatile(1) };
        ask_first(vec![(p, p + 4 * PAGE)]).wait();
        assert!(!journal().notes.range(p / PAGE + 1, p / PAGE + 2).is_empty(), "the first asking did not note the page written");
        // SAFETY: committed pages of ours.
        unsafe { ((q + 2 * PAGE) as *mut u8).write_volatile(1) };
        assert!(ask_more(&[(p, p + 4 * PAGE), (q, q + 4 * PAGE)]), "a range nobody held was not asked about");
        assert!(!journal().notes.range(q / PAGE + 2, q / PAGE + 3).is_empty(), "the asking for more did not note the page written");
        ask_once_end();
        // An empty asking is waited for like any other.
        ask_first(Vec::new()).wait();
        ask_once_end();
        // SAFETY: our allocations.
        unsafe {
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
            VirtualFree(q as *mut _, 0, MEM_RELEASE);
        }
    }

    /// The runtime maps a view and then tells us; a copy's refresh on another
    /// thread can guard it in between. Told, it is the title's to write: a
    /// block left read-only and forgotten would be an access violation in
    /// title code.
    #[test]
    fn a_block_guarded_before_we_were_told_of_its_view_can_be_written() {
        let _one = crate::watch::one_view_test();
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write; the
        // view holds it.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK);
            CloseHandle(section);
            view
        };
        let at = view.Value as usize;
        assert_ne!(at, 0);
        // The refresh, then the runtime's word.
        assert!(guard(at, at + 2 * BLOCK).is_some());
        assert!(!writable(at) && !writable(at + BLOCK));
        title_memory_remapped(at, 2 * BLOCK);
        assert!(writable(at) && writable(at + BLOCK));
        // SAFETY: inside the view; a fault would end the process.
        unsafe { ((at + 8) as *mut u8).write_volatile(1) };
        // Looked at again as new, and guarded again.
        assert_eq!(
            guard(at, at + 2 * BLOCK),
            Some((at / PAGE..at / PAGE + 2).collect())
        );
        assert!(!writable(at));
        title_memory_unmapping(at, 2 * BLOCK);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(view) };
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        let _one = crate::watch::one_view_test();
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_unmapping(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        let _one = crate::watch::one_view_test();
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        title_memory_unmapping(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        let _one = crate::watch::one_view_test();
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        title_memory_unmapping(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        let _one = crate::watch::one_view_test();
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        let _one = crate::watch::one_view_test();
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        let _one = crate::watch::one_view_test();
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        let _one = crate::watch::one_view_test();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A write opens the view it goes through and no other: the others stay
    /// read-only, are told as written, and take writes of their own.
    #[test]
    fn a_write_opens_the_view_it_goes_through() {
        let _one = crate::watch::one_view_test();
        let (section, one, two) = two_views(BLOCK);
        let page = |view: usize| view / PAGE;
        for view in [one, two] {
            assert_eq!(guard(view, view + BLOCK), Some(vec![page(view)]));
        }
        // SAFETY (all): a view, opened by the handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert!(writable(two) && !writable(one));
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert!(writable(two) && writable(one));
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        // Written at the view that was left as it was, after the other.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]), "not clean, though never opened");
        assert!(!writable(two) && !writable(one));
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        // Of a page that was told at one view, a kernel write to the other
        // finds it open.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert!(!writable(one));
        writing(one, 8);
        assert!(writable(one));
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write over a run of guarded blocks of one view opens them with one
    /// `VirtualProtect` call, not one a block.
    #[test]
    fn a_write_over_a_guarded_run_opens_it_in_one_call() {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};
        const BLOCKS: usize = 8;
        // SAFETY: a pagefile section, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (BLOCKS * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCKS * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + BLOCKS * BLOCK).is_some());
        assert!((0..BLOCKS).all(|b| !writable(view + b * BLOCK)), "guarded");
        TEST_OPEN_CALLS.with(|c| c.set(0));
        writing(view, BLOCKS * BLOCK);
        assert_eq!(TEST_OPEN_CALLS.with(|c| c.get()), 1, "one call for the run");
        assert!((0..BLOCKS).all(|b| writable(view + b * BLOCK)), "every block writable again");
        title_memory_unmapping(view, BLOCKS * BLOCK);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ })
        };
    }

    /// A block noted guarded but not made read-only yet is told as written,
    /// and one opened before its protection is made is writable after.
    #[test]
    fn a_block_opened_before_its_late_protection_is_writable_after() {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};
        if !protect_later() {
            return;
        }
        // SAFETY: a pagefile section of a block, mapped read-write.
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK)
        };
        let at = view.Value as usize;
        assert_ne!(at, 0);
        let block = at / BLOCK;
        assert!(guard(at, at + BLOCK).is_some());
        assert!(!writable(at), "guarded");
        assert!(!PENDING.any(block, block + 1));
        // SAFETY: inside the view; the handler opens it.
        unsafe { ((at + 8) as *mut u8).write_volatile(1) };
        let mut later = Vec::new();
        {
            let mut g = guards();
            loose(&mut g, block);
            g.protected.remove(block);
            g.clean.remove(block);
            PROTECTED.clear(block);
            pend(&mut g, block, block + 1, &mut later);
        }
        assert_eq!(later, [(block, block + 1)]);
        assert!(writable(at), "not read-only yet");
        assert!(written_since_taken(at, BLOCK));
        open(&mut guards(), at, at + 1);
        assert!(read_only(block, block + 1, 1));
        protected_later(&mut guards(), block, block + 1, true);
        assert!(writable(at), "opened in between");
        assert!(!PENDING.any(block, block + 1));
        assert!(!PROTECTED.any(block, block + 1));
        title_memory_unmapping(at, BLOCK);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(view) };
    }

    /// A write to a block opened and then made read-only by a late
    /// protection, before that one has the lock again, waits for it and
    /// lands.
    #[test]
    fn a_write_between_a_late_protection_and_its_notes_waits() {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};
        if !protect_later() {
            return;
        }
        // SAFETY: a pagefile section of a block, mapped read-write.
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK)
        };
        let at = view.Value as usize;
        assert_ne!(at, 0);
        let block = at / BLOCK;
        install();
        let mut later = Vec::new();
        pend(&mut guards(), block, block + 1, &mut later);
        open(&mut guards(), at, at + 1);
        assert!(!PROTECTED.any(block, block + 1));
        assert!(read_only(block, block + 1, 1));
        let (sent, came) = std::sync::mpsc::channel();
        let writer = std::thread::spawn(move || {
            // SAFETY: inside the view; the handler waits for it to open.
            unsafe { ((at + 8) as *mut u8).write_volatile(7) };
            sent.send(()).unwrap();
        });
        let early = came.recv_timeout(std::time::Duration::from_millis(100)).is_ok();
        protected_later(&mut guards(), block, block + 1, true);
        let landed = came.recv_timeout(std::time::Duration::from_secs(10)).is_ok();
        writer.join().unwrap();
        assert!(!early, "the write waited");
        assert!(landed, "and landed after");
        // SAFETY: inside the view.
        assert_eq!(unsafe { ((at + 8) as *const u8).read_volatile() }, 7);
        // A block opened while a run has it is not taken into another.
        pend(&mut guards(), block, block + 1, &mut later);
        open(&mut guards(), at, at + 1);
        assert_eq!(guard(at, at + BLOCK), Some(vec![block]));
        assert!(writable(at), "not guarded again while the first run has it");
        assert!(read_only(block, block + 1, 1));
        protected_later(&mut guards(), block, block + 1, true);
        assert!(writable(at));
        title_memory_unmapping(at, BLOCK);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(view) };
    }

    /// A write fault opens a guarded block while another thread holds the
    /// guards' lock, and the next asking tells it as written.
    #[test]
    fn a_write_fault_opens_its_block_with_the_lock_held_elsewhere() {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};
        // SAFETY: a pagefile section of a block, mapped read-write.
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK)
        };
        let at = view.Value as usize;
        assert_ne!(at, 0);
        assert!(guard(at, at + BLOCK).is_some());
        assert!(!writable(at), "guarded");
        let held = guards();
        let (sent, came) = std::sync::mpsc::channel();
        let writer = std::thread::spawn(move || {
            // SAFETY: inside the view; the handler opens it.
            unsafe { ((at + 8) as *mut u8).write_volatile(1) };
            sent.send(()).unwrap();
        });
        let done = came.recv_timeout(std::time::Duration::from_secs(10)).is_ok();
        let noted = held.protected.contains(at / BLOCK);
        drop(held);
        writer.join().unwrap();
        assert!(done, "the write waited for the lock");
        assert!(noted, "noted guarded until a holder of the lock looks");
        assert!(writable(at));
        assert_eq!(guard(at, at + BLOCK), Some(vec![at / PAGE]));
        assert!(!LOOSE.any(at / BLOCK, at / BLOCK + 1));
        title_memory_unmapping(at, BLOCK);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(view) };
    }

    /// Blocks next to each other but of separate allocations (one call over
    /// them fails) are opened one at a time, all of them.
    #[test]
    fn a_guarded_run_over_separate_allocations_opens_block_by_block() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MapViewOfFile3, UnmapViewOfFile2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let p = placeholder(2);
        // SAFETY: a section of one block in place of each placeholder.
        let views = [0, 1].map(|i| unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile3(
                section,
                GetCurrentProcess(),
                (p + i * BLOCK) as *const _,
                0,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        });
        assert!(views.iter().enumerate().all(|(i, v)| v.Value as usize == p + i * BLOCK));
        assert!(guard(p, p + 2 * BLOCK).is_some());
        assert!(!writable(p) && !writable(p + BLOCK), "guarded");
        TEST_OPEN_CALLS.with(|c| c.set(0));
        writing(p, 2 * BLOCK);
        assert_eq!(TEST_OPEN_CALLS.with(|c| c.get()), 2, "one call a block");
        assert!(writable(p) && writable(p + BLOCK), "both writable again");
        title_memory_unmapping(p, 2 * BLOCK);
        // SAFETY: our views and the placeholders left by them.
        unsafe {
            for view in views {
                UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
            }
            for i in 0..2 {
                VirtualFree((p + i * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
        }
    }

    /// A block the title has just written is not made read-only, and is told
    /// as written at each asking, at every view of it, until it has rested.
    #[test]
    fn a_block_just_written_rests() {
        let _one = crate::watch::one_view_test();
        let (section, one, two) = two_views(BLOCK);
        let page = one / PAGE;
        let rests = |rests: bool| REST_IN_TEST.set(std::time::Duration::from_secs(if rests { 3600 } else { 0 }));
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        rests(true);
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()), "nothing written: as it was");
        // SAFETY (all): a view, opened by the handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        for _ in 0..3 {
            assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
            assert!(writable(two), "no fault for the next write");
            assert!(writes(two, two + BLOCK, true).is_some() && !published(two, two + BLOCK) && !published(one, one + BLOCK));
            unsafe { ((two + 8) as *mut u8).write(1) };
        }
        rests(false);
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert!(!writable(two) && !writable(one));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        drop_views(section, &[one, two], BLOCK);
    }

    /// A block written again soon after it rested ([`hot`]) rests longer: no
    /// fault for its writes meanwhile, told as written at each asking, at
    /// every view, and guarded again when that is over.
    #[test]
    fn a_block_written_again_and_again_rests_longer() {
        let _one = crate::watch::one_view_test();
        const KEEP: std::time::Duration = std::time::Duration::from_millis(300);
        let (section, one, two) = two_views(BLOCK);
        let page = |view: usize| view / PAGE;
        HOT_IN_TEST.set(KEEP);
        for view in [one, two, one, two] {
            assert!(writes(view, view + BLOCK, true).is_some());
        }
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        // (Not opened lately as far as the guards know: another test's
        // memory may have been at the address.)
        guards().opened.remove(&(two / BLOCK));
        // SAFETY (all): a view, opened by the handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        // The first time: guarded again at the next asking (no rest here).
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        assert!(!writable(two));
        // Written again: it rests.
        unsafe { ((two + 8) as *mut u8).write(2) };
        for _ in 0..3 {
            assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
            assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]), "the other view is not clean while it rests");
            assert!(writable(two) && !writable(one), "no fault for the next write");
            unsafe { ((two + 8) as *mut u8).write(3) };
        }
        std::thread::sleep(KEEP + std::time::Duration::from_millis(50));
        HOT_IN_TEST.set(std::time::Duration::ZERO);
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        assert!(!writable(two) && !writable(one));
        assert_eq!(guard(two, two + BLOCK), Some(Vec::new()));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        drop_views(section, &[one, two], BLOCK);
    }

    /// What is put in beside a guard is told at every view, once, and the
    /// block stays read-only; until it has had its share.
    #[test]
    fn what_the_title_puts_in_beside_a_guard_is_told() {
        let _one = crate::watch::one_view_test();
        let (section, one, two) = two_views(BLOCK);
        crate::title_memory_section(section, BLOCK as u64, true);
        let page = |view: usize| view / PAGE;
        for view in [one, two] {
            assert_eq!(guard(view, view + BLOCK), Some(vec![page(view)]));
        }
        assert!(writes(one, one + BLOCK, true).is_some() && writes(two, two + BLOCK, true).is_some());
        put_count(one / BLOCK).store(0, std::sync::atomic::Ordering::Relaxed);
        let put = |byte: u8| loop {
            // SAFETY: a byte of the section's only page.
            let done = put_beside(one / BLOCK, 1, |delta| unsafe { ((one + 100).wrapping_add(delta) as *mut u8).write(byte) });
            // (Other tests hold changes of mapping.)
            if done != Beside::Cannot {
                return done;
            }
        };
        for round in 1..4 {
            assert_eq!(put(round), Beside::Done);
            // SAFETY: the view the title reads it by.
            assert_eq!(unsafe { ((two + 100) as *const u8).read_volatile() }, round);
            assert!(!writable(one) && !writable(two) && !published(one, one + BLOCK) && !published(two, two + BLOCK));
            for view in [two, one] {
                assert_eq!(guard(view, view + BLOCK), Some(vec![page(view)]));
                assert_eq!(guard(view, view + BLOCK), Some(Vec::new()));
            }
        }
        for _ in 0..crate::titlewrite::MOST {
            assert_eq!(put(9), Beside::Done);
        }
        assert_eq!(put(9), Beside::Full);
        // A refresh between gives it its share again.
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        assert_eq!(put(9), Beside::Done);
        // Opened by a fault, it is told by that.
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        // SAFETY: a view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(put_beside(one / BLOCK, 1, |_| panic!("it is not guarded")), Beside::Cannot);
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        drop_views(section, &[one, two], BLOCK);
        crate::title_memory_section_gone(section);
    }

    /// Bytes put in beside a guard leave no mark at the address once the
    /// memory is freed or no longer mapped there: the next memory at it would
    /// be told as written before anything is.
    #[test]
    fn what_was_put_beside_goes_with_the_memory() {
        let (section, one, two) = two_views(BLOCK);
        crate::title_memory_section(section, BLOCK as u64, true);
        for view in [one, two] {
            assert_eq!(guard(view, view + BLOCK), Some(vec![view / PAGE]));
        }
        assert!(writes(one, one + BLOCK, true).is_some() && writes(two, two + BLOCK, true).is_some());
        put_count(one / BLOCK).store(0, std::sync::atomic::Ordering::Relaxed);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            // SAFETY: a byte of the section's only page.
            match put_beside(one / BLOCK, 1, |delta| unsafe { ((one + 100).wrapping_add(delta) as *mut u8).write(7) }) {
                Beside::Done => break,
                // (Other tests hold changes of mapping.)
                Beside::Cannot if std::time::Instant::now() < deadline => std::hint::spin_loop(),
                other => panic!("put_beside: {other:?}"),
            }
        }
        let beside = |view: usize| BESIDE.any(view / BLOCK, view / BLOCK + 1);
        assert!(beside(one) && beside(two));
        title_memory_freed(two, BLOCK);
        assert!(!beside(two), "the freed view keeps its mark");
        assert!(beside(one), "the other view lost its mark");
        drop_views(section, &[one, two], BLOCK);
        assert!(!beside(one));
        crate::title_memory_section_gone(section);
    }

    /// Stores that fault, and copies, as the title makes them: what is there
    /// after, the block's guard and what the next asking says.
    #[test]
    fn stores_and_copies_of_the_title_leave_the_guard() {
        let _one = crate::watch::one_view_test();
        use std::sync::atomic::{AtomicU32, Ordering};
        unsafe extern "system-unwind" fn real(to: *mut u8, from: *const u8, n: usize) -> *mut u8 {
            // SAFETY: by the caller.
            unsafe { std::ptr::copy(from, to, n) };
            to
        }
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        crate::title_memory_section(section, SIZE as u64, true);
        let pages = |view: usize| vec![view / PAGE, view / PAGE + 1];
        assert_eq!(guard(one, one + SIZE), Some(pages(one)));
        assert_eq!(guard(two, two + SIZE), Some(pages(two)));
        let copy = crate::titlewrite::import("memcpy", real as *const () as usize);
        crate::titlewrite::ON_IN_TEST.set(true);
        let copy = if copy == real as *const () as usize {
            crate::titlewrite::copy_for_test(real as *const () as usize)
        } else {
            copy
        };
        // SAFETY: of that signature.
        let copy: unsafe extern "system-unwind" fn(*mut u8, *const u8, usize) -> *mut u8 = unsafe { std::mem::transmute(copy) };
        // No change of mapping meanwhile but the other tests' moments of it:
        // a store that meets one opens the block, as before.
        let mut kept = 0;
        let quiet = || {
            while title_memory_reading_now().is_none() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        for round in 1..200u32 {
            quiet();
            // SAFETY (all): words of the view.
            unsafe { ((one + 64) as *mut u32).write_volatile(round) };
            let was = unsafe { AtomicU32::from_ptr((two + 64) as *mut u32) }.fetch_sub(1, Ordering::SeqCst);
            assert_eq!((was, unsafe { ((one + 64) as *const u32).read_volatile() }), (round, round - 1));
            let bytes = vec![round as u8; 3000 + round as usize];
            // Over the end of the first block.
            assert_eq!(unsafe { copy((one + BLOCK - 1000) as *mut u8, bytes.as_ptr(), bytes.len()) } as usize, one + BLOCK - 1000);
            let there = unsafe { std::slice::from_raw_parts((two + BLOCK - 1000) as *const u8, bytes.len()) };
            assert!(there == bytes && unsafe { ((two + BLOCK + 2000 + round as usize) as *const u8).read_volatile() } != round as u8);
            kept += usize::from(!writable(one) && !writable(two));
            for view in [one, two] {
                assert_eq!(guard(view, view + SIZE), Some(pages(view)), "round {round}");
                assert_eq!(guard(view, view + SIZE), Some(Vec::new()), "round {round}");
            }
        }
        assert!(kept > 10, "the guard stayed {kept} times: {:?}", crate::titlewrite::DONE);
        // What is there already is not written: nothing to tell.
        for k in 0..10 {
            // SAFETY (all): as above.
            unsafe { ((one + 64) as *mut u32).write_volatile(198) };
            let same = unsafe { std::slice::from_raw_parts((one + BLOCK - 900) as *const u8, 1500) }.to_vec();
            unsafe { copy((one + BLOCK - 900) as *mut u8, same.as_ptr(), same.len()) };
            assert!(!writable(one) && !writable(two), "{k}");
        }
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        // Many stores to a block: it is opened.
        quiet();
        for k in 0..20 {
            unsafe { ((one + 128 + k * 4) as *mut u32).write_volatile(7) };
        }
        assert!(writable(one));
        crate::titlewrite::ON_IN_TEST.set(false);
        drop_views(section, &[one, two], SIZE);
        crate::title_memory_section_gone(section);
    }

    /// Of a page taken before, what is taken again is the kernel's pages
    /// written since; where they are not known, all of it.
    #[test]
    fn what_is_taken_again_is_what_was_written() {
        let _one = crate::watch::one_view_test();
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        let write = |at: usize| {
            // SAFETY: committed pages of ours.
            unsafe { ((p + at) as *mut u8).write_volatile(1) }
        };
        let tracked = track(p, 4 * PAGE, false).expect("watched");
        // (Other tests' memory was here.)
        (0..4).for_each(|page| write(page * PAGE + 5));
        assert_eq!(tracked.take(p, p + 4 * PAGE), [(p, p + 4 * PAGE)], "never taken");
        assert!(tracked.take(p, p + 4 * PAGE).is_empty());
        write(PAGE + 5000);
        write(PAGE + 8192);
        write(PAGE + 65535);
        write(2 * PAGE);
        assert_eq!(
            tracked.take(p, p + 4 * PAGE),
            [(p + PAGE + 4096, p + PAGE + 3 * 4096), (p + 2 * PAGE - 4096, p + 2 * PAGE + 4096)]
        );
        write(2 * PAGE + 70);
        // Two askings' writes are in what is taken after both.
        assert!(writes(p, p + 4 * PAGE, false).is_some());
        write(2 * PAGE + 9000);
        assert_eq!(tracked.take(p, p + 4 * PAGE), [(p + 2 * PAGE, p + 2 * PAGE + 4096), (p + 2 * PAGE + 8192, p + 2 * PAGE + 12288)]);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// The frames pages were last landed in.
    #[test]
    fn landed_pages_are_known_by_their_frame() {
        static LANDED: Landed = Landed([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; Landed::PIECES]);
        let landed = &LANDED;
        for page in [0, 5, Landed::PAGES - 1, Landed::PAGES, 7 * Landed::PAGES + 3] {
            assert!(!landed.since(page, 0, 100));
            landed.note(page, 100);
            assert!(landed.since(page, 100, 100) && landed.since(page, 0, 100) && landed.since(page, 90, 110));
            assert!(!landed.since(page, 101, 110));
            landed.note(page, 0x1_0000_0005);
            assert!(landed.since(page, 0x1_0000_0000, 0x1_0000_0007) && !landed.since(page, 0x1_0000_0006, 0x1_0000_0007));
            assert!(!landed.since(page + 1, 0, 0x1_0000_0007));
        }
        landed.note(usize::MAX / 4096, 1);
        assert!(!landed.since(usize::MAX / 4096, 0, 1));
    }

    /// The set of blocks against another of the same.
    #[test]
    fn blocks_are_a_set() {
        let (mut set, mut other) = (Blocks::new(), BTreeSet::new());
        let mut seed = 1u64;
        let mut next = |n: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        for round in 0..40_000 {
            // About the ends of pieces and words.
            let block = [0, 63, Blocks::IN_PIECE, 3 * Blocks::IN_PIECE - 70][next(4)] + next(140);
            if next(3) == 0 {
                assert_eq!(set.remove(block), other.remove(&block));
            } else {
                assert_eq!(set.insert(block), other.insert(block));
            }
            assert_eq!(set.contains(block), other.contains(&block));
            if round % 64 == 0 {
                let a = [0, 63, Blocks::IN_PIECE, 3 * Blocks::IN_PIECE - 70][next(4)] + next(140);
                let b = a + [next(10), next(300), 2 * Blocks::IN_PIECE + next(100)][next(3)];
                assert!(set.range(a..b).eq(other.range(a..b).copied()), "{a}..{b}");
                assert!(set.range(a..).eq(other.range(a..).copied()) && set.range(..).eq(other.iter().copied()));
                assert!(set.range(a..=b).eq(other.range(a..=b).copied()));
                assert_eq!(set.len(), other.len());
            }
        }
        assert_eq!(set.range(5..usize::MAX).next(), other.range(5..).next().copied());
    }

    /// What is known of a block by its bit is so at every view of its page:
    /// the write through one takes the bits of all.
    #[test]
    fn bits_go_at_every_view() {
        let _one = crate::watch::one_view_test();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let ask = |at: usize| writes(at, at + SIZE, true).expect("a view");
        for at in [one, two, one, two] {
            ask(at);
        }
        assert!(published(one, one + SIZE) && published(two, two + SIZE));
        let (before, _) = ask(two);
        // SAFETY: the first view, opened by the handler.
        unsafe { ((one + BLOCK + 8) as *mut u8).write(1) };
        assert!(published(one, one + BLOCK) && published(two, two + BLOCK), "the other block is as it was");
        assert!(!published(one + BLOCK, one + SIZE) && !published(two + BLOCK, two + SIZE));
        let told = |at: usize| ask(at).1.iter().find(|w| w.0 == (at + BLOCK) / PAGE).map_or(0, |w| w.1);
        assert!(told(two) > before && told(one) > before);
        drop_views(section, &[one, two], SIZE);
    }

    /// While the kernel writes through one view, the same page at another
    /// is told as written at every asking, and the first is not guarded.
    #[test]
    fn what_the_kernel_writes_through_is_not_guarded_for_another_view() {
        let _one = crate::watch::one_view_test();
        let (section, one, two) = two_views(BLOCK);
        let page = one / PAGE;
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        let writing = kernel_writing(two, 100);
        for _ in 0..3 {
            assert_eq!(guard(one, one + BLOCK), Some(vec![page]), "the kernel may be writing it");
            assert!(writable(two));
        }
        assert!(!written_since_taken(one, 8), "by whom is not known");
        drop(writing);
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert!(!writable(two));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        // SAFETY: opened by the handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        drop_views(section, &[one, two], BLOCK);
    }

    /// What shaders wrote, put in past the guards and told at one view, is
    /// told at the others, whose copies have not got it; and so is what we
    /// wrote while a submission's one asking stands.
    #[test]
    fn what_is_put_in_beside_the_guards_is_told_at_every_view() {
        let _one = crate::watch::one_view_test();
        let (section, one, two) = two_views(BLOCK);
        writes(two, two + BLOCK, true).expect("a view");
        let (before, _) = writes(two, two + BLOCK, true).expect("a view");
        assert!(published(two, two + BLOCK));
        landed_beside(&[one / 4096 + 3], &[(one + 3 * 4096, 16)]);
        let (_, told) = writes(two, two + BLOCK, true).expect("a view");
        assert!(told.iter().any(|w| w.0 == two / PAGE && w.1 > before), "{told:?} after {before}");
        assert!(landed_lately(two + 3 * 4096, 16, 1) && !landed_lately(two + 4 * 4096, 16, 1));
        drop_views(section, &[one, two], BLOCK);
    }

    /// A view is cut in two by the runtime: until that is done, its pages
    /// count as written wherever else they are shown.
    #[test]
    fn a_view_being_split_leaves_its_pages_unguarded_elsewhere() {
        let _one = crate::watch::one_view_test();
        const SIZE: usize = 3 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages: Vec<usize> = (0..3).map(|k| one / PAGE + k).collect();
        assert_eq!(guard(one, one + SIZE), Some(pages.clone()));
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        let changing = title_memory_remapping();
        title_memory_splitting(two, SIZE);
        title_memory_unmapping(two, SIZE);
        for _ in 0..2 {
            assert_eq!(guard(one, one + SIZE), Some(pages.clone()), "written through the part that stays");
        }
        // The first and last page stay mapped.
        for k in [0, 2] {
            title_memory_remapped(two + k * BLOCK, BLOCK);
            crate::inplace::title_memory_view(two + k * BLOCK, BLOCK, section, (k * BLOCK) as u64);
        }
        drop(changing);
        assert_eq!(guard(one, one + SIZE), Some(pages));
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert!(!writable(two) && writable(two + BLOCK) && !writable(two + 2 * BLOCK));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let _one = crate::watch::one_view_test();
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let _one = crate::watch::one_view_test();
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        let _one = crate::watch::one_view_test();
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }
}

#[cfg(test)]
mod many_askers_tests {
    use super::{PAGE, writes};
    use windows_sys::Win32::System::Memory::*;

    /// Threads that write pages and ask about overlapping ranges at the same
    /// time: whoever wrote a page before asking is told of that write (by
    /// its own asking or by one that took the kernel's answer before it).
    #[test]
    fn no_write_is_lost_between_askers() {
        let _one = crate::watch::one_view_test();
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const PAGES: usize = 48;
        const THREADS: usize = 8;
        // SAFETY: a reservation of our own, freed at the end.
        let at = unsafe {
            VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert!(at != 0 && at.is_multiple_of(PAGE));
        std::thread::scope(|s| {
            for t in 0..THREADS {
                s.spawn(move || {
                    let mut last = [0u64; PAGES];
                    let mut seed = t as u64 * 7919 + 1;
                    let mut next = |n: usize| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (seed >> 33) as usize % n
                    };
                    for _ in 0..4000 {
                        let page = next(PAGES);
                        let from = page - next(page.min(12) + 1);
                        let to = page + 1 + next((PAGES - page).min(12));
                        // SAFETY: committed memory of ours; what the racing
                        // writers leave in it does not matter.
                        unsafe { ((at + page * PAGE + t * 8) as *mut u64).write_volatile(1) };
                        let (now, written) = writes(at + from * PAGE, at + to * PAGE, false).expect("write-watched");
                        let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                        assert!(number > last[page], "page {page}: {number} is not after {}", last[page]);
                        for l in &mut last[from..to] {
                            *l = now;
                        }
                    }
                });
            }
        });
        // SAFETY: the reservation made above.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }

    /// The same of view memory, which is guarded: what its bits answer
    /// misses no write either, however often blocks are opened and guarded
    /// again between.
    #[test]
    fn no_write_to_views_is_lost_between_askers() {
        let _one = crate::watch::one_view_test();
        writes_to_views_are_not_lost::<1>(0);
    }

    /// The same with blocks that rest after a write ([`super::rest`]).
    #[test]
    fn no_write_to_resting_views_is_lost_between_askers() {
        let _one = crate::watch::one_view_test();
        writes_to_views_are_not_lost::<1>(50);
        writes_to_views_are_not_lost::<3>(50);
    }

    /// The same of pages shown at three addresses, each of which has copies
    /// of its own: a write through any of them, or put past the guards, is
    /// told at all. A fourth view comes and goes meanwhile, and is written
    /// too. (The first assertion fails within a few rounds if a guard holds
    /// for one address only, `VICTORY_VIEW_GUARD=0`.)
    #[test]
    fn no_write_through_another_view_is_lost_between_askers() {
        let _one = crate::watch::one_view_test();
        writes_to_views_are_not_lost::<3>(0);
    }

    fn writes_to_views_are_not_lost<const VIEWS: usize>(rest: u64) {
        let rest = std::time::Duration::from_micros(rest);
        const PAGES: usize = 12;
        const THREADS: usize = 8;
        // SAFETY: views of a section of our own.
        let (section, all): (usize, [usize; VIEWS]) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (PAGES * PAGE) as u32,
                std::ptr::null(),
            );
            crate::title_memory_section(section as usize, (PAGES * PAGE) as u64, true);
            let all = std::array::from_fn(|_| {
                let at = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGES * PAGE).Value as usize;
                assert!(at != 0 && at.is_multiple_of(PAGE));
                super::title_memory_remapped(at, PAGES * PAGE);
                crate::title_memory_view(at, PAGES * PAGE, section as usize, 0);
                at
            });
            (section as usize, all)
        };
        let by_bits = std::sync::atomic::AtomicUsize::new(0);
        let past = std::sync::atomic::AtomicUsize::new(0);
        let title_beside = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            if VIEWS > 1 {
                s.spawn(move || {
                    super::REST_IN_TEST.set(rest);
                    let mut last = [0u64; VIEWS];
                    for round in 0..300usize {
                        let (page, asked) = (round * 5 % PAGES, round % VIEWS);
                        // SAFETY (all): a view of one page of the section,
                        // this thread's alone.
                        let one = unsafe {
                            MapViewOfFile(section as _, FILE_MAP_WRITE, 0, (page * PAGE) as u32, PAGE).Value as usize
                        };
                        assert_ne!(one, 0);
                        super::title_memory_remapped(one, PAGE);
                        crate::title_memory_view(one, PAGE, section, (page * PAGE) as u64);
                        unsafe { ((one + THREADS * 8) as *mut u64).write_volatile(round as u64) };
                        let (now, written) =
                            writes(all[asked] + page * PAGE, all[asked] + (page + 1) * PAGE, true).expect("a view");
                        let number = written.first().map_or(0, |w| w.1);
                        assert!(number > last[asked], "page {page}, through a new view: {number} is not after {}", last[asked]);
                        last[asked] = now.max(last[asked]);
                        {
                            let _changing = super::title_memory_remapping();
                            super::title_memory_unmapping(one, PAGE);
                            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: one as *mut _ }) };
                        }
                        last = [last[asked]; VIEWS];
                    }
                });
            }
            for t in 0..THREADS {
                let (by_bits, past, title_beside) = (&by_bits, &past, &title_beside);
                s.spawn(move || {
                    super::REST_IN_TEST.set(rest);
                    let mut last = [[0u64; PAGES]; VIEWS];
                    let mut seed = t as u64 * 7919 + 1;
                    let mut next = |n: usize| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (seed >> 33) as usize % n
                    };
                    for _ in 0..500 {
                        let page = next(PAGES);
                        let from = page - next(page.min(3) + 1);
                        let to = page + 1 + next((PAGES - page).min(3));
                        // Written through one view, asked about at another.
                        let (through, asked) = (all[next(VIEWS)], next(VIEWS));
                        #[allow(unused_assignments)]
                        let mut how = 0;
                        let (at, last) = (all[asked], &mut last[asked]);
                        if next(4) != 0 {
                            // As in the test above, or after a word of ours,
                            // or past the guard.
                            let word = through + page * PAGE + t * 8;
                            // SAFETY (both): as above.
                            // (Waited for: other tests hold changes of mapping.)
                            // (Some as a copy of the title's, or a store of it, does.)
                            if next(3) == 0
                                && super::put_beside(word / PAGE, 1, |delta| unsafe {
                                    (word.wrapping_add(delta) as *mut u64).write_volatile(3);
                                }) == super::Beside::Done
                            {
                                title_beside.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                let (now, written) = writes(at + from * PAGE, at + to * PAGE, true).expect("a view");
                                let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                                assert!(number > last[page], "page {page}, beside: {number} is not after {}", last[page]);
                                for l in &mut last[from..to] {
                                    *l = now;
                                }
                                continue;
                            }
                            let reading = (next(3) == 0).then(super::title_memory_reading);
                            how = usize::from(reading.is_some());
                            let beside = reading.as_ref().and_then(|_| crate::inplace::beside(word, 8));
                            if let Some(to) = beside {
                                unsafe { (to as *mut u64).write_volatile(2) };
                                super::landed_beside(&[word / 4096], &[(word, 8)]);
                                past.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                assert_eq!(unsafe { (word as *const u64).read_volatile() }, 2, "put elsewhere");
                            } else {
                                how += 2;
                                if next(4) == 0 {
                                    super::writing(word, 8);
                                    how += 4;
                                }
                                unsafe { (word as *mut u64).write_volatile(1) };
                            }
                        } else {
                            // Nothing written: what is asked counts as before.
                            by_bits.fetch_add(
                                usize::from(super::published(at + from * PAGE, at + to * PAGE)),
                                std::sync::atomic::Ordering::Relaxed,
                            );
                            let (now, _) = writes(at + from * PAGE, at + to * PAGE, true).expect("a view");
                            for l in &mut last[from..to] {
                                *l = now;
                            }
                            continue;
                        }
                        let (now, written) = writes(at + from * PAGE, at + to * PAGE, true).expect("a view");
                        let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                        assert!(
                            number > last[page],
                            "{VIEWS} views, page {page}, through {:x} ({how}), asked at {:x}: {number} is not after {}",
                            through / PAGE + page,
                            at / PAGE + page,
                            last[page]
                        );
                        for l in &mut last[from..to] {
                            *l = now;
                        }
                    }
                });
            }
        });
        assert!(by_bits.into_inner() > 0, "the bits answered none");
        assert!(past.into_inner() > 0, "nothing was put past the guards");
        assert!(title_beside.into_inner() > 0, "nothing of the title's was put in beside the guards");
        // (The view stays: at its address other tests' memory would be told
        // of these writes.)
    }

    /// A range of a placeholder with views in it is known whole, its holes
    /// among it, until something is mapped into one, unmapped from it or
    /// allocated there. A hole that is no placeholder's is asked about.
    #[test]
    fn holes_are_known_until_the_runtime_tells() {
        let _one = crate::watch::one_view_test();
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const BLOCKS: usize = 8;
        // SAFETY (all): a placeholder and a section of our own, which stay.
        let (at, section) = unsafe {
            let at = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                BLOCKS * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (BLOCKS * PAGE) as u32, std::ptr::null());
            (at, section)
        };
        let map = |block: usize| unsafe {
            let to = at + block * PAGE;
            VirtualFree(to as *mut _, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            let view = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                to as *const _,
                (block * PAGE) as u64,
                PAGE,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(view.Value as usize, to);
            super::title_memory_remapped(to, PAGE);
        };
        map(1);
        map(5);
        assert!(super::placeholder(at / PAGE) && super::placeholder(at / PAGE + 7) && !super::placeholder(at / PAGE + 5));
        let end = at + BLOCKS * PAGE;
        let ask = |since: u64| -> (u64, Vec<usize>) {
            let (now, written) = writes(at, end, true).expect("a view");
            (now, written.into_iter().filter(|w| w.1 > since).map(|w| w.0 - at / PAGE).collect())
        };
        let (first, all) = ask(0);
        assert_eq!(all, (0..BLOCKS).collect::<Vec<_>>());
        assert!(super::published(at, end), "with its holes");
        // Written; mapped into; unmapped.
        unsafe { ((at + 5 * PAGE) as *mut u8).write_volatile(1) };
        assert!(!super::published(at, end));
        let (second, all) = ask(first);
        assert_eq!(all, [5]);
        map(2);
        assert!(!super::published(at, end));
        let (third, all) = ask(second);
        assert_eq!(all, [2]);
        unsafe { ((at + 2 * PAGE) as *mut u8).write_volatile(1) };
        let (fourth, all) = ask(third);
        assert_eq!(all, [2]);
        super::title_memory_unmapping(at + 5 * PAGE, PAGE);
        unsafe {
            let view = MEMORY_MAPPED_VIEW_ADDRESS { Value: (at + 5 * PAGE) as *mut _ };
            UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
        }
        let (fifth, all) = ask(fourth);
        assert_eq!(all, [5]);
        assert!(super::published(at, end));
        // Told of memory there, though there is none: asked about again.
        super::title_memory_allocated(at, PAGE);
        assert!(!super::published(at, end));
        let (_, all) = ask(fifth);
        assert_eq!(all, []);
        // A reservation of the ordinary kind is no placeholder.
        let plain = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert!(!super::placeholder(plain / PAGE));
        super::look_at_holes();
        assert!(!super::published(at, end), "block 0 stays asked about");
        assert!(super::published(at + PAGE, end), "the others are as they were");
        super::title_memory_unmapping(at, BLOCKS * PAGE);
    }

    /// Pages mapped into a placeholder, written and unmapped again, each by a
    /// thread of its own as the runtime does it, while others ask about all
    /// of it: what was written, and that a page came or went, is told to
    /// whoever asks after it, and blocks are known by their bits meanwhile.
    #[test]
    fn no_mapping_is_lost_between_askers() {
        let _one = crate::watch::one_view_test();
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const BLOCKS: usize = 12;
        // SAFETY (all): a placeholder and a section of our own, which stay.
        let (at, section) = unsafe {
            let at = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                BLOCKS * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            for block in 0..BLOCKS - 1 {
                VirtualFree((at + block * PAGE) as *mut _, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            }
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (BLOCKS * PAGE) as u32, std::ptr::null());
            (at, section as usize)
        };
        let end = at + BLOCKS * PAGE;
        let (stop, by_bits) = (AtomicBool::new(false), AtomicUsize::new(0));
        let number = |written: &[(usize, u64)], block: usize| written.iter().find(|w| w.0 == at / PAGE + block).map_or(0, |w| w.1);
        std::thread::scope(|s| {
            let mappers: Vec<_> = (0..4usize)
                .map(|t| {
                    let number = &number;
                    s.spawn(move || {
                        for round in 0..120 {
                            let block = t * 3 + round % 3;
                            let to = at + block * PAGE;
                            let (before, _) = writes(at, end, true).expect("a view");
                            unsafe {
                                let view = MapViewOfFile3(
                                    section as _,
                                    GetCurrentProcess(),
                                    to as *const _,
                                    (block * PAGE) as u64,
                                    PAGE,
                                    MEM_REPLACE_PLACEHOLDER,
                                    PAGE_READWRITE,
                                    std::ptr::null_mut(),
                                    0,
                                );
                                assert_eq!(view.Value as usize, to);
                                super::title_memory_remapped(to, PAGE);
                                (to as *mut u8).write_volatile(round as u8);
                            }
                            let (mapped, written) = writes(at, end, true).expect("a view");
                            assert!(number(&written, block) > before, "mapped and written: not told");
                            unsafe { ((to + 9) as *mut u8).write_volatile(round as u8) };
                            let (again, written) = writes(to, to + PAGE, true).expect("a view");
                            assert!(number(&written, block) > mapped, "written again: not told");
                            super::title_memory_unmapping(to, PAGE);
                            unsafe {
                                let view = MEMORY_MAPPED_VIEW_ADDRESS { Value: to as *mut _ };
                                assert_ne!(UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER), 0);
                            }
                            let _ = again;
                        }
                    })
                })
                .collect();
            for t in 0..3usize {
                let (stop, by_bits) = (&stop, &by_bits);
                s.spawn(move || {
                    let mut k = t;
                    while !stop.load(Ordering::Relaxed) {
                        k = (k * 5 + 3) % 64;
                        let (from, to) = (at + k % 6 * PAGE, end - k / 8 % 5 * PAGE);
                        by_bits.fetch_add(usize::from(super::published(from, to)), Ordering::Relaxed);
                        assert!(writes(from, to, true).is_some());
                    }
                });
            }
            let failed = mappers.into_iter().map(|m| m.join()).filter(Result::is_err).count();
            stop.store(true, Ordering::Relaxed);
            assert_eq!(failed, 0);
        });
        assert!(by_bits.into_inner() > 0, "the bits answered none");
        // All unmapped: known whole, and nothing but holes.
        writes(at, end, true).expect("a view");
        assert!(super::published(at, end));
        super::title_memory_unmapping(at, BLOCKS * PAGE);
    }

    /// A block's bit goes with everything that makes it the title's to write,
    /// or forgets it.
    #[test]
    fn bits_go_with_the_guard() {
        let _one = crate::watch::one_view_test();
        // SAFETY: a view of a section of our own, which stays.
        let at = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (2 * PAGE) as u32, std::ptr::null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * PAGE).Value as usize
        };
        let end = at + 2 * PAGE;
        let takes: [&dyn Fn(); 4] = [
            &|| super::title_memory_remapped(at, PAGE),
            // SAFETY: the view.
            &|| unsafe { ((at + 8) as *mut u8).write_volatile(1) },
            &|| super::writing(at, 1),
            // (Told again, as of a view mapped there next: it is read-only.)
            &|| {
                super::title_memory_unmapping(at, PAGE);
                assert!(!super::published(at, end), "forgotten");
                super::title_memory_remapped(at, PAGE);
                let mut old = 0;
                // SAFETY: the view.
                unsafe { VirtualProtect(at as *const _, PAGE, PAGE_READWRITE, &mut old) };
            },
        ];
        for (i, take) in takes.iter().enumerate() {
            writes(at, end, true).expect("a view");
            assert!(super::published(at, end), "{i}");
            take();
            assert!(!super::published(at, end), "{i}");
            assert!(super::published(at + PAGE, end), "{i}: the other block");
        }
        super::title_memory_unmapping(at, 2 * PAGE);
    }

    /// A range's bits are all of its blocks', whatever lies beside them.
    #[test]
    fn bits_of_ranges() {
        static BITS: super::Bits = super::Bits::new();
        let blocks = [0usize, 1, 63, 64, 65, 127, 128, 200, 65535, 65536, 65537, 1 << 20, (1 << 31) - 1];
        for &b in &blocks {
            assert!(!BITS.all(b, b + 1));
            BITS.set(b);
            assert!(BITS.all(b, b + 1));
        }
        assert!(BITS.all(0, 2) && BITS.all(63, 66) && BITS.all(127, 129) && BITS.all(65535, 65538));
        assert!(!BITS.all(0, 3) && !BITS.all(62, 64) && !BITS.all(65534, 65536) && !BITS.all(5, 5));
        for b in 300..700 {
            BITS.set(b);
        }
        assert!(BITS.all(300, 700) && !BITS.all(299, 700) && !BITS.all(300, 701));
        BITS.clear(511);
        assert!(!BITS.all(300, 700) && BITS.all(300, 511) && BITS.all(512, 700));
        assert!(!BITS.all(1 << 40, (1 << 40) + 1), "outside the address space");
    }
}

/// Renderer port #7 ([`port_journal`]): the journal's notes in parts.
#[cfg(test)]
mod port7_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use windows_sys::Win32::System::Memory::*;

    fn rng(seed: &mut u64, n: usize) -> usize {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as usize % n.max(1)
    }

    /// The parts of a range, its pieces, and what is read of it a part at a
    /// time or holding its parts, over ranges across the parts' bounds.
    #[test]
    fn parts_pieces_and_ranges_agree() {
        const P: usize = 1 << SHARD_SHIFT;
        assert_eq!(shards_of(0, 0), 0);
        assert_eq!(shards_of(0, 1), 1);
        assert_eq!(shards_of(P - 1, P + 1), 0b11);
        assert_eq!(shards_of(63 * P, 65 * P), 1 << 63 | 1);
        assert_eq!(shards_of(5, 5 + SHARDS * P), u64::MAX);
        assert_eq!(pieces(P - 2, 2 * P + 3), Some(vec![(P - 2, P), (P, 2 * P), (2 * P, 2 * P + 3)]));
        assert_eq!(pieces(0, 1 << 40), None);
        let notes = Notes::new();
        let mut want: BTreeMap<usize, u64> = BTreeMap::new();
        let mut seed = 7;
        for n in 1..4000u64 {
            let page = rng(&mut seed, 70 * P);
            let mut held = notes.hold(shards_of_pages([page]), true);
            held.note(page, WHOLE, n);
            want.insert(page, n);
        }
        for _ in 0..500 {
            let first = rng(&mut seed, 70 * P);
            let last = first + rng(&mut seed, 3 * P);
            let expected: Vec<(usize, u64)> = want.range(first..last).map(|(&p, &w)| (p, w)).collect();
            assert_eq!(notes.range(first, last), expected, "{first}..{last} a part at a time");
            assert_eq!(notes.hold(shards_of(first, last), false).range(first, last), expected, "{first}..{last} held");
        }
        // The whole space: each part once.
        assert_eq!(notes.range(0, usize::MAX / 2), want.iter().map(|(&p, &w)| (p, w)).collect::<Vec<_>>());
        assert_eq!(notes.hold(u64::MAX, false).range(0, usize::MAX / 2).len(), want.len());
    }

    /// Noted out of order (shared noters' numbers may come in any order): a
    /// page's note, and its parts', only go up.
    #[test]
    fn notes_only_go_up() {
        let mut s = Shard::new();
        assert_eq!(s.note(3, WHOLE, 9), 0);
        assert_eq!(s.note(3, 0b11, 5), 9);
        assert_eq!(s.written[&3], 9);
        assert_eq!(s.parts_since(3, 8), WHOLE, "all of it at 9");
        assert_eq!(s.note(3, 0b100, 12), 9);
        assert_eq!(s.parts_since(3, 9), 0b100);
        // An older whole note leaves the newer part as it is.
        assert_eq!(s.note(3, WHOLE, 10), 12);
        assert_eq!(s.parts_since(3, 10), 0b100);
        assert_eq!(s.parts_since(3, 9), WHOLE);
        assert_eq!(s.written[&3], 12);
    }

    /// Many threads note pages (holding their parts for writing, a number
    /// taken under them) and read ranges with a fresh number (holding their
    /// parts for reading), the parts taken in any mix: every reader sees
    /// every note of a lower number of its range's pages (what keeps a write
    /// from being lost), and nobody waits for ever. (A watchdog fails the
    /// test after 60 s.)
    #[test]
    fn a_fresh_number_sees_every_lower_note_and_nobody_deadlocks() {
        const P: usize = 1 << SHARD_SHIFT;
        const THREADS: usize = 8;
        const ROUNDS: usize = 3000;
        static NOTES: Notes = Notes::new();
        static NUMBER: AtomicU64 = AtomicU64::new(0);
        let (done, wait) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // Per thread: what it noted (page, number) and what it read
            // (first, last, number, the notes read).
            type Read = (usize, usize, u64, Vec<(usize, u64)>);
            let logs: Vec<(Vec<(usize, u64)>, Vec<Read>)> = std::thread::scope(|s| {
                let all: Vec<_> = (0..THREADS)
                    .map(|t| {
                        s.spawn(move || {
                            let mut seed = t as u64 * 104_729 + 3;
                            let (mut noted, mut read) = (Vec::new(), Vec::new());
                            for _ in 0..ROUNDS {
                                let first = rng(&mut seed, 200 * P);
                                if rng(&mut seed, 2) == 0 {
                                    // A few pages, some far apart.
                                    let mut pages: Vec<usize> =
                                        (0..1 + rng(&mut seed, 6)).map(|_| first + rng(&mut seed, 3 * P)).collect();
                                    if rng(&mut seed, 4) == 0 {
                                        pages.push(rng(&mut seed, 200 * P));
                                    }
                                    let mut held = NOTES.hold(shards_of_pages(pages.iter().copied()), true);
                                    let now = NUMBER.fetch_add(1, Relaxed) + 1;
                                    for &p in &pages {
                                        held.note(p, WHOLE, now);
                                        noted.push((p, now));
                                    }
                                } else {
                                    let last = first + 1 + rng(&mut seed, 3 * P);
                                    let held = NOTES.hold(shards_of(first, last), false);
                                    let now = NUMBER.fetch_add(1, Relaxed) + 1;
                                    read.push((first, last, now, held.range(first, last)));
                                }
                            }
                            (noted, read)
                        })
                    })
                    .collect();
                all.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut by_page: std::collections::HashMap<usize, Vec<u64>> = Default::default();
            for (noted, _) in &logs {
                for &(p, n) in noted {
                    by_page.entry(p).or_default().push(n);
                }
            }
            by_page.values_mut().for_each(|v| v.sort_unstable());
            let mut checked = 0;
            for (_, read) in &logs {
                for (first, last, now, seen) in read {
                    let seen: std::collections::HashMap<usize, u64> = seen.iter().copied().collect();
                    for (&p, ns) in by_page.iter().filter(|(p, _)| (*first..*last).contains(*p)) {
                        // The highest note below the reader's number.
                        let below = ns.partition_point(|n| n < now);
                        if below > 0 {
                            let wanted = ns[below - 1];
                            let got = seen.get(&p).copied().unwrap_or(0);
                            assert!(got >= wanted, "page {p}: read {got} with number {now}, note {wanted} missed");
                            checked += 1;
                        }
                    }
                }
            }
            assert!(checked > 1000, "only {checked} checks");
            done.send(()).unwrap();
        });
        wait.recv_timeout(std::time::Duration::from_secs(60)).expect("the parts' holders deadlocked (or a check failed)");
    }

    /// The journal's functions themselves, with the switch on in all threads
    /// and in every other one (as in an A/B run): writers and askers of one
    /// region; a submission's one asking standing over another region, its
    /// thread writing there (told with `wrote` or found by its asking), and
    /// other threads asking about that region meanwhile. Whoever wrote a page
    /// before asking is told of that write.
    #[test]
    fn no_write_is_lost_with_port_7() {
        askers_and_a_submission(1);
        askers_and_a_submission(2);
    }

    /// `port`: 1, renderer port #7 on in all threads; 2, in every other.
    fn askers_and_a_submission(port: u8) {
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const PAGES: usize = 48;
        const HALF: usize = PAGES / 2;
        const THREADS: usize = 6;
        // SAFETY: a reservation of our own, freed at the end.
        let at = unsafe {
            VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert!(at != 0 && at.is_multiple_of(PAGE));
        let shared_before = PORT7[P7_SHARED_NOTES].load(Relaxed);
        let skipped_before = PORT7[P7_SKIPPED].load(Relaxed);
        std::thread::scope(|s| {
            // Writers and askers of the first half.
            for t in 0..THREADS {
                s.spawn(move || {
                    PORT_JOURNAL_IN_TEST.set(port == 1 || t % 2 == 0);
                    let mut last = [0u64; HALF];
                    let mut seed = t as u64 * 7919 + 1;
                    for _ in 0..2500 {
                        let page = rng(&mut seed, HALF);
                        let from = page - rng(&mut seed, page.min(8) + 1);
                        let to = page + 1 + rng(&mut seed, (HALF - page).min(8));
                        // SAFETY: committed memory of ours.
                        unsafe { ((at + page * PAGE + t * 8) as *mut u64).write_volatile(1) };
                        let (now, written) = writes(at + from * PAGE, at + to * PAGE, false).expect("write-watched");
                        let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                        assert!(number > last[page], "page {page}: {number} is not after {}", last[page]);
                        for l in &mut last[from..to] {
                            *l = now;
                        }
                        // Now and then a no-op of the standing asking's.
                        if rng(&mut seed, 16) == 0 {
                            wrote(at + page * PAGE, 8);
                            ask_ahead_done();
                        }
                    }
                });
            }
            // Askers of the second half that write nothing there: they take
            // the kernel's answers from the submission's thread.
            for t in 0..2 {
                s.spawn(move || {
                    PORT_JOURNAL_IN_TEST.set(port == 1 || t == 0);
                    let mut seed = t as u64 * 31 + 5;
                    for _ in 0..3000 {
                        let page = HALF + rng(&mut seed, HALF);
                        let to = (page + 1 + rng(&mut seed, 4)).min(PAGES);
                        let _ = writes(at + page * PAGE, at + to * PAGE, false);
                    }
                });
            }
            // The submission: writes pages of the second half, asks once over
            // all of it, writes more and tells `wrote`, reads everything from
            // the standing asking, ends it.
            s.spawn(move || {
                let mut last = [0u64; HALF];
                let mut seed = 99;
                for round in 0..400 {
                    PORT_JOURNAL_IN_TEST.set(port == 1 || round % 2 == 0);
                    let before: Vec<usize> = (0..3).map(|_| rng(&mut seed, HALF)).collect();
                    for &p in &before {
                        // SAFETY: committed memory of ours.
                        unsafe { ((at + (HALF + p) * PAGE) as *mut u64).write_volatile(round) };
                    }
                    let ranges: Vec<(usize, usize)> =
                        (0..4).map(|k| (at + (HALF + k * 6) * PAGE, at + (HALF + k * 6 + 6) * PAGE)).collect();
                    ask_once(&ranges);
                    let told = rng(&mut seed, HALF);
                    // SAFETY: committed memory of ours.
                    unsafe { ((at + (HALF + told) * PAGE + 64) as *mut u64).write_volatile(round) };
                    wrote(at + (HALF + told) * PAGE + 64, 8);
                    let mut read: Vec<usize> = before.iter().copied().chain([told]).collect();
                    read.sort_unstable();
                    read.dedup();
                    for p in read {
                        let (now, written) =
                            writes(at + (HALF + p) * PAGE, at + (HALF + p + 1) * PAGE, false).expect("write-watched");
                        let number = written.iter().find(|w| w.0 == at / PAGE + HALF + p).map_or(0, |w| w.1);
                        assert!(number > last[p], "round {round}, page {p}: {number} is not after {}", last[p]);
                        last[p] = now;
                    }
                    ask_ahead_done();
                    ask_once_end();
                }
            });
        });
        assert!(PORT7[P7_SHARED_NOTES].load(Relaxed) > shared_before, "the shared notings ran");
        assert!(PORT7[P7_SKIPPED].load(Relaxed) > skipped_before, "needless takings alone were left out");
        // SAFETY: the reservation made above.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }
}

#[cfg(test)]
mod rearm_tests {
    use super::*;
    use std::sync::atomic::Ordering::Relaxed;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_GUARD,
        PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualQuery,
    };

    /// A committed page is guarded again; one that is not committed any more is
    /// not, and that is counted, as is a fault that gave no address.
    #[test]
    fn a_page_that_cannot_be_guarded_again_is_counted() {
        // SAFETY: a fresh page of our own.
        let page = unsafe {
            VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
        } as usize;
        assert_ne!(page, 0);
        let before = NOT_REARMED.load(Relaxed);
        assert!(rearm_guard(Some(page)));
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a valid out pointer of the right size.
        assert_ne!(unsafe { VirtualQuery(page as *const _, &mut info, size_of_val(&info)) }, 0);
        assert_ne!(info.Protect & PAGE_GUARD, 0, "a committed page is guarded again");
        // SAFETY: our page; decommitted, it is reserved and not committed.
        unsafe { VirtualFree(page as *mut _, 4096, MEM_DECOMMIT) };
        assert!(!rearm_guard(Some(page)), "a page that is not committed cannot be guarded");
        assert!(!rearm_guard(None), "a fault that gave no address");
        assert_eq!(NOT_REARMED.load(Relaxed), before + 2);
        // SAFETY: our allocation.
        unsafe { VirtualFree(page as *mut _, 0, MEM_RELEASE) };
    }
}

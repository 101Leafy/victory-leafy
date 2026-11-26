//! Bookkeeping of the pass over the view heap's buffer views for shaders
//! that pick them by a computed index (`draw::heap_buffer_pass`): which
//! slots it looks at again, and the span of each host copy their views
//! read. Kept apart from D3D12 so that it can be tested alone.
//!
//! The pass used to look again at a sixteenth of the heap's buffer views
//! every pass, each one every 16 passes, written or not, and to rebuild the
//! spans of all host copies from all views every pass. Now a slot is looked
//! at again only when
//! - its buffer view was written or went (the heap's own set of changes);
//! - the view table entry its record names was freed (the title released
//!   the resource: `descriptor::forget`), which leaves the record as it was;
//! - its record in title memory is not the one seen at the last look (a
//!   record copied by the title itself, past the API); a compare of 12
//!   bytes per slot in slot order, no lock and no lookup;
//! - it found no host copy at its last look (no copy of its memory yet, or
//!   a draw's table brought it up to date): every 16 passes, as before;
//! - every [`FULL_EVERY`] passes, all of them, as a safety net.
//!
//! Those last three are the sweep, done once a pass; the first two can be
//! looked at in rounds between passes as well ([`Picked::plan`] without a
//! sweep). The last two are spread: a part of the slots each pass, each
//! slot in the same part each time. The span of a copy is kept as counts of
//! its views' starts and ends, so a view that goes costs a lookup and not a
//! walk over the copy's views. `VICTORY_HEAP_PASS_FULL=1` brings back the
//! old pass to compare.

use std::collections::{BTreeMap, HashMap};

/// Passes between two looks at every slot, besides those marked.
pub(crate) const FULL_EVERY: u64 = 256;
/// Passes between two looks at slots that found no host copy (and between
/// two looks at every slot with the old pass).
pub(crate) const ORPHANS_EVERY: u64 = 16;

/// Whether `VICTORY_HEAP_PASS_FULL=1` asks for the old pass.
pub(crate) fn full_pass() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_PASS_FULL").is_some_and(|v| v == "1"))
}

/// A slot's record in title memory: the tag and two words
/// (`descriptor::record_words`).
pub(crate) type Record = [u32; 3];

/// Whether `slot` is in this pass's part of `every` parts.
fn in_part(slot: u32, pass: u64, every: u64) -> bool {
    u64::from(slot) % every == pass % every
}

struct Slot {
    copy: usize,
    from: u64,
    to: u64,
    record: Record,
    /// The table entry under which `by_entry` has the slot.
    entry: Option<u32>,
}

struct Copy<C> {
    copy: C,
    /// How many of its views begin, and end, at each address.
    froms: BTreeMap<u64, u32>,
    tos: BTreeMap<u64, u32>,
}

fn count_in(counts: &mut BTreeMap<u64, u32>, at: u64) {
    *counts.entry(at).or_default() += 1;
}

fn count_out(counts: &mut BTreeMap<u64, u32>, at: u64) {
    if let Some(n) = counts.get_mut(&at) {
        *n -= 1;
        if *n == 0 {
            counts.remove(&at);
        }
    }
}

/// What a pass looks at (see the module).
pub(crate) struct Picked<C> {
    passes: u64,
    /// In slot order: every pass reads the record of each of them from
    /// title memory ([`Picked::plan`]), and in the order of a hash table
    /// those reads fall all over the heap.
    slots: BTreeMap<u32, Slot>,
    copies: HashMap<usize, Copy<C>>,
    /// Slots that found no host copy, with the record seen and the entry
    /// `by_entry` has them under. In slot order as `slots` is.
    orphans: BTreeMap<u32, (Record, Option<u32>)>,
    /// View table entry -> slots whose record named it at their last look
    /// (some may name another by now: checked when used). A slot is under
    /// one entry once, however often it is looked at.
    by_entry: HashMap<u32, Vec<u32>>,
    /// Table entries freed since the last plan; `None`: not known.
    forgotten: Option<Vec<u32>>,
    /// Slots planned and not looked at ([`Picked::put_back`]).
    carried: Vec<u32>,
    /// The last plan was every slot.
    whole: bool,
    /// The old pass (`VICTORY_HEAP_PASS_FULL=1`).
    full: bool,
}

impl<C: Clone> Picked<C> {
    pub(crate) fn new(full: bool) -> Self {
        Picked {
            passes: 0,
            slots: BTreeMap::new(),
            copies: HashMap::new(),
            orphans: BTreeMap::new(),
            by_entry: HashMap::new(),
            forgotten: None,
            carried: Vec::new(),
            whole: false,
            full,
        }
    }

    /// View table entries were freed (`None`: some were, which is not
    /// known): their slots are looked at next plan. Every heap's pass is
    /// told, though one of them asks.
    pub(crate) fn note_forgotten(&mut self, entries: Option<&[u32]>) {
        match (entries, &mut self.forgotten) {
            (Some(new), Some(kept)) => kept.extend_from_slice(new),
            _ => self.forgotten = None,
        }
    }

    /// The slots to look at again, in order. `changed`: the slots whose
    /// buffer view was written or went since the last plan (and the entries
    /// [`Picked::note_forgotten`] was told of: if it was told it does not
    /// know, all slots are looked at); `sweep`: the records of all slots
    /// held are compared and the parts of this pass are taken, and the pass
    /// is counted (a plan without it is a round between passes); `all`:
    /// every slot that holds a buffer view; `record`: a slot's record now;
    /// `entry`: the table entry a record names, if it names one.
    pub(crate) fn plan(
        &mut self,
        changed: Vec<u32>,
        sweep: bool,
        all: impl FnOnce() -> Vec<u32>,
        record: impl Fn(u32) -> Option<Record>,
        entry: impl Fn(Record) -> Option<u32>,
    ) -> Vec<u32> {
        let pass = self.passes;
        let forgotten = self.forgotten.replace(Vec::new());
        self.whole = pass == 0 || (!self.full && forgotten.is_none());
        if self.whole || sweep {
            self.passes += 1;
        }
        if self.whole {
            self.clear();
            return all();
        }
        let mut due = changed;
        due.append(&mut self.carried);
        if self.full {
            // A part of every slot, and of the ones held.
            let part = |&s: &u32| in_part(s, pass, ORPHANS_EVERY);
            due.extend(all().into_iter().filter(part));
            due.extend(self.slots.keys().chain(self.orphans.keys()).copied().filter(part));
        } else {
            for e in forgotten.into_iter().flatten() {
                if let Some(slots) = self.by_entry.remove(&e) {
                    for s in slots {
                        // Taken out from under the entry: noted again at
                        // its next look.
                        let seen = match (self.slots.get_mut(&s), self.orphans.get_mut(&s)) {
                            (Some(kept), _) if kept.entry == Some(e) => {
                                kept.entry = None;
                                Some(kept.record)
                            }
                            (None, Some((seen, under))) if *under == Some(e) => {
                                *under = None;
                                Some(*seen)
                            }
                            _ => None,
                        };
                        // Only those whose record still names the entry.
                        if seen.is_some_and(|r| entry(r) == Some(e)) {
                            due.push(s);
                        }
                    }
                }
            }
            if sweep {
                due.extend(
                    self.slots
                        .iter()
                        .filter(|&(&slot, kept)| in_part(slot, pass, FULL_EVERY) || record(slot) != Some(kept.record))
                        .map(|(&slot, _)| slot),
                );
                due.extend(
                    self.orphans
                        .iter()
                        .filter(|&(&slot, &(seen, _))| in_part(slot, pass, ORPHANS_EVERY) || record(slot) != Some(seen))
                        .map(|(&slot, _)| slot),
                );
            }
        }
        due.sort_unstable();
        due.dedup();
        due
    }

    /// Slots of the last plan that were not looked at: the next plan has
    /// them.
    pub(crate) fn put_back(&mut self, slots: &[u32]) {
        self.carried.extend_from_slice(slots);
    }

    /// Whether the last [`Picked::plan`] was every slot (what was held is
    /// let go).
    pub(crate) fn whole(&self) -> bool {
        self.whole
    }

    fn clear(&mut self) {
        self.slots.clear();
        self.copies.clear();
        self.orphans.clear();
        self.by_entry.clear();
        self.carried.clear();
    }

    /// What slot `slot` reads: its copy's identity and range; `None` for no
    /// copy.
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    pub(crate) fn held(&self, slot: u32) -> Option<(usize, u64, u64)> {
        self.slots.get(&slot).map(|s| (s.copy, s.from, s.to))
    }

    /// What slot `slot` held is let go (it is looked at now). The entry
    /// `by_entry` has it under, if any.
    pub(crate) fn forget(&mut self, slot: u32) -> Option<u32> {
        if let Some((_, under)) = self.orphans.remove(&slot) {
            return under;
        }
        let old = self.slots.remove(&slot)?;
        if let Some(copy) = self.copies.get_mut(&old.copy) {
            count_out(&mut copy.froms, old.from);
            count_out(&mut copy.tos, old.to);
            if copy.froms.is_empty() {
                self.copies.remove(&old.copy);
            }
        }
        old.entry
    }

    /// Slot `slot` (with `record`, naming table entry `entry`) reads title
    /// memory `[from, to)` through host copy `copy` (`key`: its identity).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn keep(&mut self, slot: u32, key: usize, copy: &C, from: u64, to: u64, record: Record, entry: Option<u32>) {
        let under = self.forget(slot);
        let entry = self.note_entry(slot, under, entry);
        self.slots.insert(slot, Slot { copy: key, from, to, record, entry });
        let c = self.copies.entry(key).or_insert_with(|| Copy {
            copy: copy.clone(),
            froms: BTreeMap::new(),
            tos: BTreeMap::new(),
        });
        count_in(&mut c.froms, from);
        count_in(&mut c.tos, to);
    }

    /// Slot `slot` holds a buffer view whose memory has no host copy yet,
    /// or one the pass leaves alone for now.
    pub(crate) fn orphan(&mut self, slot: u32, record: Record, entry: Option<u32>) {
        let under = self.forget(slot);
        let entry = self.note_entry(slot, under, entry);
        self.orphans.insert(slot, (record, entry));
    }

    /// The entry the slot is under from now on: put under `entry` unless
    /// it is under it already (`under`).
    fn note_entry(&mut self, slot: u32, under: Option<u32>, entry: Option<u32>) -> Option<u32> {
        // Only kept with the incremental pass (the old one does not use it).
        if self.full {
            return None;
        }
        match entry {
            Some(e) if under != Some(e) => {
                // Left under the old one: checked when that one is used.
                self.by_entry.entry(e).or_default().push(slot);
                Some(e)
            }
            Some(_) => under,
            None => under,
        }
    }

    /// The views kept.
    pub(crate) fn views(&self) -> usize {
        self.slots.len()
    }

    /// Per host copy, the span its views read: `(copy, from, to)`.
    pub(crate) fn spans(&self) -> Vec<(C, u64, u64)> {
        if self.full {
            // As the old pass did: from every view, every pass.
            let mut spans: HashMap<usize, (u64, u64)> = HashMap::new();
            for s in self.slots.values() {
                let span = spans.entry(s.copy).or_insert((s.from, s.to));
                span.0 = span.0.min(s.from);
                span.1 = span.1.max(s.to);
            }
            return spans
                .into_iter()
                .filter_map(|(key, (from, to))| Some((self.copies.get(&key)?.copy.clone(), from, to)))
                .collect();
        }
        self.copies
            .values()
            .filter_map(|c| Some((c.copy.clone(), *c.froms.keys().next()?, *c.tos.keys().next_back()?)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A heap of slots with records, as title memory holds them.
    struct Heap {
        records: HashMap<u32, Record>,
    }

    impl Heap {
        fn record(&self, slot: u32) -> Option<Record> {
            self.records.get(&slot).copied()
        }
    }

    fn entry(r: Record) -> Option<u32> {
        Some(r[1])
    }

    /// One pass: plans, then "looks" at each due slot: a slot reads 16
    /// bytes from `slot * 16` of copy `slot / 4`.
    fn pass(p: &mut Picked<u32>, heap: &Heap, changed: Vec<u32>, forgotten: Vec<u32>) -> Vec<u32> {
        round(p, heap, changed, forgotten, true)
    }

    /// The same with or without the sweep.
    fn round(p: &mut Picked<u32>, heap: &Heap, changed: Vec<u32>, forgotten: Vec<u32>, sweep: bool) -> Vec<u32> {
        p.note_forgotten(Some(&forgotten));
        let due = p.plan(
            changed,
            sweep,
            || {
                let mut all: Vec<u32> = heap.records.keys().copied().collect();
                all.sort_unstable();
                all
            },
            |s| heap.record(s),
            entry,
        );
        for &slot in &due {
            let Some(record) = heap.record(slot) else {
                p.forget(slot);
                continue;
            };
            let copy = slot / 4;
            let from = u64::from(slot) * 16;
            p.keep(slot, copy as usize, &copy, from, from + 16, record, entry(record));
        }
        due
    }

    /// A heap whose slots are all in the part of the last pass of a
    /// round, so that the spread safety net leaves them alone in a test.
    fn heap(n: u32) -> Heap {
        Heap {
            records: (0..n).map(|s| (s * FULL_EVERY as u32 + 255, [1, 100 + s, 1])).collect(),
        }
    }

    fn slot(s: u32) -> u32 {
        s * FULL_EVERY as u32 + 255
    }

    #[test]
    fn only_slots_changed_since_the_last_pass_are_looked_at_again() {
        let mut heap = heap(64);
        let mut p = Picked::new(false);
        // The first pass looks at all of them.
        assert_eq!(pass(&mut p, &heap, vec![], vec![]).len(), 64);
        assert!(p.whole());
        assert_eq!(p.views(), 64);
        // Nothing changed: nothing is looked at.
        assert!(pass(&mut p, &heap, vec![], vec![]).is_empty());
        assert!(!p.whole());
        // A view written through the API.
        heap.records.insert(slot(5), [1, 500, 1]);
        assert_eq!(pass(&mut p, &heap, vec![slot(5)], vec![]), vec![slot(5)]);
        // A record copied past the API: found by its words alone.
        heap.records.insert(slot(9), [1, 900, 1]);
        assert_eq!(pass(&mut p, &heap, vec![], vec![]), vec![slot(9)]);
        // The resource of entry 117 (slot 17) released: the record is as
        // it was, the slot is looked at all the same; slot 5 no longer
        // names its old entry 105 and is not.
        assert_eq!(pass(&mut p, &heap, vec![], vec![117, 105]), vec![slot(17)]);
        // And the next pass leaves them alone again.
        assert!(pass(&mut p, &heap, vec![], vec![]).is_empty());
    }

    #[test]
    fn a_slot_that_went_shrinks_its_copy_span() {
        let mut heap = Heap {
            records: (0..8).map(|s| (s, [1, 100 + s, 1])).collect(),
        };
        let mut p = Picked::new(false);
        pass(&mut p, &heap, vec![], vec![]);
        let span = |p: &mut Picked<u32>, copy: u32| p.spans().into_iter().find(|s| s.0 == copy).map(|s| (s.1, s.2));
        assert_eq!(span(&mut p, 1), Some((64, 128)));
        heap.records.remove(&7);
        pass(&mut p, &heap, vec![7], vec![]);
        assert_eq!(span(&mut p, 1), Some((64, 112)));
        for s in 4..7 {
            heap.records.remove(&s);
        }
        pass(&mut p, &heap, vec![4, 5, 6], vec![]);
        assert_eq!(span(&mut p, 1), None);
        assert_eq!(span(&mut p, 0), Some((0, 64)));
    }

    #[test]
    fn orphans_and_all_slots_are_tried_again_in_parts() {
        let heap = Heap {
            records: (0..4).map(|s| (s, [1, 100 + s, 1])).collect(),
        };
        let mut p: Picked<u32> = Picked::new(false);
        let _ = p.plan(vec![], true, || vec![0, 1, 2, 3], |s| heap.record(s), entry);
        for s in 0..3 {
            p.keep(s, 0, &0, 0, 16, heap.record(s).unwrap(), None);
        }
        p.orphan(3, heap.record(3).unwrap(), None);
        for n in 1..2 * FULL_EVERY {
            let due = p.plan(vec![], true, || vec![0, 1, 2, 3], |s| heap.record(s), entry);
            let mut want: Vec<u32> = (0..3).filter(|&s| n % FULL_EVERY == u64::from(s)).collect();
            if n % ORPHANS_EVERY == 3 {
                want.push(3);
            }
            assert_eq!(due, want, "pass {n}");
        }
    }

    #[test]
    fn the_old_pass_looks_at_a_sixteenth_of_all_each_pass() {
        let heap = Heap {
            records: (0..40).map(|s| (s, [1, 100 + s, 1])).collect(),
        };
        let all = || (0..40).collect::<Vec<u32>>();
        let mut p: Picked<u32> = Picked::new(true);
        assert_eq!(p.plan(vec![], true, all, |s| heap.record(s), entry), all());
        for n in 1..40u64 {
            p.note_forgotten(None);
            let due = p.plan(vec![2], true, all, |s| heap.record(s), entry);
            let mut want: Vec<u32> = all().into_iter().filter(|&s| u64::from(s) % ORPHANS_EVERY == n % ORPHANS_EVERY).collect();
            want.push(2);
            want.sort_unstable();
            want.dedup();
            assert_eq!(due, want, "pass {n}");
        }
    }

    #[test]
    fn a_round_without_the_sweep_looks_only_at_written_and_freed_slots() {
        let mut heap = heap(64);
        let mut p = Picked::new(false);
        pass(&mut p, &heap, vec![], vec![]);
        // Written through the API, and copied past it.
        heap.records.insert(slot(5), [1, 500, 1]);
        heap.records.insert(slot(9), [1, 900, 1]);
        assert_eq!(round(&mut p, &heap, vec![slot(5)], vec![117], false), vec![slot(5), slot(17)]);
        // The sweep finds the copied one.
        assert_eq!(pass(&mut p, &heap, vec![], vec![]), vec![slot(9)]);
        // Rounds do not count passes: the parts stay where they were.
        let passes = p.passes;
        round(&mut p, &heap, vec![], vec![], false);
        assert_eq!(p.passes, passes);
    }

    #[test]
    fn slots_not_looked_at_are_planned_again() {
        let heap = heap(8);
        let mut p = Picked::new(false);
        pass(&mut p, &heap, vec![], vec![]);
        p.put_back(&[slot(3)]);
        assert_eq!(round(&mut p, &heap, vec![slot(1)], vec![], false), vec![slot(1), slot(3)]);
        assert!(round(&mut p, &heap, vec![], vec![], false).is_empty());
    }

    #[test]
    fn a_slot_looked_at_again_is_under_its_entry_once() {
        let heap = heap(4);
        let mut p = Picked::new(false);
        pass(&mut p, &heap, vec![], vec![]);
        for _ in 0..10 {
            pass(&mut p, &heap, vec![slot(2)], vec![]);
        }
        assert_eq!(p.by_entry[&102], vec![slot(2)]);
        // Freed, looked at, and put under it again.
        assert_eq!(pass(&mut p, &heap, vec![], vec![102]), vec![slot(2)]);
        assert_eq!(p.by_entry[&102], vec![slot(2)]);
    }

    #[test]
    fn spans_follow_views_that_come_and_go() {
        let mut p: Picked<u32> = Picked::new(false);
        let _ = p.plan(vec![], true, Vec::new, |_| None, entry);
        p.keep(0, 7, &7, 100, 200, [0; 3], None);
        p.keep(1, 7, &7, 100, 150, [0; 3], None);
        p.keep(2, 7, &7, 50, 120, [0; 3], None);
        assert_eq!(p.spans(), vec![(7, 50, 200)]);
        p.forget(2);
        assert_eq!(p.spans(), vec![(7, 100, 200)]);
        p.forget(0);
        assert_eq!(p.spans(), vec![(7, 100, 150)]);
        p.keep(1, 8, &8, 0, 10, [0; 3], None);
        assert_eq!(p.spans(), vec![(8, 0, 10)]);
    }
}

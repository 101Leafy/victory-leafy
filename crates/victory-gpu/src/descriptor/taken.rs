//! Which dynamic slots of a heap are taken, one bit each, so a search for
//! a run of free slots skips a word of 64 taken or 64 free slots at once.

pub(crate) struct Taken {
    words: Vec<u64>,
    len: usize,
}

impl Taken {
    pub(crate) fn new(len: usize) -> Self {
        Self { words: vec![0; len.div_ceil(64)], len }
    }

    pub(crate) fn get(&self, at: usize) -> Option<bool> {
        (at < self.len).then(|| self.words[at / 64] >> (at % 64) & 1 != 0)
    }

    /// Marks `[from, to)` taken or free; the count that were taken before.
    pub(crate) fn fill(&mut self, from: usize, to: usize, taken: bool) -> u32 {
        let to = to.min(self.len);
        let mut was = 0;
        let mut at = from;
        while at < to {
            let bits = (to - at).min(64 - at % 64);
            let mask = if bits == 64 { u64::MAX } else { ((1u64 << bits) - 1) << (at % 64) };
            let word = &mut self.words[at / 64];
            was += (*word & mask).count_ones();
            if taken {
                *word |= mask;
            } else {
                *word &= !mask;
            }
            at += bits;
        }
        was
    }

    /// The first slot at or after `at` and before `to` that is taken
    /// (`want` true) or free (`want` false); `to` if none. Adds the words
    /// looked at to `steps`.
    fn next(&self, at: usize, to: usize, want: bool, steps: &mut u64) -> usize {
        let mut at = at;
        while at < to {
            *steps += 1;
            let word = if want { self.words[at / 64] } else { !self.words[at / 64] };
            let left = word >> (at % 64);
            if left != 0 {
                return (at + left.trailing_zeros() as usize).min(to);
            }
            at = (at / 64 + 1) * 64;
        }
        to
    }

    /// The first start in `[from, to - n]` of `n` free slots in a row that
    /// all lie before `to`.
    fn first_run(&self, from: usize, to: usize, n: usize, steps: &mut u64) -> Option<usize> {
        let mut at = from;
        while at + n <= to {
            let free = self.next(at, to, false, steps);
            if free + n > to {
                return None;
            }
            let taken = self.next(free, free + n, true, steps);
            if taken == free + n {
                return Some(free);
            }
            at = taken + 1;
        }
        None
    }

    /// `n` free slots in a row: the first such run from `cursor` on, else
    /// the first from the start that ends before `cursor + n - 1` (so a run
    /// across the cursor is found). Runs do not wrap past the end. Also
    /// the words looked at.
    pub(crate) fn find(&self, cursor: usize, n: usize) -> (Option<usize>, u64) {
        let mut steps = 0;
        if n == 0 || n > self.len {
            return (None, steps);
        }
        let cursor = cursor.min(self.len);
        let found = self.first_run(cursor, self.len, n, &mut steps).or_else(|| {
            let end = cursor.saturating_add(n - 1).min(self.len);
            self.first_run(0, end, n, &mut steps)
        });
        (found, steps)
    }

    /// The longest run of free slots (runs do not wrap past the end), and
    /// the words looked at.
    pub(crate) fn longest(&self) -> (usize, u64) {
        let mut steps = 0;
        let mut most = 0;
        let mut at = 0;
        while at < self.len {
            let free = self.next(at, self.len, false, &mut steps);
            let taken = self.next(free, self.len, true, &mut steps);
            most = most.max(taken - free);
            at = taken;
        }
        (most, steps)
    }
}

#[cfg(test)]
mod tests {
    use super::Taken;

    /// The search this replaced: slot by slot over a `Vec<bool>`.
    fn slow(taken: &[bool], cursor: usize, n: usize) -> Option<usize> {
        let end = taken.len();
        let second_end = cursor.saturating_add(n.saturating_sub(1)).min(end);
        let mut run = 0;
        let mut last = usize::MAX;
        for at in (cursor..end).chain(0..second_end) {
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                return Some(at + 1 - n);
            }
        }
        None
    }

    #[test]
    fn finds_what_the_slot_by_slot_search_found() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut random = |below: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % below as u64) as usize
        };
        for len in [1, 63, 64, 65, 130, 300] {
            for _ in 0..400 {
                let mut bits = Taken::new(len);
                let mut plain = vec![false; len];
                for _ in 0..random(len + 1) {
                    let from = random(len);
                    let to = (from + 1 + random(70)).min(len);
                    let taken = random(3) != 0;
                    let was = plain[from..to].iter().filter(|t| **t).count() as u32;
                    assert_eq!(bits.fill(from, to, taken), was);
                    plain[from..to].fill(taken);
                }
                for at in 0..len {
                    assert_eq!(bits.get(at), Some(plain[at]));
                }
                let cursor = random(len);
                let n = 1 + random(len.min(140));
                assert_eq!(bits.find(cursor, n).0, slow(&plain, cursor, n), "len {len} cursor {cursor} n {n}");
                let most = plain.split(|t| *t).map(<[bool]>::len).max().unwrap_or(0);
                assert_eq!(bits.longest().0, most, "len {len}");
                // No run longer than the longest is found from any cursor.
                assert_eq!(bits.find(cursor, most + 1).0, None);
            }
        }
    }

    #[test]
    fn longest_of_full_and_empty() {
        let mut bits = Taken::new(200);
        assert_eq!(bits.longest().0, 200);
        bits.fill(0, 200, true);
        assert_eq!(bits.longest().0, 0);
        bits.fill(70, 75, false);
        bits.fill(190, 200, false);
        assert_eq!(bits.longest().0, 10);
    }

    #[test]
    fn a_taken_word_is_one_step() {
        let mut bits = Taken::new(64 * 100);
        bits.fill(0, 64 * 99, true);
        let (found, steps) = bits.find(0, 10);
        assert_eq!(found, Some(64 * 99));
        assert!(steps <= 101, "{steps}");
    }
}

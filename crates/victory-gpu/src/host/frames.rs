//! Diagnostics: a line for every presented frame (`VICTORY_FRAMES=<file>`,
//! `VICTORY_FRAMES=1` for `victory-frames.csv` in the working folder; off by
//! default). A CSV with the frame's time, its Present, memory and every
//! counter of the frame-rate line and `copystat` as it grew during that
//! frame, so a long frame can be told apart from the ones around it rather
//! than averaged into a second. Lines go to a thread of their own and reach
//! the file about once a second.
//!
//! The first line says which card and driver ran and the `VICTORY_*`
//! switches. Nothing of the PC's owner goes in: values that look like paths
//! are written as `<path>`, and the user and computer names are taken out
//! of everything written (screenshots of the panel and these files get
//! shared).

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::time::{Duration, Instant};

use super::Host;
use super::spans;

/// Frames written so far, for the panel.
pub(super) static ROWS: AtomicU64 = AtomicU64::new(0);

/// Counters of the host and its neighbours. They grow, and some go back to
/// 0 with the frame-rate line.
const COUNTERS: &[(&str, fn() -> u64)] = &[
    ("worked_us", super::worked),
    // Of the card's clock (`super::CARD_CLOCK_US`), and worth reading right:
    // the first is the time the card sat with nothing sent; the second is how
    // much of that time our own submission thread spent working, so the card
    // was idle *behind us* and not waiting for us - it is not idle time and a
    // spreadsheet column called `card_idle_behind_us` reads as one; the third
    // is how long that thread stood waiting for the card, which is the same
    // fact as the card being busy, seen from the CPU side.
    ("card_idle_us", || get(&super::CARD_CLOCK_US[0])),
    ("card_idle_behind_us", || get(&super::CARD_CLOCK_US[1])),
    ("stood_waiting_us", || get(&super::CARD_CLOCK_US[2])),
    ("preparing_us", || get(&super::SUBMISSION_US[0])),
    ("gpu_wait_us", || get(&super::SUBMISSION_US[1])),
    ("copy_us", || get(&super::PREPARING[0])),
    ("copy_bytes", || get(&super::PREPARING[1])),
    ("texture_us", || get(&super::PREPARING[2])),
    ("texture_bytes", || get(&super::PREPARING[3])),
    ("copy_back_us", || get(&super::COPY_BACK[0])),
    ("copy_back_bytes", || get(&super::COPY_BACK[1])),
    ("card_hash_passes", || get(&super::CARD_PASSES[0])),
    ("card_hash_pieces", || get(&super::CARD_PASSES[1])),
    ("card_compare_passes", || get(&super::CARD_PASSES[2])),
    ("mesh_groups", || get(&super::MESH_GROUPS[1])),
    ("budget_refused", || get(&super::BUDGET_NOTES[0])),
    ("drawn", crate::draw::drawn),
    ("dropped", crate::draw::dropped_now),
    ("heap_buffer_passes", || get(&crate::draw::HEAP_BUFFER_PASSES)),
    ("heap_buffer_us", || get(&crate::draw::HEAP_BUFFER_MICROS)),
    ("heap_textures", || get(&crate::draw::HEAP_TEXTURES)),
    ("heap_texture_bytes", || get(&crate::draw::HEAP_TEXTURE_BYTES)),
    ("card_up_bytes", || get(&crate::shadow::CARD_STATS[0])),
    ("card_up_ranges", || get(&crate::shadow::CARD_STATS[1])),
    ("card_compared", || get(&crate::shadow::CARD_STATS[2])),
    ("card_fetched", || get(&crate::shadow::CARD_STATS[3])),
    ("card_fetched_bytes", || get(&crate::shadow::CARD_STATS[4])),
    ("card_fetch_waits", || get(&crate::shadow::CARD_STATS[5])),
    ("card_fetch_wait_us", || get(&crate::shadow::CARD_STATS[6])),
    ("refresh_lock_us", || get(&crate::texture::REFRESH_WAITS[0])),
    ("refresh_watch_us", || get(&crate::texture::REFRESH_WAITS[1])),
    ("refresh_still_us", || get(&crate::texture::REFRESH_WAITS[2])),
    ("refresh_us", || get(&crate::texture::REFRESH_WAITS[3])),
];

/// Read as they are, not as growth: memory in bytes.
const GAUGES: [&str; 9] = [
    "private",
    "working",
    "vram_used",
    "vram_budget",
    "vram_shared",
    // The card's own clock, as `spans::line` leaves it each second: submissions
    // a frame, the card working, the card idle between submissions, and the
    // longest one. `VICTORY_CARD_SPANS=1` measures them; without it they stay 0,
    // which says "not measured" rather than "nothing found". They are what
    // separates a card that is working all of the frame from one that is starved
    // between the pieces it is handed.
    //
    // Submissions are written in HUNDREDTHS (`spans::LAST[0]` is
    // `count / frames * 100`, so that a frame of the second does not round to
    // none): 5 800 is 58 submissions a frame. The column says so in its name,
    // because a scale nobody can see is a number read a hundred times too big.
    "card_span_submissions_x100",
    "card_span_busy_us",
    "card_span_between_us",
    "card_span_longest_us",
];

fn get(a: &AtomicU64) -> u64 {
    a.load(Ordering::Relaxed)
}

fn counters() -> Vec<u64> {
    let mut all: Vec<u64> = COUNTERS.iter().map(|(_, read)| read()).collect();
    all.extend(crate::copystat::peek_all());
    all
}

/// What came since `before`; a counter that went back to 0 counts from 0.
fn grown(now: &[u64], before: &[u64]) -> impl Iterator<Item = u64> {
    now.iter().zip(before).map(|(&is, &was)| if is >= was { is - was } else { is })
}

/// `text` without the names of the user and the computer, wherever they
/// are and however they are written.
pub(super) fn scrubbed(text: &str) -> String {
    let mut out = text.to_owned();
    for name in ["USERNAME", "COMPUTERNAME", "USERDOMAIN"] {
        let Ok(name) = std::env::var(name) else { continue };
        out = without(&out, &name);
    }
    out
}

fn without(text: &str, name: &str) -> String {
    if name.chars().count() < 3 {
        return text.to_owned();
    }
    let (lower, name) = (text.to_lowercase(), name.to_lowercase());
    if lower.len() != text.len() {
        return text.to_owned();
    }
    let mut out = String::new();
    let mut at = 0;
    while let Some(found) = lower[at..].find(&name) {
        out.push_str(&text[at..at + found]);
        out.push_str("<name>");
        at += found + name.len();
    }
    out.push_str(&text[at..]);
    out
}

/// A switch's value as it goes into the file.
fn shown_value(value: &str) -> String {
    if value.contains([':', '\\', '/']) {
        "<path>".to_owned()
    } else {
        scrubbed(value)
    }
}

/// The card's name as its driver gives it, with nothing of the owner in it.
pub(super) fn card(host: &Host) -> String {
    let Some(adapter) = host.adapter.as_ref() else { return String::new() };
    // SAFETY: a plain query.
    let Ok(desc) = (unsafe { adapter.GetDesc1() }) else { return String::new() };
    let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
    scrubbed(String::from_utf16_lossy(&desc.Description[..end]).trim())
}

fn first_line(host: &Host) -> String {
    let driver = host.driver() as u64;
    let parts = [48, 32, 16, 0].map(|shift| (driver >> shift) & 0xffff);
    let mut switches: Vec<String> = std::env::vars()
        .filter(|(name, _)| name.starts_with("VICTORY_") && name != "VICTORY_FRAMES")
        .map(|(name, value)| format!("{name}={}", shown_value(&value)))
        .collect();
    switches.sort();
    format!(
        "# card {}, driver {}.{}.{}.{}, switches {}",
        card(host).replace(',', " "),
        parts[0],
        parts[1],
        parts[2],
        parts[3],
        if switches.is_empty() { "none".to_owned() } else { switches.join(" ") }
    )
}

fn header() -> String {
    let mut names = vec!["frame", "at_us", "frame_us", "present_us"];
    names.extend(GAUGES);
    names.extend(COUNTERS.iter().map(|(name, _)| *name));
    names.extend(crate::copystat::NAMES);
    names.join(",")
}

/// The recorder of a presenter.
pub(super) struct Frames {
    out: Option<Sender<Vec<u64>>>,
    began: Instant,
    last: Instant,
    frame: u64,
    before: Vec<u64>,
    /// The last sample of [`GAUGES`], and when it was taken: they cost a
    /// process query and two driver queries, so they are read on a clock
    /// of their own rather than once per recorded frame ([`Frames::frame`]).
    gauges: [u64; 5],
    gauges_at: Instant,
}

impl Frames {
    pub(super) fn new(host: &Host) -> Frames {
        let now = Instant::now();
        let mut frames = Frames {
            out: None,
            began: now,
            last: now,
            frame: 0,
            before: counters(),
            gauges: [0; 5],
            gauges_at: now - Duration::from_secs(1),
        };
        let Some(asked) = std::env::var_os("VICTORY_FRAMES").filter(|v| v != "0") else { return frames };
        let path = if asked == "1" { "victory-frames.csv".into() } else { std::path::PathBuf::from(asked) };
        let file = match std::fs::File::create(&path) {
            Ok(file) => file,
            Err(e) => {
                tracing::warn!(error = %e, "frame lines: the file was not made");
                return frames;
            }
        };
        let (send, take) = channel::<Vec<u64>>();
        let top = format!("{}\n{}\n", first_line(host), header());
        let started = std::thread::Builder::new().name("frame lines".into()).spawn(move || {
            let mut file = std::io::BufWriter::with_capacity(1 << 20, file);
            let _ = file.write_all(top.as_bytes());
            let mut flushed = Instant::now();
            let mut line = String::new();
            while let Ok(row) = take.recv() {
                line.clear();
                for (i, n) in row.iter().enumerate() {
                    if i > 0 {
                        line.push(',');
                    }
                    line.push_str(&n.to_string());
                }
                line.push('\n');
                if file.write_all(line.as_bytes()).is_err() {
                    return;
                }
                ROWS.fetch_add(1, Ordering::Relaxed);
                if flushed.elapsed() >= Duration::from_secs(1) {
                    let _ = file.flush();
                    flushed = Instant::now();
                }
            }
            let _ = file.flush();
        });
        if started.is_ok() {
            tracing::info!("frame lines on");
            frames.out = Some(send);
        }
        frames
    }

    /// A frame was presented, its Present took `present`. Called right
    /// before the frame-rate line, which may set counters back to 0:
    /// [`Frames::after`] then reads them again.
    pub(super) fn frame(&mut self, host: &Host, present: Duration) {
        let Some(out) = &self.out else { return };
        let now = Instant::now();
        // The gauges are not the frame's cost and are not on its clock: a
        // process query and two `QueryVideoMemoryInfo` calls, which on some
        // drivers take longer than a frame and land inside the time the line
        // about that frame reports. Read once a second and held in between,
        // so what the file says a frame took is what the frame took.
        if self.gauges_at.elapsed() >= Duration::from_secs(1) {
            let (working, private) = crate::copystat::process_memory();
            let [budget, used, _, shared] = host.video_memory().unwrap_or_default();
            self.gauges = [private, working, used, budget, shared];
            self.gauges_at = now;
        }
        let counters = counters();
        let mut row = Vec::with_capacity(4 + GAUGES.len() + counters.len());
        // The five memory gauges as the second handed them (`self.gauges`,
        // read once a second for the reason above) and the card's own four
        // from the last line the stamps drew: one extend of thirteen, so the
        // row carries exactly as many values as the header names.
        let [private, working, used, budget, shared] = self.gauges;
        let [submissions, busy, between, longest] = spans::last();
        row.extend([
            self.frame,
            (now - self.began).as_micros() as u64,
            (now - self.last).as_micros() as u64,
            present.as_micros() as u64,
            private,
            working,
            used,
            budget,
            shared,
            submissions,
            busy,
            between,
            longest,
        ]);
        row.extend(grown(&counters, &self.before));
        self.before = counters;
        self.last = now;
        self.frame += 1;
        if out.send(row).is_err() {
            self.out = None;
        }
    }

    /// The frame-rate line was written.
    pub(super) fn after(&mut self) {
        if self.out.is_some() {
            self.before = counters();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_taken_out_however_written() {
        assert_eq!(without(r"C:\Users\Someone\x", "someone"), r"C:\Users\<name>\x");
        assert_eq!(without("SOMEONE-PC and someone-pc", "Someone-PC"), "<name> and <name>");
        assert_eq!(without("a b", "ab"), "a b");
    }

    #[test]
    fn paths_are_not_written() {
        assert_eq!(shown_value(r"J:\some\folder"), "<path>");
        assert_eq!(shown_value("/home/x"), "<path>");
        assert_eq!(shown_value("1"), "1");
    }

    #[test]
    fn counters_that_went_back_to_nothing_count_from_there() {
        let grew: Vec<u64> = grown(&[30, 20, 5], &[10, 50, 5]).collect();
        assert_eq!(grew, [20, 20, 0]);
    }

    #[test]
    fn every_column_has_a_name() {
        let columns = header().split(',').count();
        assert_eq!(columns, 4 + GAUGES.len() + counters().len());
    }
}

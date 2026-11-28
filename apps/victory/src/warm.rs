//! `victory warm`: host pipelines made ahead of a run (`known_pipelines`), in
//! processes of their own: a compiler that crashes or hangs costs one pipeline.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use victory_hle::known_pipelines as known;

/// Pipelines a process makes before another takes its place: the compiler's
/// and the translations' memory goes with it.
const AT_ONCE: usize = 40;
/// A pipeline that takes longer is taken for one the compiler hangs on.
const LONGEST: Duration = Duration::from_secs(180);

/// As the runs count one that takes long.
const SLOW_MS: f64 = 1000.0;
/// A book error requires both this exit code and an explicit worker report:
/// the MSVC runtime can also return 3 from a native compiler abort.
const BOOK_ERROR: i32 = 3;

fn folder() -> Result<PathBuf> {
    let cfg = victory_core::Config::load()?;
    Ok(cfg.cache_subdir("shaders")?.with_file_name("pipelines"))
}

/// The keys `warm-bad.txt` (`text`) holds against `fingerprint`: a pipeline
/// that ended a process stays out until the translator or the driver
/// changes, then it is tried again. Lines are `<key> fp=<fingerprint> <why>`
/// (the key first, as the older readers of the journal take it),
/// `<fingerprint> <key>` from builds that put it first, or `<key> <why>`
/// without one from builds before: those stay out whatever the fingerprint.
/// From TaynanGT's Vice-Tay#316, adapting BlueCatsz's Belobug#294.
fn bad_keys<'a>(text: &'a str, fingerprint: &str) -> HashSet<&'a str> {
    let hex = |word: &str, len| word.len() == len && word.bytes().all(|b| b.is_ascii_hexdigit());
    text.lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            let first = words.next()?;
            let second = words.next();
            let (key, recorded) = match second {
                Some(key) if hex(first, 16) && hex(key, 32) => (key, Some(first)),
                _ => (first, second.and_then(|word| word.strip_prefix("fp=")).filter(|fp| hex(fp, 16))),
            };
            (hex(key, 32) && recorded.is_none_or(|fp| fp.eq_ignore_ascii_case(fingerprint))).then_some(key)
        })
        .collect()
}

pub fn run(jobs: Option<usize>, limit: Option<usize>, again: bool, used: bool, these: Option<String>) -> Result<()> {
    let dir = folder()?;
    if let Some(these) = these {
        return make(&dir, &these);
    }
    let began = Instant::now();
    let keys = known::keys(&dir);
    if !keys.is_empty() {
        known::check_shader_book(&dir).with_context(|| format!("cannot open known-shaders.bin in {}", dir.display()))?;
    }
    victory_hle::known_pipelines::use_translations(dir.with_file_name("shaders"));
    let fingerprint = known::fingerprint();
    let done_file = dir.join("warm-done.txt");
    let bad_file = dir.join("warm-bad.txt");
    let listed = std::fs::read_to_string(&done_file).unwrap_or_default();
    let mut done: HashSet<&str> = HashSet::new();
    for line in listed.lines() {
        if let Some((of, key)) = line.split_once(' ')
            && of == fingerprint
        {
            done.insert(key);
        }
    }
    // `--again` and `VICTORY_WARM_BAD=0`: those that ended a process are tried again too.
    let listed_bad = if again || std::env::var_os("VICTORY_WARM_BAD").is_some_and(|v| v == "0") {
        String::new()
    } else {
        std::fs::read_to_string(&bad_file).unwrap_or_default()
    };
    let bad = bad_keys(&listed_bad, &fingerprint);
    let early = std::fs::read_to_string(dir.join("early-pipelines.txt")).unwrap_or_default();
    let early: HashSet<&str> = early.lines().collect();
    // Those earlier runs drew with first, the others in no order (by key).
    let mut all: Vec<usize> = (0..keys.len())
        .filter(|&n| {
            let key = keys[n].as_str();
            (again || !done.contains(key)) && !bad.contains(key) && (!used || early.contains(key))
        })
        .collect();
    all.sort_by_key(|&n| (!early.contains(keys[n].as_str()), &keys[n]));
    all.truncate(limit.unwrap_or(usize::MAX));
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let jobs = jobs.unwrap_or(cores * 2 / 3).max(1);
    println!(
        "{} pipelines known, {} made before ({fingerprint}), {} bad; {} to make, {jobs} processes",
        keys.len(),
        done.len(),
        bad.len(),
        all.len()
    );
    let chunks: Vec<&[usize]> = all.chunks(AT_ONCE).collect();
    let next = AtomicUsize::new(0);
    let book_failed = AtomicBool::new(false);
    let state = Mutex::new((Vec::<f64>::new(), 0usize, 0usize, 0usize));
    let done_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&done_file)?);
    // (Read back by the runs: these are made beside the others, not before.)
    let slow_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(dir.join("slow-pipelines.txt"))?);
    let bad_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&bad_file)?);
    std::thread::scope(|s| {
        for _ in 0..jobs.min(chunks.len()) {
            s.spawn(|| {
                while !book_failed.load(Ordering::Relaxed)
                    && let Some(&chunk) = chunks.get(next.fetch_add(1, Ordering::Relaxed))
                {
                    let mut rest = chunk;
                    while !rest.is_empty() && !book_failed.load(Ordering::Relaxed) {
                        let (made, ended) = child(rest, |n, ms, there| {
                            if ms >= SLOW_MS {
                                let _ = writeln!(lock(&slow_out), "\n{} {ms:.0}", keys[n]);
                            }
                            let mut state = lock(&state);
                            state.0.push(ms);
                            state.1 += usize::from(!there);
                            let count = state.0.len();
                            if count.is_multiple_of(1000) {
                                println!("{count} of {}, {:.0} s", all.len(), began.elapsed().as_secs_f64());
                            }
                            drop(state);
                            // One with no host pipeline was left out (a
                            // pipeline the run's switches leave out, such as
                            // `CS_RaytracingResolve` without the structure
                            // builder) or refused: not done, so that a warm
                            // with other switches still makes it.
                            if there {
                                let _ = writeln!(lock(&done_out), "{fingerprint} {}", keys[n]);
                            }
                        });
                        rest = &rest[made.min(rest.len())..];
                        match (ended, rest.split_first()) {
                            (Some(stop), Some((&n, after))) => {
                                match stop {
                                    WorkerStop::BookUnavailable => {
                                        book_failed.store(true, Ordering::Relaxed);
                                        return;
                                    }
                                    WorkerStop::UnreadableRecipe => {
                                        println!("pipeline {n} {}: missing or unreadable recipe", keys[n]);
                                        lock(&state).3 += 1;
                                    }
                                    WorkerStop::Failed(why) => {
                                        println!("pipeline {n} {}: {why}", keys[n]);
                                        let _ = writeln!(lock(&bad_out), "{} fp={fingerprint} {why}", keys[n]);
                                        lock(&state).2 += 1;
                                    }
                                }
                                rest = after;
                            }
                            _ => break,
                        }
                    }
                }
            });
        }
    });
    if book_failed.load(Ordering::Relaxed) {
        anyhow::bail!("cannot open known-shaders.bin in {}", dir.display());
    }
    let (mut times, refused, crashed, unreadable) = state.into_inner().unwrap_or_else(|e| e.into_inner());
    times.sort_by(f64::total_cmp);
    let at = |q: f64| times.get((times.len() as f64 * q) as usize).copied().unwrap_or(0.0);
    println!(
        "{} made in {:.0} s ({refused} have no host pipeline, {crashed} ended their process, {unreadable} unreadable recipes); ms each: median {:.1}, p90 {:.1}, p99 {:.1}, most {:.0}, sum {:.0} s",
        times.len(),
        began.elapsed().as_secs_f64(),
        at(0.5),
        at(0.9),
        at(0.99),
        times.last().copied().unwrap_or(0.0),
        times.iter().sum::<f64>() / 1e3
    );
    Ok(())
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

enum WorkerStop {
    BookUnavailable,
    UnreadableRecipe,
    Failed(String),
}

/// Has a process make `these`; how many it made, and how it ended if not well.
fn child(these: &[usize], mut each: impl FnMut(usize, f64, bool)) -> (usize, Option<WorkerStop>) {
    let list: Vec<String> = these.iter().map(usize::to_string).collect();
    let spawned = std::env::current_exe().and_then(|exe| {
        use std::os::windows::process::CommandExt;
        const IDLE_PRIORITY_CLASS: u32 = 0x40;
        Command::new(exe)
            .args(["warm", "--these", &list.join(",")])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .creation_flags(IDLE_PRIORITY_CLASS)
            .spawn()
    });
    let Ok(mut process) = spawned else { return (these.len(), None) };
    let Some(out) = process.stdout.take() else { return (these.len(), None) };
    let process = Mutex::new(process);
    let heard = Mutex::new(Instant::now());
    let hung = AtomicBool::new(false);
    let mut made = 0;
    let mut book_error = None;
    std::thread::scope(|s| {
        // A completed batch must not wait for the watchdog's next poll.
        // Dropping the sender also wakes it if the reader unwinds.
        let (finished, wake) = std::sync::mpsc::channel::<()>();
        let (watch_heard, watch_hung, watch_process) = (&heard, &hung, &process);
        s.spawn(move || {
            while let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
                wake.recv_timeout(Duration::from_millis(500))
            {
                if lock(watch_heard).elapsed() > LONGEST {
                    watch_hung.store(true, Ordering::Relaxed);
                    let _ = lock(watch_process).kill();
                    return;
                }
            }
        });
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            *lock(&heard) = Instant::now();
            let mut words = line.split(' ');
            match words.next() {
                Some("made")
                    if let (Some(n), Some(ms), Some(there)) = (words.next(), words.next(), words.next())
                        && let (Ok(n), Ok(ms)) = (n.parse(), ms.parse())
                        && these.get(made) == Some(&n) =>
                {
                    made += 1;
                    each(n, ms, there == "1");
                }
                Some("unreadable-recipe")
                    if let Some(n) = words.next().and_then(|n| n.parse::<usize>().ok())
                        && these.get(made) == Some(&n) =>
                {
                    book_error = Some(WorkerStop::UnreadableRecipe);
                }
                Some("book-unavailable") if made == 0 => book_error = Some(WorkerStop::BookUnavailable),
                _ => {}
            }
        }
        drop(finished);
    });
    let status = lock(&process).wait();
    if status.as_ref().is_ok_and(|s| s.code() == Some(BOOK_ERROR))
        && let Some(error) = book_error
    {
        return (made, Some(error));
    }
    let well = status.as_ref().is_ok_and(|s| s.success());
    if well && made == these.len() {
        return (made, None);
    }
    if well {
        return (made, Some(WorkerStop::Failed("worker exited without reporting all requested pipelines".into())));
    }
    let why = if hung.load(Ordering::Relaxed) {
        format!("hung for {} s", LONGEST.as_secs())
    } else {
        status.map_or_else(|e| e.to_string(), |s| format!("{s}"))
    };
    (made, Some(WorkerStop::Failed(why)))
}

/// What such a process does.
fn make(dir: &Path, these: &str) -> Result<()> {
    let these: Vec<usize> = these.split(',').map(|n| n.parse().context("--these")).collect::<Result<_>>()?;
    let cfg = victory_core::Config::load()?;
    victory_hle::known_pipelines::use_translations(cfg.cache_subdir("shaders")?);
    let mut out = std::io::stdout();
    let mut completed = 0;
    let results = known::make(dir, &these, |n, came| {
        let told = match came {
            None => writeln!(out, "at {n}"),
            Some((ms, there)) => {
                completed += 1;
                writeln!(out, "made {n} {ms:.2} {}", u8::from(there))
            }
        };
        if told.and_then(|()| out.flush()).is_err() {
            // Whoever asked is gone.
            std::process::exit(1);
        }
    });
    // Skip normal run teardown: the title is not running, and worker
    // children can own driver threads even after the last recipe.
    if let Some(n) = these.get(completed) {
        let report = match results {
            Ok(()) => {
                eprintln!("warm: missing or unreadable pipeline recipe {n} in {}", dir.display());
                writeln!(out, "unreadable-recipe {n}")
            }
            Err(error) => {
                eprintln!("warm: cannot open known-shaders.bin for pipeline recipe {n} in {}: {error}", dir.display());
                writeln!(out, "book-unavailable")
            }
        };
        let published = report.and_then(|()| out.flush()).is_ok();
        std::process::exit(if published { BOOK_ERROR } else { 1 });
    }
    std::process::exit(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pipeline that ended a process stays out under the fingerprint it did
    /// so under, and is tried again under another; lines without one stay out
    /// whatever the fingerprint, and those with the fingerprint first are read.
    #[test]
    fn the_quarantine_follows_the_fingerprint_and_keeps_older_lines() {
        let [current, stale, legacy, first] = ['1', '2', '3', '4'].map(|c| c.to_string().repeat(32));
        let fp = "1234567890abcdef";
        let other = "ffffffffffffffff";
        let listed = format!(
            "# not a key\n\n{current} fp={fp} timed out\n{stale} fp={other} ended the process\n{legacy} timed out\n{fp} {first}\n"
        );
        assert_eq!(bad_keys(&listed, fp), HashSet::from([current.as_str(), legacy.as_str(), first.as_str()]));
        assert_eq!(bad_keys(&listed, other), HashSet::from([stale.as_str(), legacy.as_str()]));
        // A fingerprint is compared without case; a word that is not one is a reason.
        assert!(bad_keys(&format!("{current} fp=1234567890ABCDEF x"), fp).contains(current.as_str()));
        assert!(bad_keys(&format!("{current} fp=nothex x"), other).contains(current.as_str()));
    }
}

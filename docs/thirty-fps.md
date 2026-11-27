# Thirty frames a second, and the arithmetic it has to obey

This is the plan, in order, with what each step removes and what would prove it did
not work. It is written against three sets of numbers, none of them mine to invent:

| Where | Frame | Preparing a submission | The card |
|---|---|---|---|
| RX 7900 XTX, 1440p (`findings.md`) | 46-65 ms | **48 ms** | 41 ms busy |
| RTX 5070 Laptop 8 GiB, 4K, world load (Logan, 2026-11-27) | 629 ms median over the frames over 300 ms | **93 ms** | 378 ms busy, 111 ms idle between submissions, longest 234 ms |
| RTX 5060 Ti, in play (cool_fox's report) | 56-119 ms | 38-92 ms | **under 12 ms** |

The last row matters as much as the second: on a slower card than Logan's the card
is *not* the wall in play, and 12 ms of card work fits inside a 33 ms frame with
room to spare. His 378 ms is a world load, where everything is being made for the
first time. Whether the card is a wall in play on his machine is one of the things
the test run is for — and it is a question, not a conclusion.

## The arithmetic, before any switch

A frame cannot be shorter than the work of preparing it: the card is handed the
frame when the preparing is done. So, with `P` the preparing and `C` the card:

    frame >= max(P, C)        and, on this port, usually frame = P + something

30 fps is 33 ms. Two conditions, both necessary:

1. **P <= 33 ms.** Today on Logan's machine P is 93 ms. That is 2.8 times too much.
2. **C <= 33 ms at the size chosen.** Plausibly already true at 1080p in play (12 ms
   on a 5060 Ti), almost certainly not during a world load.

And one number that names the lever inside P:

    93 ms / 58 submissions a frame = 1.6 ms a submission

58 is measured (`card_span_submissions_x100`, in hundredths: 5 800 is 58). So:

- **any 20 submissions a frame would be about 32 ms of preparing**, if each costs
  what it costs today;
- **or 0.57 ms each**, if the count stays at 58.

Thirty fps is one of those two, or a mixture. There is no switch that is both, which
is why this is a plan and not a line in a launcher.

## Where the 1.6 ms a submission goes

Proportions from the XTX's 48 ms (`findings.md`, "Where a frame goes"), which are the
only breakdown anyone has measured, and which agree with the split of Logan's 93 ms
(`copy_us` 30 ms = 32 %, the rest 63 ms = 68 %):

| Part | Of 48 ms | Of Logan's 93 ms, if it scales | What it is |
|---|---|---|---|
| Steps of the lists | 18 | ~35 | recording what the card will do |
| — refreshes of host copies | 10 | ~19 | noticing and moving title writes |
| Steps done on the card | 7 | ~14 | barriers, copies, queries |
| Waiting for descriptor records | 6 | ~12 | the title's tables, mirrored |
| Asking the write watch | 5 | ~10 | guards asked what changed |
| Copies in the card's memory | 4 | ~8 | keeping the working set resident |
| Title's bytes under shader writes | 4 | ~8 | comparing what the card wrote |

Under 33 ms means removing about 60 ms from that column. The rows are not
interchangeable and do not add up to a fifth of what they look like separately, but
they say where to cut first: **the steps of the lists**, at a third of it, and the
**refreshes** inside them.

## The path, in the order I would take it

Each step says what it should remove, what measures it, and what would falsify it.
Nothing here is claimed to work. Only the first step is written; the rest is work.

### 0. What is already in the pull requests, and costs nothing to try

- The device on the high-performance adapter (**#27**, Logan's) — without it a hybrid
  laptop runs on the integrated GPU, which is where his first run's one frame a second
  came from. This is a fault, not an optimisation.
- Card memory following the budget, render targets let go (**#26**, **#29**) — 6.44 GiB
  of *shared* memory out of 7.08 used is memory the card fetches over PCIe.
- The gauges off the frame's clock, and the card's own account of itself (**#24**).
- The launcher, thirty switches (**#31**), the integration of all of them (**#32**).

### 1. Fewer pieces: `VICTORY_JOIN_ONE_CALL=1` (in the launcher already)

The parts of a submission go to the queue in one call instead of one call a part,
with barriers standing in for the cuts (Skyfrye, AnyXboxX#337).

- Removes: a `ExecuteCommandLists` per part, and the driver's work behind it. With 58
  submissions this is 58 calls a frame.
- Does **not** claim: to remove preparing. It is in `stood_waiting`, not `preparing_us`.
  Saying otherwise would be the fifth number I have had to correct publicly.
- Measures it: `stood_waiting_us` and `card_span_between_us` against `card_span_busy_us`.
- Falsified if: `preparing_us` does not move and neither does the frame. Then this is a
  wait that was not there, and the next step is not a call to the queue.

### 2. Split one submission over two threads (the first item of "what is next")

What one submission does, done over two threads in a row. `findings.md` measured
preparing on a thread of its own as 103 → 75 ms, so threads do pay here; this is the
same idea one level down.

- Should remove: about half of the 68 % that is steps (≈ 30 ms of 93).
- Costs: memory. Snapshots and the ranges shaders write are **per host copy, not per
  submission** (`Host::submit` holds `self.submitted` from preparation to copy-back for
  exactly this reason), so two threads in flight means two sets of that. On an 8 GiB
  card with 6.44 GiB already shared, this is the step that can lose more than it gains.
- Measures it: `preparing_us` median, with `vram_shared` and `card_fetched_bytes` beside
  it — if shared goes over the budget, the gain is a loan.
- Falsified if: `preparing_us` falls but `frame_us` does not (the work moved off the
  critical path without shortening it), or shared memory eats the difference.

### 3. Ask for the next submission while this one is sent (the second item)

Today the preparing thread waits on the submission it just handed over, because
`self.submitted` is held from preparation to copy-back.

- Should remove: the serialisation of preparing behind the send, which is the part of
  `stood_waiting_us` (212 ms on Logan) that is ours and not the card's.
- Costs: the same memory as step 2, for the same reason. **So steps 2 and 3 are one
  decision, not two**, and the cheapest first is to measure the memory (step 0's
  `vram_shared`) before writing either.
- Falsified if: `stood_waiting_us` falls but `preparing_us` and `frame_us` do not.

### 4. Narrow what the steps touch

The record of what has been tried: descriptor records of draws' tables (280 → 124 ms),
guards looked at first (69 → 6 ms of threads' time), AVX2 over the title's bytes under
shader writes (14 → 4-7). Narrowing has paid more than any single rewrite here.

- Should remove: part of the 6 ms of descriptor records and the 5 ms of write watch
  (≈ 12 and 10 ms of Logan's 93).
- Measures it: `preparing_us` with `copy_bytes` flat — narrowing must not move bytes.
- Falsified if: bytes move too, which would mean it was a width change, not a narrowing.

### 5. Stop making host copies the card already has

"View the card's own buffers where fills copy them now" (item 1 of "what is next").

- Should remove: the 10 ms of 48 that is refreshes, ≈ 19 of Logan's 93.
- Measures it: `copy_us` and `copy_bytes` inside `preparing_us` — this is the one step
  that must move bytes down, and the only one where a fall in bytes is the proof.
- Falsified if: `copy_bytes` falls and `preparing_us` does not. Then moving memory was
  never the cost, and the copies were paying for something else.

### 6. Then the size, and only then

If P is under 33 ms and C is not: 540p is four times fewer pixels than 1080p and
sixteen times fewer than 4K, so a card at 378 ms on a world load is about 24 ms there,
and one at 12 ms in play is about 1 ms. 720p (42 ms from Logan's load figure) is the
in-between that would still miss 33 ms — worth knowing before promising it.

## What would make me say thirty is out of reach

Stated in advance, so it is not a conclusion moved to fit the result:

- **`preparing_us` does not move when submissions fall from 58 to 20.** Then the cost
  is not per submission, and the arithmetic above is the wrong model.
- **Preparing falls but `frame_us` does not**, in both step 2 and step 3. Then the
  critical path is elsewhere and this plan is aimed at the wrong wall.
- **The card is over 33 ms in play at 1080p on an 8 GiB card.** Then the size is the
  only lever left, and the answer is 540p frames, not 30 fps at 1080p.
- **Shared memory above the budget after step 2.** Then the split cannot be bought at
  this memory, and 30 fps waits for a card with more of it.

## What the code says about each step (read 2026-11-27, before writing any of it)

Going through the steps one by one, against the code rather than the plan, changed two
of them. This section is the result, and it is the reason steps 2, 3 and 5 are not in
the pull requests yet.

**Step 1 — done.** `VICTORY_JOIN_ONE_CALL=1`, off by default upstream, on in
`Play-VIctory-8GB.cmd` (#31). It is a wait removed, not a preparing.

**Step 2 — the lists' steps cannot be forked and joined, and this is the finding.**
`HostList::run_deferred` (`host.rs`, the loop over a list's `deferred().copies`) is not a
pure function of its list: inside it, in order, it calls `batch.fill()`,
`batch.flush_one(shadow)`, `crate::watch::wrote`, `crate::watch::ask_once_end`,
`crate::port_journal::prep_wrote` and `batch.title_written()`. That last one is the one
that matters: it means *"what was refreshed before this write is stale, refresh it
again"*. Those calls are a **serial log of writes**, not a set of independent jobs, and
splitting them across two threads changes which refreshes happen, not only when —
which is a rendering difference, not a timing one. Two threads in a row therefore need
**two batches and two journals with a stated merge point** (where `title_written` of one
half becomes a refresh for the other), before any thread is spawned. That is a design
first; the patch is the easy half, and it cannot be written from this machine where
the title cannot be run to see whether the log still holds.

**Step 3 — the lock is cheaper to describe than to take.** `self.submitted` is held from
preparation to copy-back because "snapshots and the ranges shaders write are per host
copy, not per submission". Letting the next submission be prepared while this one is
sent means two of those alive at once: `shadow::take_snapshots` and
`shadow::writing` would each have two sets. That is the memory cost the plan names, and
on a card with 6.44 GiB already shared it is the difference between a gain and a swap.
It is also the step where a wrong lock shows up as a rare lost write — the class of bug
this repository's own findings record as *six of two hundred runs*, i.e. one that will
not be caught by a single test run.

**Step 4 — largely already done, and the record says so.** The narrowing this step asks
for is what took 280 → 124 ms (descriptor records left to the draws), 69 → 6 ms (guards
looked at first) and 14 → 4-7 ms (AVX2 over the title's bytes). What is left of the
6 ms and 5 ms is the log above, not bytes looked at: there is no widened comparison
to narrow. I would not claim a gain here without a profile, and there is none on this
machine.

**Step 5 — a storage change, not a sleeve.** Viewing the card's own buffers where
fills copy them now means a `Shadow` (`Arc`, one per host copy) that sometimes has no
host copy at all, and every `Step::Refresh` and `batch.add` would have to answer what
to do when there is nothing on this side. It is the step with the largest measured
prize (~19 ms of Logan's 93) and the largest surface: `refresh_front`, `refresh_ahead`
and `apart()` all assume a host side exists.

**Step 6 — already upstream.** `VICTORY_OUTPUT` (the loader's patch) and
`VICTORY_TEST_LOWRES` for 540p; the launcher's ladder (#31) drives both.

What is measured today stands: **preparing is 93 ms against a 33 ms budget**, and a
frame cannot be shorter than the work of preparing it. Steps 2, 3 and 5 are the work
that would remove it. They are also the three that change what the title sees, and
none of them can be validated without the card they are for.

The honest current answer, with what is measured today: **30 fps is not reachable by
changing switches.** Preparing is 93 ms against a 33 ms budget, on a design whose own
record on a much faster card is 48 ms. It is reachable by removing about 60 ms of
work that exists, in the order above, and the first number that says whether the path
is real is `preparing_us` when the pieces per frame fall.

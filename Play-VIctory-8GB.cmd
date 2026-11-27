@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem ---------------------------------------------------------------------------
rem Play-VIctory-8GB.cmd [1080p|720p|540p|1440p|4k]: the settings that fit a
rem card that is short of memory, and a ladder of output sizes to find out
rem whether a frame is spent on pixels or on preparation.
rem
rem Written for the 8 GiB cards (RTX 4060/4070/5070 Laptop, 3060 Ti, 3070,
rem 5060), where the port runs out of a 7.5 GiB budget, spills into shared
rem memory, and then pays for it in paging: copy-backs that take a second and
rem a half for 35 MB, textures re-uploaded thousands of times, and a frame rate
rem in the single digits. On those cards the card itself measured under 12 ms
rem of work inside frames of 56 to 119 ms, so what is being paid for is memory,
rem not shading.
rem
rem THE LADDER IS THE POINT. Run it twice or three times at different sizes and
rem compare the median frame time:
rem   4K -> 1080p is 4 times fewer pixels, 1080p -> 540p another 4 times.
rem If the frame time falls by about those factors, the frame is spent on
rem pixels and the size is the lever. If it barely moves, the frame is spent on
rem preparing submissions, and no size will help: the project's own record says
rem the thread that prepares submissions took 48 ms of a 55 ms frame on an
rem RX 7900 XTX, and that preparing is the wall. A frame cannot be shorter than
rem the work of preparing it, so 30 fps needs preparing under 33 ms.
rem
rem Every switch below exists in this tree. The profile is the one the leading
rem continuation (cool_fox) ships for its "pareto-residency" launcher, minus
rem the two switches this tree does not carry (VICTORY_PORT_GUARD,
rem VICTORY_LIBRARY_STORE_LATER), plus the output size, which is the largest
rem single lever here and is off by default. Nothing in the engine changes;
rem set any VICTORY_* yourself to override a line (each is only a default).
rem ---------------------------------------------------------------------------

set "profile=%~1"
if not defined profile set "profile=1080p"

if /i "%profile%"=="540p" set "lowres=1"
if /i "%profile%"=="540p" set "pixels=960x540"
if /i "%profile%"=="720p" set "pixels=1280x720"
if /i "%profile%"=="1080p" set "pixels=1920x1080"
if /i "%profile%"=="1440p" set "pixels=2560x1440"
if /i "%profile%"=="4k" set "pixels=3840x2160"
if not defined pixels (
  echo Unknown size "%profile%": use 540p, 720p, 1080p, 1440p or 4k. 1>&2
  exit /b 2
)

if not defined VICTORY_GAME_DIR (
  echo Set VICTORY_GAME_DIR to the folder containing your title executable. 1>&2
  exit /b 2
)
if not exist "%VICTORY_GAME_DIR%\game_scarlett_master.exe" (
  echo game_scarlett_master.exe not found in VICTORY_GAME_DIR: "%VICTORY_GAME_DIR%" 1>&2
  exit /b 2
)
set "runner=%~dp0target\release\victory.exe"
if not exist "%runner%" (
  echo Build it first: cargo build --release --locked -p victory 1>&2
  exit /b 2
)

rem The pixel lever: at load the title is made to write this size into its own
rem settings instead of the 3840x2160 a Series X writes, and its screen size,
rem back buffers and render targets are all that size afterwards. 1080p is 4
rem times fewer pixels than 4K (8.29 MP to 2.07 MP) and 1.78 times fewer than
rem the 2560x1440 the title's own 2/3 upscaler would have rendered; 960x540 is
rem 16 times fewer than 4K. Under 4K the interface panels scale themselves
rem (VICTORY_UI_VIEWBOX, on by default).
rem
rem 540p is not one of the title's five modes, so it needs VICTORY_TEST_LOWRES,
rem which is the loader's own escape hatch for sizes below 720p: it writes the
rem size past the mode list (even dimensions, at least 256x144). Its comment
rem says "test runs only" because the title has no mode for it; on the sizes
rem above 720p that caveat does not apply at all.
rem
rem Each patch site is byte-compared first and left alone, with a line saying
rem so, if your build of the title differs: this cannot corrupt a run. If the
rem size did not take, the log's "presenting to a window" line still reports
rem 3840x2160 and drawn_of_the_screen stays at 2.025.
if defined lowres if not defined VICTORY_TEST_LOWRES set "VICTORY_TEST_LOWRES=1"
if not defined VICTORY_OUTPUT set "VICTORY_OUTPUT=%pixels%"

rem Card memory: mip levels take card memory only for the levels the title
rem holds, which the continuation measures at up to 3 GiB on this title, and
rem small buffers are kept at 64 KiB rather than 16 MiB apiece.
if not defined VICTORY_LEVELS_ON_DEMAND set "VICTORY_LEVELS_ON_DEMAND=1"
if not defined VICTORY_CARD_BUFFERS_MIN_KB set "VICTORY_CARD_BUFFERS_MIN_KB=64"

rem Never hold a frame waiting for a pipeline that is still being made: a
rem 500 ms stall measured on exactly this kind of card, over and over while a
rem scene loads its shaders.
if not defined VICTORY_PIPELINE_WAIT_MS set "VICTORY_PIPELINE_WAIT_MS=0"

rem Two preparing experiments, off by default in the engine and NOT measured by
rem anyone on your card. VICTORY_REFRESH_FRONT brings everything a submission
rem refreshes up to date in one go on several threads ahead of its steps,
rem instead of 580 copies one after the other on the submitting thread (3.4 to
rem 6.5 ms measured on one session); VICTORY_QUIET_WAITS leaves out a full wait
rem when nothing that writes was recorded since the last one. They are here
rem because preparing is the wall. If you see wrong textures, a black screen or
rem a device loss, run with VICTORY_REFRESH_FRONT=0 and VICTORY_QUIET_WAITS=0
rem set in your environment before launching, and say so: that is worth knowing.
if not defined VICTORY_REFRESH_FRONT set "VICTORY_REFRESH_FRONT=1"
if not defined VICTORY_QUIET_WAITS set "VICTORY_QUIET_WAITS=1"

rem Keeping the card's clocks up between submissions, and the submission
rem stages the continuation keeps on for this profile.
if not defined VICTORY_CARD_AWAKE set "VICTORY_CARD_AWAKE=1"
if not defined VICTORY_CARD_STEPS set "VICTORY_CARD_STEPS=2"
if not defined VICTORY_JOIN_COPIES set "VICTORY_JOIN_COPIES=1"
if not defined VICTORY_RELAY set "VICTORY_RELAY=2"
if not defined VICTORY_RELAY_COPY set "VICTORY_RELAY_COPY=1"
if not defined VICTORY_WIDE_BACK set "VICTORY_WIDE_BACK=1"
if not defined VICTORY_UPLOAD_AHEAD set "VICTORY_UPLOAD_AHEAD=1"
if not defined VICTORY_MAP_THREADS set "VICTORY_MAP_THREADS=1"
if not defined VICTORY_INDIRECT_WINDOW set "VICTORY_INDIRECT_WINDOW=1"
if not defined VICTORY_INDIRECT_BATCH set "VICTORY_INDIRECT_BATCH=2"
if not defined VICTORY_SHOW_AFTER set "VICTORY_SHOW_AFTER=1"
if not defined VICTORY_CARD_REHASH_RUNS set "VICTORY_CARD_REHASH_RUNS=1"
if not defined VICTORY_PORT_JOURNAL set "VICTORY_PORT_JOURNAL=1"

rem A texture that is waiting for room is not waited for.
if not defined VICTORY_TEXTURE_NO_WAIT set "VICTORY_TEXTURE_NO_WAIT=1"

rem A constant, not the frame estimate: walking into a bright doorway raised
rem the estimate and washed the whole frame out. constant:1.0 is even light.
if not defined VICTORY_AMBIENT set "VICTORY_AMBIENT=constant:0.8"
if not defined VICTORY_AMBIENT_AIR set "VICTORY_AMBIENT_AIR=1.5"

rem The write guard misses some of the title's writes (skin on a guard's arms,
rem the phone's frame read garbage until every check hashed the texture again).
rem Costs CPU; the frame rate held at 5-7 fps with it on.
if not defined VICTORY_TEXTURE_GUARD set "VICTORY_TEXTURE_GUARD=0"

rem The asynchronous submission paths stay off on NVIDIA: with THIN the head
rem skin of people in play is not drawn (eyes and teeth are), and JOIN_PARTS
rem repeats frames of the opening video. Set them to 1 to try instead.
if not defined VICTORY_THIN set "VICTORY_THIN=0"
if not defined VICTORY_JOIN_PARTS set "VICTORY_JOIN_PARTS=0"

if not defined VICTORY_CACHE_DIR set "VICTORY_CACHE_DIR=%LOCALAPPDATA%\VIctory\cache-8gb"
if not defined VICTORY_LOG if not defined RUST_LOG set "VICTORY_LOG=warn"

echo VIctory: %profile% (VICTORY_OUTPUT=%VICTORY_OUTPUT%)
"%runner%" run "%VICTORY_GAME_DIR%\game_scarlett_master.exe" --console series-x --execute
exit /b %errorlevel%

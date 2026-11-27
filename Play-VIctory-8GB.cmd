@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem ---------------------------------------------------------------------------
rem Play-VIctory-8GB.cmd: the settings that fit a card that is short of memory.
rem
rem Written for the 8 GiB laptop cards (RTX 4060/4070/5070 Laptop, 3060 Ti,
rem 3070, 5060), where the port runs out of a 7.5 GiB budget, spills into
rem shared memory, and then pays for it in paging: copy-backs that take a
rem second and a half for 35 MB, textures re-uploaded thousands of times, and
rem a frame rate in the single digits. On those cards the card itself measured
rem under 12 ms of work inside frames of 56 to 119 ms, so what is being paid
rem for is memory, not shading.
rem
rem Every switch below exists in this tree. The profile is the one the leading
rem continuation (cool_fox) ships for its "pareto-residency" launcher, minus
rem the two switches this tree does not carry (VICTORY_PORT_GUARD,
rem VICTORY_LIBRARY_STORE_LATER), plus VICTORY_OUTPUT, which is the largest
rem single lever here and is off by default. Nothing in the engine changes;
rem set VICTORY_* yourself to override any line (each is only a default).
rem
rem What to expect, measured by others on this project and not by us: 8 GiB
rem cards report 2 to 3 fps in gameplay with the settings they tried, and the
rem strongest card in the reports (RTX 5070 Ti 16 GB, 9800X3D) reports 14 to
rem 25 fps in the prison and indoors. 30 fps is reached in menus, not in
rem gameplay, by anyone yet. This launcher is aimed at getting an 8 GiB card
rem from 1-3 fps to the teens by taking it out of shared memory, not at
rem reaching 30 in the world.
rem ---------------------------------------------------------------------------

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

rem The pixel lever, and the only one of its size: at load the title is made to
rem write 1920x1080 into its own settings instead of the 3840x2160 a Series X
rem writes, and its screen size, back buffers and render targets are all that
rem size afterwards. 4 times fewer pixels than 4K, and 1.78 times fewer than
rem the 2560x1440 the title's own 2/3 upscaler would have rendered. Under 4K
rem interface panels scale themselves (VICTORY_UI_VIEWBOX, on by default).
rem The size has to be one of the title's modes: 1280x720, 1664x936,
rem 1920x1080, 2560x1440, 3840x2160. If your build of the title differs the
rem patch is left alone and says so, so this line cannot corrupt anything.
if not defined VICTORY_OUTPUT set "VICTORY_OUTPUT=1920x1080"

rem Card memory: mip levels take card memory only for the levels the title
rem holds, which the continuation measures at up to 3 GiB on this title, and
rem small buffers are kept at 64 KiB rather than 16 MiB apiece.
if not defined VICTORY_LEVELS_ON_DEMAND set "VICTORY_LEVELS_ON_DEMAND=1"
if not defined VICTORY_CARD_BUFFERS_MIN_KB set "VICTORY_CARD_BUFFERS_MIN_KB=64"

rem Never hold a frame waiting for a pipeline that is still being made: a
rem 500 ms stall measured on exactly this kind of card, over and over while a
rem scene loads its shaders.
if not defined VICTORY_PIPELINE_WAIT_MS set "VICTORY_PIPELINE_WAIT_MS=0"

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

"%runner%" run "%VICTORY_GAME_DIR%\game_scarlett_master.exe" --console series-x --execute
exit /b %errorlevel%

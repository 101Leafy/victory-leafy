# victory-leafy

**This may not work for you. You may need to pull certain changes and combine
them with your own build. It is here as a public resource that might help
someone.**

The cool_fox build people asked for, with codec_d's work, is on the
[`cool-fox`](https://github.com/101Leafy/victory-leafy/tree/cool-fox) branch.
On our NVIDIA machine it drew a grey scene, so it may not work either.

## Credits

This build stands on other people's work. The history is kept, every commit
keeps its author, and commits taken from elsewhere name the commit they came from.

- **[VIctory](https://github.com/Marvisak/VIctory)** by Marvisak and its
  contributors: the loader, the runtime services, the Direct3D 12 back end, and
  the shader recompiler.
- **[avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX)**:
  the base of this build. Recorded list calls translated on threads of their own,
  prepared and joined submissions, write relay and landing, buffer fills on the
  card, cheaper write catching, a texture copy queue, frame record and replay, and
  the measurements in [docs/findings.md](docs/findings.md).
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)**: the fork
  avatarmanz started from, and the people whose pull requests it carries.
- **[Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD)**:
  the ambient light idea, pipeline workers, and work by romainhedouin.
- **[JoshuuDev/victory-amd](https://github.com/JoshuuDev/victory-amd)** and
  **myokex/victory-amd**: the guard over every view of a page (with Spexivus,
  joroyt, and tightfall), and Joshua Odom's reads into title memory.
- **[DexNom/VIctory](https://github.com/DexNom/VIctory)**: ray tracing fixes, cube
  gradients, sparse textures, and the descriptor heap split of pull request 68.
- **[uiliannn/VItoriaBr](https://github.com/uiliannn/VItoriaBr)**: packed dot
  products, barriers for narrow-wave GPUs, and the 64 KiB constant buffer clamp.
- **[darkworldhm/victory-fork-darkworldhm](https://github.com/darkworldhm/victory-fork-darkworldhm)**:
  the texture copy and buffer-view pass ideas behind several heap changes here.
- **[FQY55kai/ZJOU-VICTORY](https://github.com/FQY55kai/ZJOU-VICTORY)**: render
  targets that are not set again.
- **Geckomacabre** and **cxy0072019-sketch**: the texture read-back that makes
  boats float.
- Open pull requests to VIctory that are merged here: wave64 programs on 32-lane
  GPUs, f64 lowering, XMA audio, HLE and XAsync fixes, and more. Their authors are
  in the list below.
- **[XCL](https://github.com/ivall/xcl)** (GPL-2.0-only) was read for how the same
  title behaves under another layer; no code of it is here.

### Contributors

Everyone with a commit in this branch, by the name on their commits.

[101Leafy](https://github.com/101Leafy),
[9hvn9yt2vk-netizen](https://github.com/9hvn9yt2vk-netizen),
[Abdellah K.](https://github.com/Abdellah-Khanani),
[Advait Palve](https://github.com/AdvaitPalve11),
[AgneyaK](https://github.com/AgneyaK),
[AlanQiang0418](https://github.com/AlanQiang0418),
[alessiodeia](https://github.com/alessiodeia),
[Andrea Casale (Maledetto)](https://github.com/Makto94),
[AnoHunted](https://github.com/AnoHunted),
[artor410](https://github.com/artor410),
ashrafsdrop,
[avatarmanz](https://github.com/avatarmanz),
[Aws](https://github.com/AwsAli05),
[BaggyALPHA](https://github.com/BaggyALPHA),
[BastienRdz](https://github.com/BastienRodz),
[Belobug](https://github.com/Belobug),
[Bloxive](https://github.com/ZekkJJ),
[broccolize](https://github.com/broccolize),
[bvtterbroot](https://github.com/bvtterbroot),
[Calvix2k](https://github.com/Calvix2k),
[Carlos Sant'Anna](https://github.com/SantannaCarlos),
[Castyan](https://github.com/Castyan),
[cxccxc](https://github.com/cxchhh),
[Dathinel](https://github.com/Dathinel),
[David Palacios](https://github.com/dpalacios69),
delacor,
[Diego](https://github.com/DiegoFR20),
[divyeshganta](https://github.com/divyeshganta),
[DocBlack89](https://github.com/DocBlack89),
[DOT HEX](https://github.com/X13X),
[DoubleD17](https://github.com/DoubleD17),
[Duff](https://github.com/Dufftwt),
[EatCodeSleep2](https://github.com/EatCodeSleep2),
[Eduardo Bertozzi](https://github.com/EduBertozzi),
[Enzo Souza Duarte](https://github.com/enzoduarte1),
[Exorion](https://github.com/Exorion1er),
[Felix2180](https://github.com/Felix2180),
[fever308](https://github.com/fever308),
[FIocker](https://github.com/FIocker),
[fluffyflattershy-afk](https://github.com/fluffyflattershy-afk),
[FrancescoGelosoEterea](https://github.com/FrancescoGelosoEterea),
[fuzeofficial](https://github.com/fuzeofficlal),
[fxceless](https://github.com/fxceless),
[George](https://github.com/GAD15),
[Giuseppe Rainone](https://github.com/giusepperainone),
[HailLucifer666](https://github.com/HailLucifer666),
[Henrique Bueno](https://github.com/HenriqueSBueno),
[heromike405-arch](https://github.com/heromike405-arch),
[hophaver](https://github.com/hophaver),
[ifBars](https://github.com/ifBars),
[imjustprism](https://github.com/imjustprism),
[ismailchaka1](https://github.com/ismailchaka1),
[itsRevela](https://github.com/itsRevela),
[jaschinski97-ops](https://github.com/jaschinski97-ops),
[Jeremy Goldberg](https://github.com/jeremygold02),
[JigsawBR](https://github.com/JigsawBR),
[julio](https://github.com/juliocesarshopify-maker),
[KoLolX3](https://github.com/KoLolX3),
[lameloball2813-ui](https://github.com/lameloball2813-ui),
[LaunchDoesntCode](https://github.com/LaunchDoesntCode),
LF2000 Builder,
[lordon1a](https://github.com/lordon1a),
[Louis Faust](https://github.com/catminou4),
[LUIS LARIOS](https://github.com/Luislariosss),
[Luiz Neves](https://github.com/LouisRiverstone),
[Luiz Otavio Mello de Campos](https://github.com/Luizbc2),
[Lyrnc](https://github.com/Lyrnc),
[Manuel Pérez](https://github.com/AndyLatimer),
[Maraco91](https://github.com/Maraco91),
marquezini,
Martin Hartfelder,
[Marvis](https://github.com/Marvisak),
[Mauro Alvarez](https://github.com/mauroalvare),
[Max Reis](https://github.com/bymaxdev),
[meower](https://github.com/a55uka),
[mgunes1311-lab](https://github.com/mgunes1311-lab),
moe-miau,
[nathanbzrr](https://github.com/nathanbzrr),
[neonbyte1](https://github.com/neonbyte1),
[nino-exe](https://github.com/nino-exe),
[Nixitov](https://github.com/Nixitov),
[Noname0202](https://github.com/MartinIndra02),
[Nostyxx](https://github.com/Nostyxx),
[NuraadGit](https://github.com/NuraadGit),
[oddiums](https://github.com/oddiums),
[Omeya Sawant](https://github.com/omeyasawant),
[Pavel](https://github.com/Pavl3000),
Pedro Fortunato,
[pes7](https://github.com/pes7),
[pgm1207](https://github.com/pablogonz12),
[Pilote Production](https://github.com/PiloteCode),
[quaternion](https://github.com/Quat3rnion),
[r4.](https://github.com/r4tedx),
[raffclar](https://github.com/raffclar),
[Rivaldo Freitas de Carvalho](https://github.com/rivaldodev),
[Saitken91](https://github.com/SAitken91),
[salchipapa969-star](https://github.com/salchipapa969-star),
[Shadow](https://github.com/UnkwnDev),
Skyfrye,
[SlopComp](https://github.com/SlopComp),
[spacede](https://github.com/space-de),
[tedin7](https://github.com/tedin7),
[Tex](https://github.com/Texbirds),
[TG0000](https://github.com/TG0000),
[Thomas Askharoun](https://github.com/askharoun),
[Vanthys](https://github.com/Vanthys),
[Vax](https://github.com/Vaximous),
[Vyacheslav Venglovsky](https://github.com/venglov),
warmooff-ops,
[Yaekai](https://github.com/Yaekai),
[ZazaJr24](https://github.com/ZazaJr24),
[给伊莉雅补魔](https://github.com/geiyiliyabumo).

**Official Stream: https://youtube.com/live/da-WFG3W9ZI?feature=share**

101Leafy's build of [VIctory](https://github.com/Marvisak/VIctory), a compatibility layer
that runs an Xbox Series title natively on Windows from files the user supplies.
This branch brings together the forks and open pull requests above, with fixes
for NVIDIA cards, Intel CPUs, and Windows 10 on top. It is early work.

**This repository contains no game files, code, or assets.** You need your own
copy of the game. It is not affiliated with or endorsed by the game's developer
or publisher.

## Where it stands (2026-11-26)

Tested on one more PC since the 7900 XTX measurements: Windows 10, Intel
i9-13900K, NVIDIA RTX 5080. Menus and videos run at 30 fps. The open world runs
at 4-6 fps there and parts of the picture are still wrong (black blocks and dark
water where draws are dropped). The first run after a new build translates every
shader again and stutters for several minutes.

Measurements on the RX 7900 XTX, and what is known to be wrong there, are in
[docs/findings.md](docs/findings.md).

## How to run it

You need Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with shader
model 6.6 and mesh shaders, Rust (the toolchain is pinned), the MSVC build
tools, and your own dump of the title.

```powershell
git clone https://github.com/101Leafy/victory-leafy
cd victory-leafy
copy victory.example.toml victory.toml      # then set your paths, outside this folder
cargo build --locked --profile diag -p victory --features diagnostics
scripts\play.ps1 -Exe <path of your copy's executable>
```

`scripts\play.ps1 -Safe` leaves out the newest, faster ways; try it first when
something is wrong. `-RayTracing` turns the title's ray-traced indirect light on.

Before the first run, ahead of time (both can be stopped and go on later):

```powershell
target\diag\victory.exe shaders <the shader archives>   # every shader translated, minutes
target\diag\victory.exe warm --used                     # pipelines earlier runs drew with, for a new driver or build
```

Commands, debugging switches, and upstream's own description:
[docs/README-fxceless.md](docs/README-fxceless.md). Every `VICTORY_...` switch is
described in a comment where it is read.

## What is different here

On top of what the forks above bring:

- **Renderer port 5** (`VICTORY_PORT_CBV=1`, off by default): translated shaders
  read constant buffers as raw bytes, so a table's constant buffer in title
  memory in place is a raw view of the title's bytes rather than a copy made at
  every submission (~23,000 a frame and ~9 MB of the city); the shader cache
  keys on the switch (from Belobug's renderer ports, 3920dc2).
- **NVIDIA**: the wave64 pull requests merged together, and float colour exports
  held to what the console's colour block allows, with a NaN written as 0
  (`VICTORY_COLOR_FINITE`).
- **Intel CPUs without SSE4a**: the title's `extrq` and `insertq` jump to SSE2
  code after their first fault (`VICTORY_PATCH_SSE4A`).
- **Windows 10**: the Agility SDK's D3D12Core is loaded so shader model 6.6 works.
- **Descriptors**: a capped host heap keeps the title's lower 393,215 slots, with
  a faster slot search, slots settled once a frame, and a buffer-view pass that
  looks only at slots that changed.
- **Pipelines**: a pipeline library folder that can be shared
  (`VICTORY_LIBRARY_DIR`), a shader cache keyed by the translator version, the
  known-pipelines book read on its own thread, and programs over 32 KiB linked as
  mesh pipelines.
- **Memory**: shadow copies over 1 MiB kept in card memory on 16 GB cards, a pool
  of read-back buffers, one shared zero buffer, and textures hashed once a frame.
- **Switch 2 Pro Controller** over USB.
- **Per-frame timing CSV** (`VICTORY_FRAMES`) for finding where a frame goes.

## License

GPL-3.0-or-later, as VIctory. See [LICENSE](LICENSE); the one file that is not
covered is named in [docs/asset-policy.md](docs/asset-policy.md).

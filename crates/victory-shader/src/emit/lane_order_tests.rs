//! How the host GPU puts a group's threads into waves.
//!
//! A translated compute program that reads other lanes relabels each
//! thread so that its host lane is its console lane (`Lower::thread_id`):
//! its flat id becomes its wave's base plus its lane index. In a 1-D group
//! the base is the wave's first flat id rounded down to the wave size, a
//! bijection only while each host wave holds one wave-aligned run of flat
//! ids; in other groups each wave takes its base from an LDS counter
//! (`Lower::wave_base`), which needs only that a group has at most one
//! partial wave, its lanes from 0. This probe records, per thread of one
//! group, its flat id, lane, wave size and its wave's first flat id, for
//! group shapes compute programs use, and checks both properties.
//!
//! The probe is plain HLSL, compiled by the Windows SDK's `dxc.exe`
//! (`VICTORY_PC_DXC`, else the newest SDK's); skipped without it or a
//! device.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, container, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

fn dxc() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("VICTORY_PC_DXC") {
        return Some(p.into());
    }
    let kits = PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut found: Vec<PathBuf> = std::fs::read_dir(&kits)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path().join(r"x64\dxc.exe"))
        .filter(|p| p.exists())
        .collect();
    found.sort();
    found.pop()
}

/// The probe for a group of `[x, y, z]` threads, its container signed.
fn probe(
    dxc: &std::path::Path,
    dir: &std::path::Path,
    [x, y, z]: [u32; 3],
    wave: Option<u32>,
) -> Vec<u8> {
    let size = wave.map_or(String::new(), |w| format!("[WaveSize({w})]"));
    let src = format!(
        r#"
{size}
[numthreads({x}, {y}, {z})]
void main(uint flat : SV_GroupIndex) {{
    RWByteAddressBuffer o = ResourceDescriptorHeap[0];
    o.Store4(flat * 16, uint4(flat, WaveGetLaneIndex(), WaveGetLaneCount(), WaveReadLaneFirst(flat)));
}}
"#
    );
    let (input, output) = (dir.join("probe.hlsl"), dir.join("probe.bin"));
    std::fs::write(&input, src).unwrap();
    let out = std::process::Command::new(dxc)
        .args(["-T", "cs_6_6", "-E", "main", "-Fo"])
        .arg(&output)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut dxil = std::fs::read(output).unwrap();
    // Without the SDK's validator the container is unsigned: sign it.
    let digest = container::hash(&dxil[20..]);
    dxil[4..20].copy_from_slice(&digest);
    dxil
}

/// What one group's threads recorded: per flat id, (lane, wave size, the
/// wave's first flat id).
fn run(host: &Host, dxil: Vec<u8>, threads: u32) -> Result<Vec<(u32, u32, u32)>, String> {
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil,
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Vec::new(),
    };
    let (root, pso) = host.compute_pipeline(&shader)?;
    let out = host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots: vec![(0, Root::Constants(vec![0]))],
            views: &[View::RawUav(threads * 4)],
            samplers: &[],
            groups: [1, 1, 1],
        },
    )?;
    Ok((0..threads as usize)
        .map(|t| {
            let r = &out[0][t * 4..t * 4 + 4];
            assert_eq!(r[0], t as u32, "thread {t} wrote its own record");
            (r[1], r[2], r[3])
        })
        .collect())
}

/// Per shape: whether every host wave is one wave-aligned run of flat ids,
/// whether rounding each wave's first flat id down to the wave size (the
/// relabelling of 1-D groups) gives every thread its own id, and the
/// waves that are not full and the lanes they use.
#[derive(Debug, PartialEq)]
struct Finding {
    shape: [u32; 3],
    wave: u32,
    aligned_runs: bool,
    relabel_bijective: bool,
    /// Flat ids of the first wave that is not an aligned run, if any.
    example: Option<Vec<u32>>,
    /// Waves that are not full: (members, whether their lanes run from 0).
    partial: Vec<(u32, bool)>,
}

fn examine(shape: [u32; 3], records: &[(u32, u32, u32)]) -> Finding {
    let wave = records[0].1;
    let mut waves: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (flat, &(_, _, first)) in records.iter().enumerate() {
        waves.entry(first).or_default().push(flat as u32);
    }
    let mut example = None;
    for members in waves.values() {
        let base = members[0] & !(wave - 1);
        let run = members
            .iter()
            .copied()
            .eq(base..base + members.len() as u32);
        if !run && example.is_none() {
            example = Some(members.clone());
        }
    }
    // The rounded id: the wave's first flat id rounded down, plus the lane.
    let mut ids: Vec<u32> = records
        .iter()
        .map(|&(lane, _, first)| (first & !(wave - 1)) + lane)
        .collect();
    ids.sort_unstable();
    let relabel_bijective = ids.iter().copied().eq(0..records.len() as u32);
    let mut lanes: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for &(lane, _, first) in records {
        lanes.entry(first).or_default().push(lane);
    }
    let partial = lanes
        .values_mut()
        .filter(|l| l.len() < wave as usize)
        .map(|l| {
            l.sort_unstable();
            (l.len() as u32, l.iter().copied().eq(0..l.len() as u32))
        })
        .collect();
    Finding {
        shape,
        wave,
        aligned_runs: example.is_none(),
        relabel_bijective,
        example,
        partial,
    }
}

/// Group shapes compute programs use: 1-D, 2-D tiles, 3-D blocks, and
/// sizes that are not powers of two.
const SHAPES: [[u32; 3]; 24] = [
    [32, 4, 1],
    [32, 8, 1],
    [64, 2, 1],
    [64, 4, 1],
    [16, 4, 1],
    [24, 8, 1],
    [48, 1, 1],
    [96, 1, 1],
    [64, 1, 1],
    [128, 1, 1],
    [256, 1, 1],
    [32, 2, 1],
    [8, 8, 1],
    [16, 16, 1],
    [32, 32, 1],
    [8, 4, 1],
    [16, 8, 1],
    [4, 4, 4],
    [8, 8, 4],
    [16, 8, 2],
    [2, 2, 16],
    [1, 64, 1],
    [12, 12, 1],
    [10, 10, 1],
];

/// What the relabelling relies on: 1-D groups are put into waves as
/// aligned runs (rounding works there), and every group has at most one
/// partial wave, its lanes from 0 (it takes the last run of the counter).
/// Multi-dimensional groups need not be aligned runs: an RTX 5070 Ti puts
/// 2-D groups 24, 32 or 64 wide, or not a power of two wide, into 16 by 2
/// tiles, which this prints.
#[test]
fn host_waves_keep_what_the_relabelling_relies_on() {
    let Some(dxc) = dxc() else {
        eprintln!("no PC dxc.exe; skipped");
        return;
    };
    let dir = std::env::temp_dir().join(format!("victory-lane-order-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut broken = Vec::new();
    // As translated programs that read other lanes declare it (the
    // console's wave size), and without, as the host chooses.
    for (label, warp, wave) in [
        ("hardware WaveSize(32)", false, Some(32)),
        ("hardware", false, None),
        ("warp WaveSize(32)", true, Some(32)),
    ] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for shape in SHAPES {
            let threads = shape.iter().product();
            let records = match run(&host, probe(&dxc, &dir, shape, wave), threads) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{label} {shape:?}: refused ({e}); skipped");
                    continue;
                }
            };
            // Each wave: its member count and lowest and highest lane.
            let mut waves: BTreeMap<u32, (u32, u32, u32)> = BTreeMap::new();
            for &(lane, _, first) in &records {
                let w = waves.entry(first).or_insert((0, u32::MAX, 0));
                *w = (w.0 + 1, w.1.min(lane), w.2.max(lane));
            }
            eprintln!(
                "{label} {shape:?}: waves (members, lanes) {:?}",
                waves
                    .values()
                    .map(|&(n, a, b)| format!("{n}:{a}-{b}"))
                    .collect::<Vec<_>>()
            );
            let f = examine(shape, &records);
            eprintln!(
                "{label} {:?}: wave {}, aligned runs {}, relabel bijective {}{}",
                f.shape,
                f.wave,
                f.aligned_runs,
                f.relabel_bijective,
                f.example
                    .as_ref()
                    .map_or(String::new(), |e| format!(", first odd wave {e:?}"))
            );
            let one_d = shape[1] == 1 && shape[2] == 1;
            if one_d && !f.relabel_bijective {
                broken.push(format!(
                    "{label} {shape:?}: a 1-D group's waves are not aligned runs"
                ));
            }
            if f.partial.len() > 1 || f.partial.iter().any(|&(_, from_0)| !from_0) {
                broken.push(format!("{label} {shape:?}: partial waves {:?}", f.partial));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        broken.is_empty(),
        "the relabelling's assumptions fail: {broken:?}"
    );
}

/// A console wave32 program that reads lanes (`v_mbcnt`, so it is
/// relabelled) and stores 1 at its thread's flat id, `y * x_size + x`, in a
/// group of `[x, y, 1]`: every slot of the group must hold 1, where ids
/// repeated by the relabelling leave some at 0. With `lds_bytes`, it
/// allocates that much LDS and writes the 1 there too.
fn marks_its_thread_native(shape: [u32; 2], lds_bytes: u32) -> NativeProgram {
    let [x, y] = shape;
    assert!(x <= 64, "an inline multiplier");
    let mbcnt = [0xD765_0005_u32, 0x0001_00C1]; // v_mbcnt_lo_u32_b32 v5, -1, 0
    let mut code: Vec<u32> = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0000, // s_load_dwordx4 s[40:43], s[4:5], 0x0
        mbcnt[0],
        mbcnt[1],
        0x0B << 25 | 3 << 17 | 1 << 9 | (128 + x), // v_mul_u32_u24 v3, x, v1
        0x4A06_0700,                               // v_add_nc_u32 v3, v0, v3
        0x3408_0682,                               // v_lshlrev_b32 v4, 2, v3
        0x7E0C_0281,                               // v_mov_b32 v6, 1
        0xBF8C_C07F,                               // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x800A_0604, // buffer_store_dword v6, v4, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    if lds_bytes > 0 {
        // ds_write_b32 v4, v6, before the store.
        code.splice(11..11, [0xD834_0000, 0x0000_0604]);
    }
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        // One user SGPR; thread ids x and y; LDS in 512-byte blocks.
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | 1 << 11 | (lds_bytes / 512) << 15,
            threads: [x, y, 1],
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// [`marks_its_thread_native`] translated for a host of waves of 32, and
/// validated where dxil.dll is found.
fn marks_its_thread(shape: [u32; 2], lds_bytes: u32) -> Result<HostShader, ShaderError> {
    let native = marks_its_thread_native(shape, lds_bytes);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resources");
    let io = io::raise(&p, &native).expect("io");
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let result = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let l = result?;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };
    match validator::Validator::load(validator::Validator::default_path().as_deref()) {
        Ok(mut v) => {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy)
                .unwrap_or_else(|e| panic!("{shape:?}, {lds_bytes} bytes of LDS: {e}"));
        }
        Err(e) => eprintln!("DXIL validator unavailable: {e}"),
    }
    Ok(shader)
}

/// A program that reads lanes in a 2-D group and already allocates all the
/// host's group-shared memory (32 KiB) keeps `wave_base`'s counter in its
/// own first dword, and still validates.
#[test]
fn programs_with_all_the_lds_still_validate() {
    for lds in [32 * 1024, 32 * 1024 - 512] {
        marks_its_thread([8, 8], lds).expect("lower");
    }
}

/// Every thread of a relabelled program gets its own id, in group shapes
/// whose host waves are aligned runs and in shapes whose waves are not
/// (16 by 2 tiles and uneven splits on an RTX 5070 Ti), with the counter
/// past the program's LDS and in its own first dword.
#[test]
fn relabelled_threads_each_get_their_own_id() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    // Translated programs that read lanes declare the console's wave size,
    // 32; a host that runs no waves of 32 refuses them.
    if let Ok((min, max)) = host.wave_lanes()
        && !(min..=max).contains(&32)
    {
        eprintln!("the host runs waves of {min} to {max} lanes, not 32; skipped");
        return;
    }
    let mut wrong = Vec::new();
    for (shape, lds) in [
        ([64, 1], 0),
        ([8, 8], 0),
        ([16, 16], 0),
        ([32, 2], 0),
        ([32, 8], 0),
        ([64, 4], 0),
        ([24, 8], 0),
        ([12, 12], 0),
        ([10, 10], 0),
        ([32, 2], 32 * 1024),
        ([10, 10], 32 * 1024),
    ] {
        let cs = marks_its_thread(shape, lds).unwrap_or_else(|e| panic!("{shape:?}: {e}"));
        let (root, pso) = match host.compute_pipeline(&cs) {
            Ok(p) => p,
            // The 1-D group takes no counter: a host refusing its program
            // refuses the wave size, whatever range it reports (an Intel UHD,
            // Xe-LP, refused it with "The parameter is incorrect").
            Err(e) if shape == [64, 1] => {
                eprintln!("the host refuses a program with waves of 32 ({e}); skipped");
                return;
            }
            Err(e) => panic!("{shape:?}: {e}"),
        };
        let n = shape[0] * shape[1];
        let ud = vec![0u32; cs.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(n)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{shape:?}: {e}"));
        let marked = out[0].iter().filter(|&&v| v == 1).count();
        eprintln!("{shape:?}, {lds} bytes of LDS: {marked} of {n} ids marked");
        if marked != n as usize {
            wrong.push(format!("{shape:?}, {lds} bytes of LDS: {marked} of {n}"));
        }
    }
    assert!(wrong.is_empty(), "ids repeated: {wrong:?}");
}

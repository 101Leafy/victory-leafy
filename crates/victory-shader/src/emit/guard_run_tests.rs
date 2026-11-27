//! VALU runs under one EXEC mask lowered in every lane with a select per register they leave
//! ([`super::Lower::guarded_run`]), and the float form of guard selects. Wave32 programs of 32
//! threads: v1..v7 start as lane-varying old values (NaN payloads, denormals), vcc holds the
//! lanes below s3 (`v_cmp_gt_u32`), and EXEC becomes s0 for the body. After the body EXEC is
//! all lanes again and every lane stores v0..v7 at dword 8 * lane.

use super::super::host::{Dispatch, Host, Root, View};
use super::super::{HostShader, TEST_GUARD_FLOAT, TEST_GUARD_RUNS, validator};
use super::{Lowered, Parts, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::ir::Reg;
use crate::gfx10::{Fields, Operand};
use crate::{Container, FourCC, io, ir, resource, structure};
use std::collections::BTreeSet;

const VCC: u16 = 106;
const EXEC: u8 = 126;
const LITERAL: u16 = 255;
const S_ENDPGM: u32 = 0xBF81_0000;

/// What v1..v7 hold before the body, each XORed with the lane.
const OLD: [u32; 8] = [
    0,
    0x7FC0_1234,
    0xFF80_0001,
    0x0000_0F0F,
    0x3F80_0000,
    0x8000_0101,
    0x4049_0FDB,
    0,
];

const fn k(n: u16) -> u16 {
    128 + n
}

const fn v(n: u16) -> u16 {
    256 + n
}

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

const fn s_load(op: u32, dst: u32, base: u32, byte_off: u32) -> [u32; 2] {
    [0xF400_0000 | op << 18 | dst << 6 | base >> 1, 0xFA00_0000 | byte_off]
}

const fn vop1(op: u32, vdst: u32, src0: u16) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0 as u32
}

const fn vop2(op: u32, vdst: u32, src0: u16, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0 as u32
}

const fn vopc(op: u32, src0: u16, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | vsrc1 << 9 | src0 as u32
}

/// SOP2 `op` of `src0` and a literal.
const fn sop2_literal(op: u32, sdst: u8, src0: u8, literal: u32) -> [u32; 2] {
    [0x8000_0000 | op << 23 | (sdst as u32) << 16 | (LITERAL as u32) << 8 | src0 as u32, literal]
}

/// VOP3 (VOP3b when `sdst` is not 0).
const fn vop3(op: u32, vdst: u32, sdst: u32, src: [u16; 3]) -> [u32; 2] {
    [
        0xD400_0000 | op << 16 | sdst << 8 | vdst,
        src[0] as u32 | (src[1] as u32) << 9 | (src[2] as u32) << 18,
    ]
}

/// The program around `body`, with an SRV and a sampler bound when `sample`.
fn program(body: &[u32], sample: bool) -> NativeProgram {
    let mut w = vec![s_mov_b32(5, 0x80)];
    w.extend(s_load(2, 8, 4, 0)); // UAV to s[8:11]
    if sample {
        w.push(s_mov_b32(7, 0x80));
        w.extend(s_load(3, 16, 4, 32)); // SRV to s[16:23]
        w.extend(s_load(2, 24, 6, 0)); // sampler to s[24:27]
    }
    w.push(vop2(0x1A, 15, k(5), 0)); // v_lshlrev_b32 v15, 5, v0
    for (r, old) in OLD.iter().enumerate().skip(1) {
        w.extend([vop2(0x1D, r as u32, LITERAL, 0), *old]); // v_xor_b32 vr, old, v0
    }
    w.push(vopc(0xC4, 3, 0)); // v_cmp_gt_u32 vcc, s3, v0
    w.push(s_mov_b32(EXEC, 0));
    w.extend_from_slice(body);
    w.push(s_mov_b32(EXEC, 193)); // exec_lo = -1
    for r in 0..8u32 {
        w.extend([0xE070_1000 | r * 4, 0x8002_000F | r << 8]); // buffer_store_dword vr, v15 offen
    }
    w.push(S_ENDPGM);
    native(&w, sample, 16)
}

/// The program of `code`: s0..s3 root constants, the UAV at dword 0 of the table in s4, and with
/// `sample` an SRV at dword 8 and a sampler table in s6.
fn native(code: &[u32], sample: bool, vgprs: u16) -> NativeProgram {
    let table = |user_data, slot| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot,
    };
    let range = |kind, at, dwords, user_data, slot| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: at,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot,
    };
    let mut layout = vec![
        RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 4,
            base_register: 0,
            user_data: 0,
            slot: 0,
        },
        table(4, 1),
    ];
    let mut used = vec![range(ResourceType::Uav, 0, 4, 4, 1)];
    if sample {
        layout.push(table(6, 2));
        used.push(range(ResourceType::Srv, 8, 8, 4, 1));
        used.push(range(ResourceType::Sampler, 0, 4, 6, 2));
    }
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: if sample { 7 << 1 } else { 5 << 1 },
            threads: [32, 1, 1],
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `native` lowered with the two switches set, and the runs it lowered in every lane by the
/// names of their instructions.
fn lowered(native: &NativeProgram, float: bool, runs: bool) -> (Lowered, Vec<Vec<&'static str>>) {
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, native).unwrap();
    TEST_GUARD_FLOAT.set(Some(float));
    TEST_GUARD_RUNS.set(Some(runs));
    let l = lower(&Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    });
    TEST_GUARD_FLOAT.set(None);
    TEST_GUARD_RUNS.set(None);
    let l = l.unwrap();
    let names = l
        .guard_runs
        .iter()
        .map(|&(b, start, end)| p.blocks[b].insts[start..end].iter().map(|i| i.name()).collect())
        .collect();
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = l.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    (l, names)
}

fn host_shader(l: Lowered) -> HostShader {
    HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        root_indexed_views: Vec::new(),
    }
}

/// One group of `shader` with s0..s3 = `inputs`, and the `dwords` of its UAV after.
fn run(host: &Host, shader: &HostShader, inputs: &[u32; 4], dwords: u32) -> Vec<u32> {
    let (root, pso) = host.compute_pipeline(shader).unwrap();
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match (u.root, u.slot) {
            (RootType::Constants, 0) => inputs[u.dword as usize],
            _ => 0,
        };
    }
    host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots: vec![(0, Root::Constants(ud))],
            views: &[View::RawUav(dwords)],
            samplers: &[],
            groups: [1, 1, 1],
        },
    )
    .unwrap()
    .swap_remove(0)
}

fn runs_of(body: &[u32], sample: bool) -> Vec<Vec<&'static str>> {
    lowered(&program(body, sample), true, true).1
}

const BEFORE: [u32; 2] = [
    vop2(0x03, 1, v(1), 3), // v_add_f32 v1, v1, v3
    vop2(0x08, 2, 1, 2),    // v_mul_f32 v2, s1, v2
];

const AFTER: [u32; 3] = {
    let fma = vop3(0x14B, 6, 0, [v(1), v(2), v(3)]); // v_fma_f32 v6, v1, v2, v3
    [vop2(0x1D, 7, v(7), 1), fma[0], fma[1]] // v_xor_b32 v7, v7, v1
};

/// The runs of BEFORE, `middle` and AFTER.
fn around(middle: &[u32], sample: bool) -> Vec<Vec<&'static str>> {
    let mut body = BEFORE.to_vec();
    body.extend_from_slice(middle);
    body.extend_from_slice(&AFTER);
    runs_of(&body, sample)
}

/// `middle` ends the run: BEFORE and AFTER are two runs and neither takes it.
fn refused(middle: &[u32], sample: bool) {
    assert_eq!(
        around(middle, sample),
        [vec!["v_add_f32", "v_mul_f32"], vec!["v_xor_b32", "v_fma_f32"]]
    );
}

#[test]
fn lane_local_instructions_make_one_run() {
    let runs = around(&[vop2(0x04, 5, v(5), 4)], false); // v_sub_f32 v5, v5, v4
    assert_eq!(
        runs,
        [vec!["v_add_f32", "v_mul_f32", "v_sub_f32", "v_xor_b32", "v_fma_f32"]]
    );
}

#[test]
fn no_runs_with_the_switch_off() {
    let mut body = BEFORE.to_vec();
    body.extend_from_slice(&AFTER);
    assert!(lowered(&program(&body, false), true, false).1.is_empty());
}

#[test]
fn a_lane_mask_picking_a_cndmask_stays_in_the_run() {
    let runs = around(&[vop2(0x01, 4, v(5), 6)], false); // v_cndmask_b32 v4, v5, v6, vcc
    assert_eq!(
        runs,
        [vec!["v_add_f32", "v_mul_f32", "v_cndmask_b32", "v_xor_b32", "v_fma_f32"]]
    );
}

#[test]
fn dpp_ends_a_run() {
    // v_mov_b32_dpp v4, v1 quad_perm:[1,0,3,2]
    refused(&[vop1(0x01, 4, 0xFA), 0xFF00_B101], false);
}

#[test]
fn v_readlane_ends_a_run() {
    let readlane = vop3(0x360, 20, 0, [v(1), k(3), 0]);
    refused(&readlane, false);
}

#[test]
fn v_readfirstlane_ends_a_run() {
    refused(&[vop1(0x02, 20, v(1))], false);
}

#[test]
fn v_writelane_ends_a_run() {
    refused(&vop3(0x361, 4, 0, [1, k(5), 0]), false);
}

#[test]
fn v_permlane16_ends_a_run() {
    refused(&vop3(0x377, 4, 0, [v(1), 1, 2]), false);
}

#[test]
fn v_mbcnt_ends_a_run() {
    refused(&vop3(0x365, 4, 0, [193, k(0), 0]), false);
}

#[test]
fn ds_swizzle_ends_a_run() {
    refused(&[0xD8D4_041F, 0x0400_0001], false); // ds_swizzle_b32 v4, v1 offset:0x41f
}

#[test]
fn ds_bpermute_ends_a_run() {
    refused(&[0xDACC_0000, 0x0400_010F], false); // ds_bpermute_b32 v4, v15, v1
}

#[test]
fn an_implicit_lod_sample_ends_a_run() {
    // image_sample v8, v[1:2], s[16:23], s[24:27] dmask:0x1 dim:2D
    refused(&[0xF080_0108, 0x00C4_0801], true);
}

#[test]
fn reading_exec_ends_a_run() {
    refused(&[vop1(0x01, 4, EXEC as u16)], false); // v_mov_b32 v4, exec_lo
}

#[test]
fn reading_a_lane_mask_as_data_ends_a_run() {
    refused(&[vop1(0x01, 4, VCC)], false); // v_mov_b32 v4, vcc_lo
}

#[test]
fn writing_vcc_ends_a_run() {
    refused(&vop3(0x30F, 4, VCC as u32, [v(1), v(2), 0]), false); // v_add_co_u32 v4, vcc_lo
}

#[test]
fn v_cmp_ends_a_run() {
    refused(&[vopc(0xC4, 3, 1)], false); // v_cmp_gt_u32 vcc, s3, v1
}

/// The four ways to lower the same program give the same dwords in every lane, on the
/// hardware adapter and on WARP, for EXEC masks from none to all; lanes outside the mask
/// keep their old values bit for bit.
#[test]
fn guarded_runs_and_float_selects_change_no_lane() {
    let fma = vop3(0x14B, 6, 0, [v(1), v(2), v(3)]);
    let body = [
        vop2(0x08, 3, 1, 4),     // v_mul_f32 v3, s1, v4 (over a denormal born an integer)
        vop2(0x03, 1, v(1), 3),  // v_add_f32 v1, v1, v3
        vop2(0x08, 2, 1, 2),     // v_mul_f32 v2, s1, v2
        vop2(0x1D, 7, v(7), 1),  // v_xor_b32 v7, v7, v1
        vop2(0x01, 4, v(5), 6),  // v_cndmask_b32 v4, v5, v6, vcc
        fma[0],
        fma[1],
        vop1(0x01, 3, v(2)),     // v_mov_b32 v3, v2
        vop1(0x02, 20, v(6)),    // v_readfirstlane_b32 s20, v6
        vop2(0x10, 5, 20, 5),    // v_max_f32 v5, s20, v5
        vop1(0x06, 1, v(7)),     // v_cvt_f32_u32 v1, v7 (v1 again)
        vop2(0x25, 2, 2, 7),     // v_add_nc_u32 v2, s2, v7
        vop2(0x08, 0, v(1), 5),  // v_mul_f32 v0, v1, v5
        vop2(0x01, 6, v(3), 1),  // v_cndmask_b32 v6, v3, v1, vcc
    ];
    let native = program(&body, false);
    let cases: [[u32; 4]; 6] = [
        [0, 0x4049_0FDB, 0x8000_0001, 13],
        [!0, 0x4049_0FDB, 0x8000_0001, 13],
        [0x5555_5555, 0xBF80_0000, 0x0000_0001, 0],
        [0xA5C3_0F96, 0x7F80_0000, 0x7FC0_0000, 32],
        [1, 0x0000_0F0F, 0xFFFF_FFFF, 7],
        [0x8000_0000, 0x3F00_0000, 0x1234_5678, 31],
    ];
    let shaders: Vec<(HostShader, usize)> = [(false, false), (true, false), (false, true), (true, true)]
        .into_iter()
        .map(|(float, runs)| {
            let (l, names) = lowered(&native, float, runs);
            assert_eq!(names.is_empty(), !runs, "runs {names:?}");
            (host_shader(l), names.len())
        })
        .collect();
    let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        ran += 1;
        for (c, inputs) in cases.iter().enumerate() {
            let mut first: Option<Vec<u32>> = None;
            for (n, (shader, _)) in shaders.iter().enumerate() {
                let out = run(&host, shader, inputs, 256);
                for lane in 0..32u32 {
                    if inputs[0] >> lane & 1 == 0 {
                        for r in 1..8 {
                            assert_eq!(
                                out[(lane * 8) as usize + r],
                                OLD[r] ^ lane,
                                "{label} case {c} lowering {n} lane {lane} v{r}"
                            );
                        }
                    }
                }
                match &first {
                    None => first = Some(out),
                    Some(want) => {
                        for (i, (a, b)) in out.iter().zip(want).enumerate() {
                            assert_eq!(
                                a,
                                b,
                                "{label} case {c} lowering {n} lane {} v{}",
                                i / 8,
                                i % 8
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(ran > 0, "no SM 6.6 device");
}

/// What a lifted run's registers start from (XORed with the lane for VGPRs): NaN payloads,
/// denormals, infinities, signed zero, f16 pairs.
const SEEDS: [u32; 12] = [
    0x3F80_0000,
    0xC049_0FDB,
    0x0000_0F0F,
    0x7FC0_1234,
    0xFF80_0000,
    0x4B00_0000,
    0x0000_0040,
    0x8000_0000,
    0x3C00_BC00,
    0x7E00_0001,
    0xFF80_0001,
    0x8000_0101,
];

fn seed(n: usize) -> u32 {
    SEEDS[n * 7 % SEEDS.len()]
}

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

/// A run in the program [`lift`] makes: EXEC is s0 ^ `exec` while it runs, and after it every
/// VGPR in `stored` (with what it held before the run) goes to the next 32 dwords.
struct Lifted {
    insts: usize,
    exec: u32,
    stored: Vec<(u8, u32)>,
}

/// SGPRs and VGPRs `run` touches, or None when it reads a special register other than VCC
/// and M0. EXEC, which every vector instruction reads, is the program's own.
fn touched(run: &[ir::Inst]) -> Option<(BTreeSet<u8>, BTreeSet<u8>)> {
    let (mut s, mut v) = (BTreeSet::new(), BTreeSet::new());
    for i in run {
        for &(r, _) in i.reads.iter().chain(&i.writes) {
            match r {
                Reg::S(c) if c == EXEC || c == EXEC + 1 => {}
                Reg::S(c) if c <= ir::VCC + 1 || c == ir::M0 => {
                    s.insert(c);
                }
                Reg::V(n) => {
                    v.insert(n);
                }
                _ => return None,
            }
        }
    }
    Some((s, v))
}

/// Registers no run uses for the program around them: a quad for the UAV, copies of s0, s1
/// and s3, the lane and the store address.
fn spare(s: &BTreeSet<u8>, v: &BTreeSet<u8>) -> Option<([u8; 4], [u8; 2])> {
    let quad = (8..=100u8).rev().step_by(4).find(|&q| (q..q + 4).all(|c| !s.contains(&c)))?;
    let singles: Vec<u8> = (6..=105u8)
        .rev()
        .filter(|c| !s.contains(c) && !(quad..quad + 4).contains(c))
        .take(3)
        .collect();
    let lanes: Vec<u8> = (0..=255u8).rev().filter(|n| !v.contains(n)).take(2).collect();
    (singles.len() == 3 && lanes.len() == 2)
        .then(|| ([quad, singles[0], singles[1], singles[2]], [lanes[0], lanes[1]]))
}

/// One program of `runs`, copied word for word: before each, every register it reads or
/// writes gets a value of its own (VCC the lanes below s3), and EXEC a mix of s0. Runs that
/// leave no spare registers are left out.
fn lift(runs: &[Vec<ir::Inst>]) -> (Vec<u32>, Vec<Lifted>) {
    let (mut used_s, mut used_v) = (BTreeSet::new(), BTreeSet::new());
    let mut taken = Vec::new();
    for run in runs {
        let Some((s, v)) = touched(run) else {
            continue;
        };
        let us: BTreeSet<u8> = used_s.union(&s).copied().collect();
        let uv: BTreeSet<u8> = used_v.union(&v).copied().collect();
        if spare(&us, &uv).is_some() {
            (used_s, used_v) = (us, uv);
            taken.push((run, s, v));
        }
    }
    let ([quad, c0, c1, c3], [lane, at]) = spare(&used_s, &used_v).unwrap();
    let mut w = vec![s_mov_b32(5, 0x80)];
    w.extend(s_load(2, quad as u32, 4, 0));
    w.extend([s_mov_b32(c0, 0), s_mov_b32(c1, 1), s_mov_b32(c3, 3)]);
    w.push(vop1(0x01, lane as u32, v(0))); // v_mov_b32 lane, v0
    w.push(vop2(0x1A, at as u32, k(2), lane as u32)); // v_lshlrev_b32 at, 2, lane
    let mut lifted = Vec::new();
    for (j, (run, s, regs)) in taken.into_iter().enumerate() {
        for c in s {
            if c == ir::VCC {
                w.push(vopc(0xC4, c3 as u16, lane as u32)); // v_cmp_gt_u32 vcc, s3, lane
            } else {
                w.extend(sop2_literal(0x12, c, c1, seed(c as usize + j))); // s_xor_b32
            }
        }
        for &r in &regs {
            w.extend([vop2(0x1D, r as u32, LITERAL, lane as u32), seed(r as usize + 3 * j)]);
        }
        let exec = (j as u32).wrapping_mul(0x9E37_79B9);
        w.extend(sop2_literal(0x12, EXEC, c0, exec));
        for i in run {
            w.extend(&i.machine.words);
        }
        w.push(s_mov_b32(EXEC, 193));
        let written: BTreeSet<u8> = run
            .iter()
            .flat_map(|i| &i.writes)
            .filter_map(|w| match w.0 {
                Reg::V(n) => Some(n),
                _ => None,
            })
            .collect();
        let mut stored = Vec::new();
        for r in written {
            // buffer_store_dword r, at offen, then at += 128
            w.extend([0xE070_1000, 0x80 << 24 | (quad as u32 / 4) << 16 | (r as u32) << 8 | at as u32]);
            w.extend([vop2(0x25, at as u32, LITERAL, at as u32), 128]);
            stored.push((r, seed(r as usize + 3 * j)));
        }
        lifted.push(Lifted { insts: run.len(), exec, stored });
    }
    w.push(S_ENDPGM);
    (w, lifted)
}

/// The guarded runs of the slow compute shaders in `VICTORY_SLOW_CS` (`list.txt` of
/// `<hash> <ms> ...` slowest first, and `<hash>.bin` containers): the ten slowest and 50 of
/// the others picked with a fixed seed, per shader the eight longest runs and 16 others.
/// Each shader's runs are lifted into one program ([`lift`]) and lowered four ways, with the
/// float selects and the runs on and off. All four have to agree in every lane, on hardware
/// and WARP, and lanes outside EXEC keep what the registers held. Every write of a lifted run
/// is left to the store after it, so the writes [`super::Lower::guarded_run`] skips because
/// nothing reads them are only checked by the tests above.
#[test]
#[ignore = "needs VICTORY_SLOW_CS"]
fn lifted_runs_of_the_slow_shaders_change_no_lane() {
    let Ok(dir) = std::env::var("VICTORY_SLOW_CS") else {
        eprintln!("VICTORY_SLOW_CS unset; skipped");
        return;
    };
    let list = std::fs::read_to_string(format!("{dir}/list.txt")).unwrap();
    let names: Vec<&str> = list.lines().filter_map(|l| l.split_whitespace().next()).collect();
    let mut lcg = Lcg(0x51_0D_C5);
    let mut picked: Vec<&str> = names.iter().take(10).copied().collect();
    let mut rest: Vec<&str> = names.iter().skip(10).copied().collect();
    for _ in 0..rest.len().min(50) {
        picked.push(rest.swap_remove(lcg.below(rest.len())));
    }
    let cases: [[u32; 4]; 3] = [
        [0xA5C3_0F96, 0x0000_0000, 0, 13],
        [0x0000_FFFF, 0x8000_0001, 0, 0],
        [0xFFFF_FFFE, 0x7F80_0000, 0, 31],
    ];
    let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let hosts: Vec<(&str, Host)> = [("hardware", false), ("warp", true)]
        .into_iter()
        .filter_map(|(label, warp)| match Host::new(warp) {
            Ok(h) => Some((label, h)),
            Err(_) => {
                eprintln!("no SM 6.6 {label} device; skipped");
                None
            }
        })
        .collect();
    assert!(!hosts.is_empty(), "no SM 6.6 device");
    let (mut shaders, mut runs, mut insts, mut as_runs, mut compared) = (0, 0, 0, 0, 0u64);
    for name in picked {
        let container = std::fs::read(format!("{dir}/{name}.bin")).unwrap();
        super::RUNS_SEEN.set(Some(Vec::new()));
        TEST_GUARD_FLOAT.set(Some(true));
        TEST_GUARD_RUNS.set(Some(true));
        let translated = super::super::translate(&container);
        TEST_GUARD_FLOAT.set(None);
        TEST_GUARD_RUNS.set(None);
        let mut seen = super::RUNS_SEEN.take().unwrap_or_default();
        if let Err(e) = translated {
            eprintln!("{name}: {e}");
        }
        seen.sort_by_key(|r| (r[0].machine.offset, r.len()));
        seen.dedup_by_key(|r| (r[0].machine.offset, r.len()));
        seen.sort_by_key(|r| std::cmp::Reverse(r.len()));
        let mut chosen: Vec<Vec<ir::Inst>> = seen.drain(..seen.len().min(8)).collect();
        for _ in 0..seen.len().min(16) {
            chosen.push(seen.swap_remove(lcg.below(seen.len())));
        }
        let (code, lifted) = lift(&chosen);
        if lifted.is_empty() {
            eprintln!("{name}: no runs");
            continue;
        }
        let native = native(&code, false, 256);
        let dwords = (lifted.iter().map(|l| l.stored.len()).sum::<usize>() * 32) as u32;
        let lowerings: Vec<HostShader> = [(false, false), (true, false), (false, true), (true, true)]
            .into_iter()
            .map(|(float, runs)| {
                let (l, names) = lowered(&native, float, runs);
                if float && runs {
                    as_runs += names.iter().map(Vec::len).sum::<usize>();
                }
                host_shader(l)
            })
            .collect();
        for (label, host) in &hosts {
            for (c, inputs) in cases.iter().enumerate() {
                let outs: Vec<Vec<u32>> = lowerings.iter().map(|s| run(host, s, inputs, dwords)).collect();
                let mut at = 0;
                for (j, l) in lifted.iter().enumerate() {
                    let exec = inputs[0] ^ l.exec;
                    for &(r, old) in &l.stored {
                        for lane in 0..32u32 {
                            let i = at + lane as usize;
                            for (n, out) in outs.iter().enumerate() {
                                let place = || format!("{name} {label} case {c} run {j} lowering {n} lane {lane} v{r}");
                                assert_eq!(out[i], outs[0][i], "{}", place());
                                if exec >> lane & 1 == 0 {
                                    assert_eq!(out[i], old ^ lane, "{} outside EXEC", place());
                                }
                            }
                        }
                        at += 32;
                    }
                }
                compared += dwords as u64 * outs.len() as u64;
            }
        }
        let n: usize = lifted.iter().map(|l| l.insts).sum();
        eprintln!("{name}: {} runs, {n} insts, {dwords} dwords", lifted.len());
        shaders += 1;
        runs += lifted.len();
        insts += n;
    }
    eprintln!(
        "{shaders} shaders, {runs} runs of {insts} insts ({as_runs} lowered as runs in the lifted \
         programs), {compared} lane results compared"
    );
    assert!(shaders > 0, "no shader had runs");
}

/// What a VALU instruction leaves in its VGPR, for [`f16_rounding_of_the_title_fade`].
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Half,
    F32,
    Other,
}

fn result_kind(name: &str) -> Kind {
    if let Some(rest) = name.strip_prefix("v_cvt_") {
        if rest.starts_with("f16_") || rest.starts_with("pkrtz_f16") {
            Kind::Half
        } else if rest.starts_with("f32_") {
            Kind::F32
        } else {
            Kind::Other
        }
    } else if name.contains("frexp_exp") {
        Kind::Other
    } else if name.ends_with("_f16") {
        Kind::Half
    } else if name.ends_with("_f32") {
        Kind::F32
    } else {
        Kind::Other
    }
}

/// Whether `i` reads `r` as an f32 (otherwise it gets a pair of halves).
fn reads_f32(i: &ir::Inst, r: Reg) -> bool {
    let name = i.name();
    if let Some(rest) = name.strip_prefix("v_cvt_") {
        return rest.ends_with("_f32");
    }
    if name.starts_with("v_fma_mix") || name.starts_with("v_mad_mix") {
        let Fields::Valu(v) = &i.fields else {
            return true;
        };
        let at = v.src.iter().position(|o| match (*o, r) {
            (Operand::Vector { reg, .. }, Reg::V(n)) => reg == n,
            (Operand::Scalar { code, .. }, Reg::S(c)) => code == c,
            _ => false,
        });
        return at.is_none_or(|k| v.opsel_hi >> k & 1 == 0);
    }
    !name.contains("f16") && name.contains("f32")
}

fn waits(name: &str) -> bool {
    name.starts_with("s_nop") || name.starts_with("s_waitcnt") || name.starts_with("s_delay_alu")
}

/// A plain VALU instruction that can be copied out of its program: what
/// [`super::Lower::runs_alone`] takes, without its waterfall and lane mask checks (no compares
/// are taken).
fn liftable(i: &ir::Inst) -> bool {
    let Fields::Valu(v) = &i.fields else {
        return false;
    };
    let name = i.name();
    v.dpp.is_none()
        && !["v_cmp", "v_interp", "v_mbcnt", "v_movrel"].iter().any(|p| name.starts_with(p))
        && !name.contains("lane")
        && i.writes.iter().all(|w| matches!(w.0, Reg::V(_)))
        && !v.src.iter().any(|o| match *o {
            Operand::Scalar { code, dwords } => code < EXEC + 2 && code + dwords > EXEC,
            Operand::Const(c) => (251..=254).contains(&c),
            _ => false,
        })
        && touched(std::slice::from_ref(i)).is_some()
}

/// The longest stretches of [`liftable`] instructions in each block (waits and nops skipped)
/// that have an f16 result.
fn windows(p: &ir::Program) -> Vec<Vec<ir::Inst>> {
    let mut out = Vec::new();
    for b in &p.blocks {
        let mut cur: Vec<ir::Inst> = Vec::new();
        for i in &b.insts {
            if liftable(i) {
                cur.push(i.clone());
            } else if !waits(i.name()) {
                out.push(std::mem::take(&mut cur));
            }
        }
        out.push(cur);
    }
    out.retain(|w| w.iter().any(|i| result_kind(i.name()) == Kind::Half));
    out
}

/// A stored VGPR of a lifted window: the window, the register, what wrote it last and the
/// halves to compare (bit 0 low, bit 1 high).
struct Output {
    window: usize,
    reg: u8,
    kind: Kind,
    writer: &'static str,
    halves: u8,
}

/// One program of `windows`: every VGPR they read seeded per lane from s0..s3 (see
/// [`f16_rounding_of_the_title_fade`]), SGPRs fixed in [0.25, 1), all lanes on, and after each
/// window every VGPR it writes stored, 32 dwords per register.
fn lift_windows(windows: &[Vec<ir::Inst>]) -> (Vec<u32>, Vec<Output>) {
    let (mut used_s, mut used_v) = (BTreeSet::new(), BTreeSet::new());
    for w in windows {
        let (s, v) = touched(w).unwrap();
        used_s.extend(s);
        used_v.extend(v);
    }
    used_s.extend(0..6);
    let quad = (8..=100u8).rev().step_by(4).find(|&q| (q..q + 4).all(|c| !used_s.contains(&c))).unwrap();
    let c: Vec<u8> = (6..=105u8)
        .rev()
        .filter(|c| !used_s.contains(c) && !(quad..quad + 4).contains(c))
        .take(4)
        .collect();
    let l: Vec<u32> = (0..=255u8).rev().filter(|n| !used_v.contains(n)).take(3).map(u32::from).collect();
    let ([cs, cb, cl, ch], [lane, at, t]) = ([c[0], c[1], c[2], c[3]], [l[0], l[1], l[2]]);
    let mut w = vec![s_mov_b32(5, 0x80), s_mov_b32(EXEC, 193)];
    w.extend(s_load(2, quad as u32, 4, 0));
    w.extend([s_mov_b32(cs, 0), s_mov_b32(cb, 1), s_mov_b32(cl, 2), s_mov_b32(ch, 3)]);
    w.push(vop1(0x01, lane, v(0))); // v_mov_b32 lane, v0
    w.push(vop2(0x1A, at, k(2), lane)); // v_lshlrev_b32 at, 2, lane
    let vt = v(t as u16);
    let mut outputs = Vec::new();
    for (j, win) in windows.iter().enumerate() {
        let (mut f32_reads, mut seen) = (BTreeSet::new(), BTreeSet::new());
        for i in win {
            for &(r, _) in &i.reads {
                if seen.insert(r) && reads_f32(i, r) {
                    f32_reads.insert(r);
                }
            }
            seen.extend(i.writes.iter().map(|w| w.0));
        }
        let (s, regs) = touched(win).unwrap();
        let ju = j as u32;
        for c in s {
            if c == ir::VCC {
                w.extend([vopc(0xC4, LITERAL, lane), 7 + ju % 19]); // v_cmp_gt_u32 vcc, n, lane
            } else if c != ir::VCC + 1 {
                let cu = c as u32;
                let value = if f32_reads.contains(&Reg::S(c)) {
                    0x3E80_0000 + ((cu * 0x1_2345 + ju * 0x777) & 0x7F_FFFF)
                } else {
                    0x3400_3800 + ((cu * 0x0107_0003 + ju * 0x0011_0005) & 0x03FF_03FF)
                };
                w.extend([s_mov_b32(c, LITERAL as u8), value]);
            }
        }
        for r in regs {
            let ru = r as u32;
            let jitter = (ru * 5 + ju * 3) % 16 | ((ru * 7 + ju) % 16) << 16;
            w.extend([
                vop2(0x0B, t, ch as u16, lane), // v_mul_u32_u24 t, s_step_hi, lane
                vop2(0x1A, t, k(16), t),
                vop2(0x0B, ru, cl as u16, lane), // v_mul_u32_u24 r, s_step_lo, lane
                vop2(0x25, ru, vt, ru),
                vop2(0x25, ru, cb as u16, ru), // v_add_nc_u32 r, s_base, r
                vop2(0x25, ru, LITERAL, ru),
                jitter,
                // the low half's sign from lane bit 0, the high half's from bit 1
                vop2(0x1A, t, k(15), lane),
                vop2(0x1B, t, cs as u16, t),
                vop2(0x1D, ru, vt, ru),
                vop2(0x1A, t, k(30), lane),
                vop2(0x1B, t, cs as u16, t),
                vop2(0x1D, ru, vt, ru),
            ]);
            if f32_reads.contains(&Reg::V(r)) {
                // the low half as an f32, with mantissa bits below the half's
                w.push(vop1(0x0B, ru, v(r as u16))); // v_cvt_f32_f16
                w.extend([vop2(0x0B, t, LITERAL, lane), 0x1F3]);
                w.push(vop2(0x25, ru, vt, ru));
                w.extend([vop2(0x25, ru, LITERAL, ru), (ru * 0x2F + ju * 0x65) & 0x1FFF]);
            }
        }
        let mut last: std::collections::BTreeMap<u8, &ir::Inst> = Default::default();
        for i in win {
            if !waits(i.name()) {
                w.extend(&i.machine.words);
            }
            for &(r, _) in &i.writes {
                if let Reg::V(n) = r {
                    last.insert(n, i);
                }
            }
        }
        for (&r, i) in &last {
            // buffer_store_dword r, at offen, then at += 128
            w.extend([0xE070_1000, 0x80 << 24 | (quad as u32 / 4) << 16 | (r as u32) << 8 | at]);
            w.extend([vop2(0x25, at, LITERAL, at), 128]);
            let name = i.name();
            let Fields::Valu(fv) = &i.fields else {
                unreachable!()
            };
            let halves = if name.starts_with("v_pk_") || name.starts_with("v_cvt_pkrtz") {
                3
            } else if name.starts_with("v_fma_mixhi") || name.starts_with("v_mad_mixhi") || fv.opsel & 8 != 0 {
                2
            } else {
                1
            };
            outputs.push(Output { window: j, reg: r, kind: result_kind(name), writer: name, halves });
        }
    }
    w.push(S_ENDPGM);
    (w, outputs)
}

/// Differences of one output over its lanes in f16 ulps (an f32 difference / 2^13), the
/// switch off minus on: `sum` by value, `mag` by magnitude (negative is toward zero).
#[derive(Default)]
struct Tally {
    values: u64,
    differ: u64,
    nans: u64,
    max: f64,
    sum: f64,
    mag: f64,
}

impl Tally {
    fn add(&mut self, d: f64, m: f64) {
        self.values += 1;
        self.differ += u64::from(d != 0.0);
        self.max = self.max.max(d.abs());
        self.sum += d;
        self.mag += m;
    }

    fn merge(&mut self, o: &Tally) {
        self.values += o.values;
        self.differ += o.differ;
        self.nans += o.nans;
        self.max = self.max.max(o.max);
        self.sum += o.sum;
        self.mag += o.mag;
    }

    fn line(&self) -> String {
        let n = self.values.max(1) as f64;
        format!(
            "values {} differ {} max {:.3} bias {:+.4} mag {:+.4} nan {}",
            self.values,
            self.differ,
            self.max,
            self.sum / n,
            self.mag / n,
            self.nans
        )
    }
}

fn compare(t: &mut Tally, kind: Kind, halves: u8, off: u32, on: u32) {
    match kind {
        Kind::Half => {
            for h in 0..2 {
                if halves >> h & 1 == 0 {
                    continue;
                }
                let (a, b) = (off >> (16 * h) & 0xFFFF, on >> (16 * h) & 0xFFFF);
                let nan = |x: u32| x & 0x7C00 == 0x7C00 && x & 0x3FF != 0;
                if nan(a) || nan(b) {
                    t.nans += u64::from(nan(a) != nan(b));
                    continue;
                }
                let ord = |x: u32| if x & 0x8000 != 0 { -i64::from(x & 0x7FFF) } else { i64::from(x) };
                let mag = i64::from(a & 0x7FFF) - i64::from(b & 0x7FFF);
                t.add((ord(a) - ord(b)) as f64, mag as f64);
            }
        }
        Kind::F32 => {
            let nan = |x: u32| x & 0x7F80_0000 == 0x7F80_0000 && x & 0x7F_FFFF != 0;
            if nan(off) || nan(on) {
                t.nans += u64::from(nan(off) != nan(on));
                return;
            }
            let ord = |x: u32| if x >> 31 != 0 { -i64::from(x & 0x7FFF_FFFF) } else { i64::from(x) };
            let mag = i64::from(off & 0x7FFF_FFFF) - i64::from(on & 0x7FFF_FFFF);
            t.add((ord(off) - ord(on)) as f64 / 8192.0, mag as f64 / 8192.0);
        }
        Kind::Other => t.add(f64::from(u8::from(off != on)), 0.0),
    }
}

/// VICTORY_F16_NEAREST off against on for the pixel shaders in `VICTORY_F16_REPLAY`
/// (`<hash>.pixel.cdxil` containers from VICTORY_SHADER_SAVE). The stretches of plain VALU
/// code with an f16 result ([`windows`]) are copied into one compute program per shader and
/// lowered with the switch off and on. Every VGPR they read gets a pair of halves per lane
/// (an f32 made from the low one when it is read as an f32) from three ranges: [0, 0.9],
/// [0.06, 3.7] and [-0.9, 0.9]. Every register they write is compared, and the differences
/// per output go to `<hash>-f16.txt` in the same folder. Nothing is asserted.
#[test]
#[ignore = "needs VICTORY_F16_REPLAY"]
fn f16_rounding_of_the_title_fade() {
    let Ok(dir) = std::env::var("VICTORY_F16_REPLAY") else {
        eprintln!("VICTORY_F16_REPLAY unset; skipped");
        return;
    };
    // s0 sign mask, s1 base pair, s2 low step, s3 high step
    let cases: [(&str, [u32; 4]); 3] = [
        ("0..0.9", [0, 0x2000_1C00, 0x0FF, 0x0E3]),
        ("0.06..3.7", [0, 0x2C00_3000, 0x0A3, 0x0BF]),
        ("-0.9..0.9", [0x8000_8000, 0x2000_1C00, 0x0FF, 0x0E3]),
    ];
    let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let hosts: Vec<(&str, Host)> = [("hardware", false), ("warp", true)]
        .into_iter()
        .filter_map(|(label, warp)| match Host::new(warp) {
            Ok(h) => Some((label, h)),
            Err(_) => {
                eprintln!("no SM 6.6 {label} device; skipped");
                None
            }
        })
        .collect();
    assert!(!hosts.is_empty(), "no SM 6.6 device");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.ends_with(".pixel.cdxil"))
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no .pixel.cdxil in {dir}");
    for file in names {
        let hash = file.split('.').next().unwrap();
        let bytes = std::fs::read(format!("{dir}/{file}")).unwrap();
        let c = Container::parse(&bytes).unwrap();
        let part = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC)).unwrap();
        let ps = crate::native_programs(part.data)
            .unwrap()
            .into_iter()
            .find(|p| p.stage == HwStage::Ps)
            .unwrap();
        let p = ir::build(&ps.code, ps.wave64, ps.vgprs).unwrap();
        let wins = windows(&p);
        let insts: usize = wins.iter().map(Vec::len).sum();
        let halves = wins.iter().flatten().filter(|i| result_kind(i.name()) == Kind::Half).count();
        let (code, outputs) = lift_windows(&wins);
        let native = native(&code, false, 256);
        let dwords = outputs.len() as u32 * 32;
        let shaders: Vec<HostShader> = [false, true]
            .into_iter()
            .map(|nearest| {
                super::TEST_F16_NEAREST.set(Some(nearest));
                let (l, _) = lowered(&native, true, true);
                super::TEST_F16_NEAREST.set(None);
                host_shader(l)
            })
            .collect();
        let mut report = format!(
            "{hash}: wave{} PS, {} windows of {insts} VALU insts, {halves} with an f16 result, {} outputs\n",
            if ps.wave64 { 64 } else { 32 },
            wins.len(),
            outputs.len()
        );
        eprint!("{report}");
        for (label, host) in &hosts {
            for (case, inputs) in &cases {
                let off = run(host, &shaders[0], inputs, dwords);
                let on = run(host, &shaders[1], inputs, dwords);
                let mut kinds = [("f16", Kind::Half), ("f32", Kind::F32), ("bits", Kind::Other)]
                    .map(|(name, kind)| (name, kind, Tally::default(), 0));
                let mut lines = String::new();
                for (n, o) in outputs.iter().enumerate() {
                    let mut t = Tally::default();
                    for i in n * 32..n * 32 + 32 {
                        compare(&mut t, o.kind, o.halves, off[i], on[i]);
                    }
                    let k = kinds.iter_mut().find(|k| k.1 == o.kind).unwrap();
                    k.2.merge(&t);
                    k.3 += usize::from(o.kind != Kind::Other && t.max > 1.0);
                    if t.differ > 0 || t.nans > 0 {
                        lines += &format!("    w{} v{} {}: {}\n", o.window, o.reg, o.writer, t.line());
                    }
                }
                let mut head = format!("{label} {case}:\n");
                for (name, _, t, over) in &kinds {
                    head += &format!("  {name}: {}, outputs above 1 ulp {over}\n", t.line());
                }
                eprint!("{head}");
                report += &head;
                report += &lines;
            }
        }
        std::fs::write(format!("{dir}/{hash}-f16.txt"), &report).unwrap();
        eprintln!("{hash}: report in {dir}/{hash}-f16.txt");
    }
}

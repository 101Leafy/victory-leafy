//! x86-64 instruction lengths, read off the opcode maps in the AMD64 and Intel manuals: legacy
//! prefixes, REX, the one-byte, `0f`, `0f 38` and `0f 3a` maps, VEX, EVEX and XOP, then ModRM,
//! SIB, displacement and immediate. It finds where an instruction ends and where its operands
//! are; past that it does not check that the instruction exists.

/// One decoded instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Insn {
    pub len: usize,
    /// 0 for the one-byte map, 1 for `0f`, 2 for `0f 38`, 3 for `0f 3a`. VEX and EVEX use the
    /// same numbers (EVEX adds 5 and 6), XOP its own 8 to 10.
    pub map: u8,
    pub op: u8,
    pub modrm: Option<u8>,
    pub sib: Option<u8>,
    pub disp: i32,
    /// Where the immediate starts.
    pub imm_at: usize,
    /// REX as `0100WRXB`, 0 without one. VEX, EVEX and XOP fill it from their own bits.
    pub rex: u8,
    pub opsize: bool,
    pub addr32: bool,
    /// The last of `f2` and `f3`, 0 for neither.
    pub rep: u8,
    /// The last segment prefix, 0 for none.
    pub segment: u8,
    pub vex: bool,
}

#[derive(Clone, Copy)]
enum Imm {
    No,
    B,
    W,
    /// 2 bytes with `66`, else 4.
    Z,
    /// 8 bytes with REX.W, 2 with `66`, else 4.
    V,
    /// `enter`: a word and a byte.
    Enter,
    /// A full address: 8 bytes, 4 with `67`.
    Moffs,
    Rel32,
}

fn one_byte(op: u8) -> Option<(bool, Imm)> {
    use Imm::*;
    Some(match op {
        0x00..=0x3f => match op & 7 {
            0..=3 => (true, No),
            4 => (false, B),
            5 => (false, Z),
            // segment pushes, daa and the like; the prefixes and 0f never get here
            _ => return None,
        },
        0x50..=0x5f => (false, No),
        0x63 => (true, No),
        0x68 => (false, Z),
        0x69 => (true, Z),
        0x6a => (false, B),
        0x6b => (true, B),
        0x6c..=0x6f => (false, No),
        0x70..=0x7f => (false, B),
        0x80 | 0x83 => (true, B),
        0x81 => (true, Z),
        0x84..=0x8f => (true, No),
        0x90..=0x99 | 0x9b..=0x9f => (false, No),
        0xa0..=0xa3 => (false, Moffs),
        0xa4..=0xa7 | 0xaa..=0xaf => (false, No),
        0xa8 => (false, B),
        0xa9 => (false, Z),
        0xb0..=0xb7 => (false, B),
        0xb8..=0xbf => (false, V),
        0xc0 | 0xc1 | 0xc6 => (true, B),
        0xc7 => (true, Z),
        0xc2 | 0xca => (false, W),
        0xc3 | 0xc9 | 0xcb | 0xcc | 0xcf => (false, No),
        0xc8 => (false, Enter),
        0xcd => (false, B),
        0xd0..=0xd3 | 0xd8..=0xdf => (true, No),
        0xd7 => (false, No),
        0xe0..=0xe7 | 0xeb => (false, B),
        0xe8 | 0xe9 => (false, Rel32),
        0xec..=0xef | 0xf1 | 0xf4 | 0xf5 | 0xf8..=0xfd => (false, No),
        // f6 and f7 add an immediate for test, below
        0xf6 | 0xf7 | 0xfe | 0xff => (true, No),
        _ => return None,
    })
}

fn two_byte(op: u8) -> Option<(bool, Imm)> {
    use Imm::*;
    Some(match op {
        0x00..=0x03 | 0x0d => (true, No),
        0x05..=0x09 | 0x0b | 0x0e => (false, No),
        // 3DNow!: its opcode is the last byte, read here as an immediate
        0x0f => (true, B),
        0x10..=0x23 | 0x28..=0x2f => (true, No),
        0x30..=0x37 => (false, No),
        0x40..=0x6f => (true, No),
        0x70..=0x73 => (true, B),
        // 78 with 66 or f2 (SSE4a) takes two immediate bytes, below
        0x74..=0x76 | 0x78 | 0x79 | 0x7c..=0x7f => (true, No),
        0x77 => (false, No),
        0x80..=0x8f => (false, Rel32),
        0x90..=0x9f => (true, No),
        0xa0..=0xa2 | 0xa8..=0xaa | 0xc8..=0xcf => (false, No),
        0xa4 | 0xac | 0xba | 0xc2 | 0xc4..=0xc6 => (true, B),
        0xa3 | 0xa5 | 0xab | 0xad..=0xb9 | 0xbb..=0xc1 | 0xc3 | 0xc7 => (true, No),
        0xd0..=0xff => (true, No),
        _ => return None,
    })
}

/// VEX and EVEX: does the opcode take an immediate byte.
fn vex_imm(map: u8, op: u8) -> bool {
    map == 3 || (map == 1 && matches!(op, 0x70..=0x73 | 0xc2 | 0xc4..=0xc6))
}

/// The instruction at the start of `code`, or `None` when it is cut short, longer than 15
/// bytes, or not valid in 64-bit mode.
pub(crate) fn decode(code: &[u8]) -> Option<Insn> {
    let mut i = Insn::default();
    let mut at = 0;
    loop {
        let b = *code.get(at)?;
        match b {
            // REX counts only right before the opcode
            0x40..=0x4f => i.rex = b,
            0x26 | 0x2e | 0x36 | 0x3e | 0x64 | 0x65 => i.segment = b,
            0x66 => i.opsize = true,
            0x67 => i.addr32 = true,
            0xf0 => {}
            0xf2 | 0xf3 => i.rep = b,
            _ => break,
        }
        if b & 0xf0 != 0x40 {
            i.rex = 0;
        }
        at += 1;
        if at >= 15 {
            return None;
        }
    }

    let b = *code.get(at)?;
    at += 1;
    let (modrm, imm) = match b {
        0x0f => {
            let b = *code.get(at)?;
            at += 1;
            match b {
                0x38 | 0x3a => {
                    i.map = if b == 0x38 { 2 } else { 3 };
                    i.op = *code.get(at)?;
                    at += 1;
                    (true, if b == 0x3a { Imm::B } else { Imm::No })
                }
                _ => {
                    i.map = 1;
                    i.op = b;
                    two_byte(b)?
                }
            }
        }
        0xc4 | 0xc5 => {
            let p = *code.get(at)?;
            if b == 0xc5 {
                i.map = 1;
                i.rex = 0x40 | ((!p >> 5) & 4);
                set_pp(&mut i, p);
                at += 1;
            } else {
                let q = *code.get(at + 1)?;
                i.map = p & 0x1f;
                i.rex = 0x40 | ((q >> 4) & 8) | ((!p >> 5) & 7);
                set_pp(&mut i, q);
                at += 2;
            }
            i.vex = true;
            i.op = *code.get(at)?;
            at += 1;
            if !(1..=3).contains(&i.map) {
                return None;
            }
            // vzeroupper and vzeroall have no ModRM
            let modrm = !(i.map == 1 && i.op == 0x77);
            (modrm, if vex_imm(i.map, i.op) { Imm::B } else { Imm::No })
        }
        0x62 => {
            let (p0, p1) = (*code.get(at)?, *code.get(at + 1)?);
            code.get(at + 2)?;
            i.map = p0 & 7;
            if !matches!(i.map, 1..=3 | 5 | 6) {
                return None;
            }
            i.rex = 0x40 | ((p1 >> 4) & 8) | ((!p0 >> 5) & 7);
            set_pp(&mut i, p1);
            i.vex = true;
            at += 3;
            i.op = *code.get(at)?;
            at += 1;
            (true, if vex_imm(i.map, i.op) { Imm::B } else { Imm::No })
        }
        // XOP; with a map field under 8 it is pop r/m
        0x8f if code.get(at)? & 0x1f >= 8 => {
            let (p, q) = (*code.get(at)?, *code.get(at + 1)?);
            i.map = p & 0x1f;
            if i.map > 10 {
                return None;
            }
            i.rex = 0x40 | ((q >> 4) & 8) | ((!p >> 5) & 7);
            set_pp(&mut i, q);
            i.vex = true;
            at += 2;
            i.op = *code.get(at)?;
            at += 1;
            let imm = match i.map {
                8 => Imm::B,
                10 => Imm::Rel32,
                _ => Imm::No,
            };
            (true, imm)
        }
        _ => {
            i.op = b;
            one_byte(b)?
        }
    };

    let mut imm_len = match imm {
        Imm::No => 0,
        Imm::B => 1,
        Imm::W => 2,
        Imm::Enter => 3,
        Imm::Z if i.opsize && i.rex & 8 == 0 => 2,
        Imm::Z | Imm::Rel32 => 4,
        Imm::V if i.rex & 8 != 0 => 8,
        Imm::V if i.opsize => 2,
        Imm::V => 4,
        Imm::Moffs if i.addr32 => 4,
        Imm::Moffs => 8,
    };

    if modrm {
        let m = *code.get(at)?;
        at += 1;
        i.modrm = Some(m);
        let (mode, mut base) = (m >> 6, m & 7);
        if mode != 3 {
            if base == 4 {
                let s = *code.get(at)?;
                at += 1;
                i.sib = Some(s);
                base = s & 7;
            }
            // base 5 with mode 0: rip-relative without a SIB, no base register with one
            match mode {
                0 if base == 5 => {
                    i.disp = i32::from_le_bytes(code.get(at..at + 4)?.try_into().ok()?);
                    at += 4;
                }
                1 => {
                    i.disp = i32::from(*code.get(at)? as i8);
                    at += 1;
                }
                2 => {
                    i.disp = i32::from_le_bytes(code.get(at..at + 4)?.try_into().ok()?);
                    at += 4;
                }
                _ => {}
            }
        }
        let reg = (m >> 3) & 7;
        if !i.vex && i.map == 0 && reg < 2 {
            imm_len += match i.op {
                0xf6 => 1,
                0xf7 if i.opsize && i.rex & 8 == 0 => 2,
                0xf7 => 4,
                _ => 0,
            };
        }
        if !i.vex && i.map == 1 && i.op == 0x78 && (i.opsize || i.rep == 0xf2) {
            imm_len = 2;
        }
    }

    i.imm_at = at;
    i.len = at + imm_len;
    (i.len <= 15 && i.len <= code.len()).then_some(i)
}

/// The prefix a VEX, EVEX or XOP `pp` field stands for.
fn set_pp(i: &mut Insn, byte: u8) {
    match byte & 3 {
        1 => i.opsize = true,
        2 => i.rep = 0xf3,
        3 => i.rep = 0xf2,
        _ => {}
    }
}

impl Insn {
    /// The memory operand's address before any segment base, or `None` for a register operand.
    /// `reg` reads a general register by number; `next` is the address after the instruction.
    pub fn address(&self, reg: impl Fn(u8) -> u64, next: u64) -> Option<u64> {
        let m = self.modrm?;
        let mode = m >> 6;
        if mode == 3 {
            return None;
        }
        let b = (self.rex & 1) << 3;
        let disp = i64::from(self.disp) as u64;
        let at = match self.sib {
            None if mode == 0 && m & 7 == 5 => next.wrapping_add(disp),
            None => reg((m & 7) | b).wrapping_add(disp),
            Some(s) => {
                let index = ((s >> 3) & 7) | ((self.rex & 2) << 2);
                let scaled = if index == 4 { 0 } else { reg(index) << (s >> 6) };
                let base = if mode == 0 && s & 7 == 5 { 0 } else { reg((s & 7) | b) };
                base.wrapping_add(scaled).wrapping_add(disp)
            }
        };
        Some(if self.addr32 { at & 0xffff_ffff } else { at })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(hex: &str) -> Vec<u8> {
        hex.split_whitespace().map(|b| u8::from_str_radix(b, 16).unwrap()).collect()
    }

    #[test]
    fn lengths() {
        let table = [
            ("90", 1),
            ("c3", 1),
            ("cc", 1),
            ("55", 1),
            ("48 89 e5", 3),
            ("48 83 ec 20", 4),
            ("48 81 ec 00 01 00 00", 7),
            ("e8 10 20 30 40", 5),
            ("e9 10 20 30 40", 5),
            ("eb fe", 2),
            ("74 05", 2),
            ("0f 84 10 20 30 40", 6),
            ("48 b8 01 02 03 04 05 06 07 08", 10),
            ("b8 01 00 00 00", 5),
            ("66 b8 01 00", 4),
            ("48 66 b8 01 00", 5),
            ("48 8b 05 10 20 30 40", 7),
            ("48 8b 44 24 08", 5),
            ("48 8b 04 24", 4),
            ("48 8b 84 24 00 01 00 00", 8),
            ("41 8b 45 00", 4),
            ("8b 04 25 10 20 30 40", 7),
            ("c7 44 24 08 01 00 00 00", 8),
            ("48 c7 c0 ff ff ff ff", 7),
            ("66 c7 44 24 08 01 00", 7),
            ("c6 05 10 20 30 40 01", 7),
            ("f6 c1 01", 3),
            ("f7 c1 00 00 01 00", 6),
            ("f7 d8", 2),
            ("66 0f 1f 44 00 00", 6),
            ("0f 1f 80 00 00 00 00", 7),
            ("66 2e 0f 1f 84 00 00 00 00 00", 10),
            ("f3 0f 1e fa", 4),
            ("0f 01 fa", 3),
            ("0f 01 fb", 3),
            ("0f 01 fc", 3),
            ("0f 01 fd", 3),
            ("66 0f 78 c1 08 10", 6),
            ("f2 0f 78 ca 08 10", 6),
            ("66 0f 79 ca", 4),
            ("f2 0f 79 ca", 4),
            ("f2 0f 2b 00", 4),
            ("f3 0f 2b 4c 24 10", 6),
            ("0f 28 c1", 3),
            ("66 0f 3a 0f c1 08", 6),
            ("66 0f 38 00 c1", 5),
            ("c5 f8 77", 3),
            ("c5 fc 28 c1", 4),
            ("c5 f9 70 c1 1b", 5),
            ("c4 e2 7d 18 05 10 20 30 40", 9),
            ("c4 e3 7d 18 c1 01", 6),
            ("62 f1 7c 48 28 c1", 6),
            ("62 f1 7c 48 29 44 24 01", 8),
            ("8f e9 78 80 c1", 5),
            ("8f 04 24", 3),
            ("a1 01 02 03 04 05 06 07 08", 9),
            ("67 a1 01 02 03 04", 6),
            ("c8 10 00 00", 4),
            ("c2 08 00", 3),
            ("0f ba e0 03", 4),
            ("0f a4 c2 04", 4),
            ("f0 48 0f b1 0a", 5),
            ("48 0f c7 0e", 4),
            ("f3 48 ab", 3),
            ("0f 0f c1 b4", 4),
            ("66 48 0f 7e c0", 5),
            ("65 48 8b 04 25 30 00 00 00", 9),
            ("48 8d 0d 10 20 30 40", 7),
            ("ff 15 10 20 30 40", 6),
            ("ff e0", 2),
            ("41 ff d3", 3),
            ("0f 0b", 2),
            ("0f a2", 2),
            ("0f 31", 2),
            ("66 66 66 66 66 66 66 66 66 66 66 66 66 66 90", 15),
        ];
        for (hex, len) in table {
            let mut code = bytes(hex);
            code.extend([0xcc; 4]);
            assert_eq!(decode(&code).map(|i| i.len), Some(len), "{hex}");
        }
    }

    #[test]
    fn refuses() {
        for hex in [
            "06",
            "d6",
            "0f 04",
            "48 8b 05 00",
            "66 66 66 66 66 66 66 66 66 66 66 66 66 66 66 90",
        ] {
            assert_eq!(decode(&bytes(hex)), None, "{hex}");
        }
    }

    #[test]
    fn fields() {
        let i = decode(&bytes("f2 0f 78 ca 08 10")).unwrap();
        assert_eq!((i.map, i.op, i.modrm, i.rep, i.imm_at), (1, 0x78, Some(0xca), 0xf2, 4));
        let i = decode(&bytes("c4 c1 7d 18 05 10 20 30 40")).unwrap();
        assert_eq!((i.vex, i.map, i.rex, i.opsize), (true, 1, 0x41, true));
    }

    #[test]
    fn addresses() {
        let reg = |n: u8| u64::from(n) * 0x100;
        // mov rax, [rip+0x40302010]
        let i = decode(&bytes("48 8b 05 10 20 30 40")).unwrap();
        assert_eq!(i.address(reg, 0x1000), Some(0x4030_3010));
        // mov eax, [r12+r9*4-8]
        let i = decode(&bytes("43 8b 44 8c f8")).unwrap();
        assert_eq!(i.address(reg, 0), Some(0xc00 + 0x900 * 4 - 8));
        // mov eax, [rsp]: no index
        let i = decode(&bytes("8b 04 24")).unwrap();
        assert_eq!(i.address(reg, 0), Some(0x400));
        // movntss [rsp+0x10], xmm1
        let i = decode(&bytes("f3 0f 2b 4c 24 10")).unwrap();
        assert_eq!(i.address(reg, 0), Some(0x410));
        // mov eax, [disp32]
        let i = decode(&bytes("8b 04 25 10 20 30 40")).unwrap();
        assert_eq!(i.address(reg, 0), Some(0x4030_2010));
        assert_eq!(decode(&bytes("0f 28 c1")).unwrap().address(reg, 0), None);
    }
}

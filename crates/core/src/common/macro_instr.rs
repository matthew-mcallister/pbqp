#![allow(dead_code)]

use crate::common::reg_alloc::AllowedRegisters;

use super::ssa_model::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Register {
    C = 1,
    X = 2,
    Y = 3,
}

impl TryFrom<usize> for Register {
    type Error = ();

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::C),
            2 => Ok(Self::X),
            3 => Ok(Self::Y),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddrMode {
    Imm = 0,
    Dir = 1,
    Reg,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MachineOpCode {
    Clc,
    Sec,
    Adc,
    Sbc,
    Inc,
    Inx,
    Iny,
    And,
    Ora,
    Eor,
}

impl std::fmt::Display for MachineOpCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use MachineOpCode::*;
        match self {
            Clc => write!(f, "CLC"),
            Sec => write!(f, "SEC"),
            Adc => write!(f, "ADC"),
            Sbc => write!(f, "SBC"),
            Inc => write!(f, "INC"),
            Inx => write!(f, "INX"),
            Iny => write!(f, "INY"),
            And => write!(f, "AND"),
            Ora => write!(f, "ORA"),
            Eor => write!(f, "EOR"),
        }
    }
}

impl MachineOpCode {
    const fn cycles(self, mode: AddrMode) -> u32 {
        use AddrMode::*;
        use MachineOpCode::*;
        match (self, mode) {
            (Clc | Sec, Reg) => 2,
            (Adc | Sbc, Imm) => 3,
            (Adc | Sbc, Dir) => 4,
            (Inc, Reg) => 2,
            (Inc, Dir) => 7,
            (Inx, Reg) => 2,
            (Iny, Reg) => 2,
            (And | Ora | Eor, Imm) => 3,
            (And | Ora | Eor, Dir) => 4,
            _ => unreachable!(),
        }
    }

    const fn output(self) -> Option<Register> {
        use Register::*;
        match self {
            MachineOpCode::Clc => None,
            MachineOpCode::Sec => None,
            MachineOpCode::Adc => Some(C),
            MachineOpCode::Sbc => Some(C),
            MachineOpCode::Inc => Some(C),
            MachineOpCode::Inx => Some(X),
            MachineOpCode::Iny => Some(Y),
            MachineOpCode::And => Some(C),
            MachineOpCode::Ora => Some(C),
            MachineOpCode::Eor => Some(C),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum MacroOp {
    Simple(MachineOpCode),
    /// CLC + ADC
    ClcAdc,
    /// SEC + SBC
    SecSbc,
    /// Computes `~x + c = -x - 1 + c`.
    EorFfInc(Const),
    /// Increments or decrements the accumulator a constant number of times.
    Inc(Const),
    /// Increments or decrements the X register a constant number of times.
    Inx(Const),
    /// Increments or decrements the Y register a constant number of times.
    Iny(Const),
}

impl MacroOp {
    fn cycles(self, mode: AddrMode) -> u32 {
        fn inc_cycles(c: Const) -> u32 {
            2 * i64::from(c).abs() as u32
        }

        match self {
            Self::Simple(code) => code.cycles(mode),
            Self::ClcAdc => 2 + MachineOpCode::Adc.cycles(mode),
            Self::SecSbc => 2 + MachineOpCode::Sbc.cycles(mode),
            Self::Inc(c) | Self::Inx(c) | Self::Iny(c) => inc_cycles(c),
            Self::EorFfInc(c) => 4 + inc_cycles(c),
        }
    }

    fn output(self) -> Option<Register> {
        use Register::*;
        match self {
            MacroOp::Simple(opcode) => opcode.output(),
            MacroOp::ClcAdc => Some(C),
            MacroOp::SecSbc => Some(C),
            MacroOp::EorFfInc(_) => Some(C),
            MacroOp::Inc(_) => Some(C),
            MacroOp::Inx(_) => Some(X),
            MacroOp::Iny(_) => Some(Y),
        }
    }
}

// This default is not very useful, it's so MacroInstr has a Default impl
impl Default for MacroOp {
    fn default() -> Self {
        Self::Simple(MachineOpCode::Adc)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MacroInstr {
    pub opcode: MacroOp,
    pub copy_c: Option<Operand>,
    pub copy_x: Option<Operand>,
    pub copy_y: Option<Operand>,
    pub operand: Option<Operand>,
    pub reg_out: Option<Register>,
    pub kill_c: bool,
    pub kill_x: bool,
    pub kill_y: bool,
}

impl MacroInstr {
    pub fn cycles(&self) -> u32 {
        let addr_mode = self
            .operand
            .and_then(|op| op.addr_mode())
            .unwrap_or(AddrMode::Reg);
        self.opcode.cycles(addr_mode)
    }

    /// Returns the registers that are not clobbered/interfered with by the
    /// instruction.
    pub fn free_registers(&self) -> AllowedRegisters {
        let mut allowed = AllowedRegisters::default();
        if self.copy_c.is_none() {
            allowed.set_c(true);
        }
        if self.copy_x.is_none() {
            allowed.set_x(true);
        }
        if self.copy_y.is_none() {
            allowed.set_y(true);
        }
        allowed
    }
}

impl std::fmt::Display for MacroInstr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Register::*;

        fn fmt_inc(f: &mut std::fmt::Formatter<'_>, reg: Register, c: Const) -> std::fmt::Result {
            let c: i64 = c.into();
            if c == 0 {
                return Ok(());
            }
            let prefix = if c < 0 { "DE" } else { "IN" };
            let suffix = match reg {
                C => "C",
                X => "X",
                Y => "Y",
            };
            for _ in 0..c.abs() {
                writeln!(f, "{}{}", prefix, suffix)?;
            }
            Ok(())
        }

        use MacroOp::*;
        match (self.opcode, self.operand) {
            (Simple(code), Some(op)) => writeln!(f, "{} {}", code, op),
            (Simple(code), None) => writeln!(f, "{}", code),
            (ClcAdc, Some(op)) => writeln!(f, "CLC\nADC {}", op),
            (SecSbc, Some(op)) => writeln!(f, "SEC\nSBC {}", op),
            (MacroOp::Inc(c), None) => fmt_inc(f, C, c),
            (MacroOp::Inx(c), None) => fmt_inc(f, X, c),
            (MacroOp::Iny(c), None) => fmt_inc(f, Y, c),
            (EorFfInc(c), None) => {
                writeln!(f, "EOR #$FFFF")?;
                fmt_inc(f, C, c)
            }
            _ => unreachable!(),
        }
    }
}

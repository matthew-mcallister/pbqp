// TODO: Overflow analysis so we can eliminate the clc/sec

use std::collections::HashSet;
use std::num::NonZeroU16;
use std::ops::{Add, Neg};

use super::macro_instr::MacroInstr;
use super::reg_alloc::AllowedRegisters;

use super::macro_instr::AddrMode;

/// Reference to a BB
// TODO: Have to replace with a stable ID
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Label(NonZeroU16);

impl From<Label> for u16 {
    fn from(value: Label) -> Self {
        u16::from(value.0) + 1
    }
}

impl From<Label> for usize {
    fn from(value: Label) -> Self {
        u16::from(value) as usize
    }
}

impl From<u16> for Label {
    fn from(value: u16) -> Self {
        Label(NonZeroU16::new(value + 1).unwrap())
    }
}

impl From<usize> for Label {
    fn from(value: usize) -> Self {
        Self::from(value as u16)
    }
}

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ".{}", u16::from(*self))
    }
}

/// Reference to a variable
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Variable(pub Label, pub u16);

impl Variable {
    pub fn block(&self) -> Label {
        self.0
    }

    pub fn instruction(&self) -> u16 {
        self.1
    }
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}.{}", self.0, self.1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Const(pub i64);

impl From<Const> for i64 {
    fn from(Const(x): Const) -> Self {
        x
    }
}

impl From<i64> for Const {
    fn from(x: i64) -> Self {
        Self(x)
    }
}

impl Neg for Const {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Add for Const {
    type Output = Self;

    fn add(self, Const(y): Const) -> Self::Output {
        Self(self.0 + y)
    }
}

impl std::fmt::Display for Const {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#${:04x}", i64::from(*self))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operand {
    Variable(Variable),
    Const(Const),
}

impl Operand {
    pub fn variable(self) -> Option<Variable> {
        match self {
            Self::Variable(var) => Some(var),
            _ => None,
        }
    }

    pub fn is_variable(self) -> bool {
        match self {
            Self::Variable(_) => true,
            _ => false,
        }
    }

    pub fn constant(self) -> Option<Const> {
        match self {
            Self::Const(c) => Some(c),
            _ => None,
        }
    }

    pub fn is_constant(self) -> bool {
        match self {
            Self::Const(_) => true,
            _ => false,
        }
    }

    pub fn addr_mode(&self) -> Option<AddrMode> {
        match self {
            Self::Variable(_) => Some(AddrMode::Dir),
            Self::Const(_) => Some(AddrMode::Imm),
        }
    }
}

impl std::fmt::Display for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Variable(x) => write!(f, "{}", x),
            Self::Const(x) => write!(f, "{}", x),
        }
    }
}

impl From<Const> for Operand {
    fn from(value: Const) -> Self {
        Self::Const(value)
    }
}

impl From<i64> for Operand {
    fn from(value: i64) -> Self {
        Self::from(Const::from(value))
    }
}

impl From<Variable> for Operand {
    fn from(value: Variable) -> Self {
        Self::Variable(value)
    }
}

#[derive(Debug)]
pub enum InstrOp {
    Neg(Operand),
    And([Operand; 2]),
    Or([Operand; 2]),
    Xor([Operand; 2]),
    Not(Operand),
    Add([Operand; 2]),
    Sub([Operand; 2]),
}

#[derive(Debug)]
pub enum Terminator {
    Jump(Label),
    If(Operand, [Label; 2]),
    Return(Option<Operand>),
}

impl Terminator {
    pub fn successors(&self) -> &[Label] {
        match self {
            Terminator::Jump(label) => std::slice::from_ref(label),
            Terminator::If(_, labels) => &labels[..],
            Terminator::Return(_) => &[],
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Terminator::Return(_))
    }
}

#[derive(Debug)]
pub struct Instr {
    pub operation: InstrOp,
    /// Estimated execution count.
    pub frequency: f32,
    pub expansions: Vec<MacroInstr>,
    /// We can compute the max amount of memory needed at a program point and,
    /// if desired, force variables to be stored in registers to save memory.
    pub num_stack_slots: u16,
    /// Record which operands are killed by this instruction.
    pub kill: [bool; 2],
    /// A variable that is anchored is assigned a unique stack slot for
    /// spilling. This is mandatory for variables whose live range spans
    /// multiple blocks with a common descendant. It is also mandatory for
    /// addressable variables.
    pub anchored: bool,
}

impl Instr {
    pub fn new(operation: InstrOp) -> Self {
        Self {
            operation,
            frequency: 0.0,
            expansions: Vec::new(),
            num_stack_slots: 0,
            kill: [false; 2],
            anchored: false,
        }
    }

    pub fn operands(&self) -> &[Operand] {
        match &self.operation {
            InstrOp::Neg(op) => std::slice::from_ref(op),
            InstrOp::And(ops) | InstrOp::Or(ops) | InstrOp::Xor(ops) => &ops[..],
            InstrOp::Not(op) => std::slice::from_ref(op),
            InstrOp::Add(ops) | InstrOp::Sub(ops) => &ops[..],
        }
    }

    /// Returns the registers which may be free in at least one instruction
    /// expansion.
    pub fn free_registers(&self) -> AllowedRegisters {
        self.expansions
            .iter()
            .map(|ex| ex.free_registers())
            .reduce(|e1, e2| e1.or(e2))
            .expect("instruction must be expanded")
    }
}

#[derive(Debug)]
pub struct BasicBlock {
    pub instructions: Vec<Instr>,
    pub terminator: Terminator,
    pub predecessors: Vec<Label>,
    /// A junction is an equivalence class of the set of edges of the CFG under
    /// the following equivalence relation:
    ///   (u, v) ~ (u, w)
    ///   (u, v) ~ (w, v)
    ///
    /// This allows us to guarantee that a variable will be placed in the same
    /// location in all predecessors of a block before it is entered.
    pub junction: Option<NonZeroU16>,
    pub live_in: HashSet<Variable>,
}

impl BasicBlock {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            terminator: Terminator::Return(None),
            predecessors: Vec::new(),
            junction: Default::default(),
            live_in: Default::default(),
        }
    }

    pub fn successors(&self) -> &[Label] {
        self.terminator.successors()
    }

    pub fn predecessors(&self) -> &[Label] {
        &self.predecessors[..]
    }

    pub fn is_terminal(&self) -> bool {
        self.terminator.is_terminal()
    }

    pub fn add_instr(&mut self, instr: Instr) -> u16 {
        let i = self.instructions.len() as u16;
        self.instructions.push(instr);
        i
    }
}

#[derive(Debug)]
pub struct Function {
    pub blocks: Vec<BasicBlock>,
}

impl Function {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    pub fn get_block_mut(&mut self, block: Label) -> &mut BasicBlock {
        &mut self.blocks[usize::from(block)]
    }

    pub fn get_instr(&self, var: Variable) -> &Instr {
        &self.blocks[usize::from(var.block())].instructions[var.instruction() as usize]
    }

    pub fn get_instr_mut(&mut self, var: Variable) -> &Instr {
        &mut self.blocks[usize::from(var.block())].instructions[var.instruction() as usize]
    }

    pub fn add_block(&mut self, block: BasicBlock) -> Label {
        let l = self.blocks.len();
        self.blocks.push(block);
        l.into()
    }
}

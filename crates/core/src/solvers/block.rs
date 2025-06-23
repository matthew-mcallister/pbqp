#![allow(dead_code)]

use std::collections::HashMap;
use std::rc::Rc;

use crate::common::{
    macro_instr::Register,
    ssa_model::{BasicBlock, Variable},
};

/// Tracks the state of a variable's stack slot while it is in a register.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StackState {
    /// The variable has no copy in memory and no assigned slot.
    Floating,
    /// The variable has a copy in memory.
    Copied,
    /// The variable has an assigned slot that is temporarily occupied by a
    /// different variable.
    Leased(Variable),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct State {
    pub register_info: [Option<(Variable, StackState)>; 3],
    // TODO:
    // m: bool,
    // n, z, c etc.
}

impl State {
    pub fn c(&self) -> Option<Variable> {
        Some(self.register_info[0]?.0)
    }

    pub fn x(&self) -> Option<Variable> {
        Some(self.register_info[1]?.0)
    }

    pub fn y(&self) -> Option<Variable> {
        Some(self.register_info[2]?.0)
    }

    pub fn register(&self, register: Register) -> Option<(Variable, StackState)> {
        self.register_info[register as usize - 1].clone()
    }

    pub fn registers(&self) -> impl Iterator<Item = (Variable, StackState)> + '_ {
        self.register_info.iter().filter_map(|&x| x)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct StateInfo {
    state: State,
    prev: Option<Rc<StateInfo>>,
    expansion: u8,
    cost: u32,
}

#[derive(Debug)]
struct Solver<'b> {
    block: &'b BasicBlock,
    num_live: u16,
    inst: u16,
    states: HashMap<State, Rc<StateInfo>>,
}

impl<'b> Solver<'b> {
    #[allow(unused_variables)]
    fn visit(&mut self, expansion: usize, prev: &Rc<StateInfo>) {
        todo!()
    }

    fn step(&mut self) {
        let prev_states = std::mem::take(&mut self.states);
        for e in 0..self.block.instructions[self.inst as usize].expansions.len() {
            for prev in prev_states.values() {
                self.visit(e, prev);
            }
        }

        // Advance to next instruction
        let instr = &self.block.instructions[self.inst as usize];
        self.inst += 1;
        self.num_live += 1;
        self.num_live -= instr.kill[0] as u16 + instr.kill[1] as u16;
    }
}

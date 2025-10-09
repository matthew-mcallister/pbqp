use std::iter::FusedIterator;

use fnv::FnvHashMap as HashMap;
use rcc_codegen::const_vec::{ConstMap, ConstVec};
use rcc_codegen::shuffle::{
    EntryIndexer, Input as ShuffleInput, RegisterShuffleEntry, ShuffleInstr, State as ShuffleState,
    deserialize_lut,
};

use crate::common::macro_instr::Register;
use crate::common::ssa_model::{Function, Variable};
use crate::solvers::block::{StackState, State};

#[derive(Debug)]
struct EntryMap {
    indexer: EntryIndexer,
    data: Vec<u8>,
}

impl EntryMap {
    fn new(input: &ShuffleInput, data: Vec<u8>) -> Self {
        let indexer = EntryIndexer::new(input);
        Self { indexer, data }
    }

    fn get(&self, state: &ShuffleState) -> RegisterShuffleEntry {
        let idx = self.indexer.index(&state);
        let start = idx as usize * RegisterShuffleEntry::SIZE;
        let end = start + RegisterShuffleEntry::SIZE;
        let bytes: [u8; 12] = self.data[start..end].try_into().unwrap();
        RegisterShuffleEntry::from_bytes(bytes).unwrap()
    }
}

#[derive(Debug)]
pub struct ShuffleSolver {
    entries: HashMap<ShuffleInput, EntryMap>,
}

#[derive(Debug)]
pub struct Solution<'solver> {
    solver: &'solver ShuffleSolver,
    input: ShuffleInput,
    output: ShuffleState,
}

impl<'solver> Solution<'solver> {
    pub fn cost(&self) -> u16 {
        self.solver.entries[&self.input].get(&self.output).cost
    }

    pub fn instructions(&self) -> impl Iterator<Item = ShuffleInstr> + 'solver {
        StateIter {
            entries: &self.solver.entries[&self.input],
            state: self.output,
        }
    }
}

#[derive(Debug)]
struct StateIter<'s> {
    entries: &'s EntryMap,
    state: ShuffleState,
}

impl<'s> Iterator for StateIter<'s> {
    type Item = ShuffleInstr;

    fn next(&mut self) -> Option<Self::Item> {
        let (prev, instr) = self.entries.get(&self.state).prev?;
        self.state = prev;
        Some(instr)
    }
}

impl<'s> FusedIterator for StateIter<'s> {}

#[derive(Clone, Copy, Debug)]
enum StackLocation {
    /// In a register, nowhere in memory
    Floating,
    /// In a movable slot
    Movable,
    /// In the immovable slot of the given variable.
    Immovable(Variable),
}

#[derive(Clone, Copy, Debug)]
struct Stack<'a> {
    anchored: &'a [Variable],
    locations: ConstMap<Variable, StackLocation, 6>,
}

impl<'a> Stack<'a> {
    fn new(anchored: &'a [Variable], state: &State) -> Self {
        let mut locations: ConstMap<Variable, StackLocation, 6> = Default::default();
        for (u, s) in state.registers() {
            let anchored = anchored.contains(&u);
            match (s, anchored) {
                (StackState::Floating, _) => {
                    locations.insert(u, StackLocation::Floating);
                }
                (StackState::Copied, false) => {
                    locations.insert(u, StackLocation::Movable);
                }
                (StackState::Copied, true) => {
                    locations.insert(u, StackLocation::Immovable(u));
                }
                (StackState::Leased(v), _) => {
                    locations.insert(v, StackLocation::Immovable(u));
                    locations.insert(u, StackLocation::Floating);
                }
            }
        }
        Self {
            anchored,
            locations,
        }
    }

    fn location(&self, var: Variable) -> StackLocation {
        if let Some(&loc) = self.locations.get(&var) {
            loc
        } else if self.anchored.contains(&var) {
            StackLocation::Immovable(var)
        } else {
            StackLocation::Movable
        }
    }
}

#[derive(Debug, Default)]
struct VariableMap {
    inner: ConstVec<Variable, 6>,
}

impl VariableMap {
    fn get(&mut self, var: Variable) -> u8 {
        if let Some(i) = self.inner.iter().position(|&x| x == var) {
            i as u8 + 1
        } else {
            self.inner.push(var);
            self.inner.len() as u8
        }
    }
}

impl ShuffleSolver {
    pub fn new() -> Self {
        let entries: HashMap<ShuffleInput, Vec<u8>> =
            deserialize_lut(super::shuffle_data::SHUFFLE_DATA).unwrap();
        let entries = entries
            .into_iter()
            .map(|(input, data)| (input, EntryMap::new(&input, data)))
            .collect();
        Self { entries }
    }

    pub fn solve<'a>(&'a self, f: &Function, initial: &State, finl: &State) -> Solution<'a> {
        let mut anchored: ConstVec<Variable, 6> = Default::default();
        for (u, _) in initial.registers().chain(finl.registers()) {
            if f.get_instr(u).anchored && !anchored.contains(&u) {
                anchored.push(u);
            }
        }
        self.solve_inner(&anchored[..], initial, finl)
    }

    fn solve_inner<'a>(
        &'a self,
        anchored: &[Variable],
        initial: &State,
        finl: &State,
    ) -> Solution<'a> {
        // Input registers
        let mut registers: [u8; 3] = [0; 3];
        let mut variable_map = VariableMap::default();
        for (reg, u) in [Register::C, Register::X, Register::Y]
            .into_iter()
            .filter_map(|r| Some((r, initial.register(r)?.0)))
        {
            registers[reg as usize - 1] = variable_map.get(u);
        }

        // Identify variable locations on stack
        let stack_vars = Stack::new(anchored, &initial);
        let mut movable: ConstVec<Variable, 3> = Default::default();
        let mut immovable: ConstMap<Variable, Variable, 3> = Default::default();
        for (u, _) in finl.registers() {
            match stack_vars.location(u) {
                StackLocation::Floating => {}
                StackLocation::Movable => {
                    movable.push(u);
                }
                StackLocation::Immovable(v) => {
                    immovable.insert(v, u);
                }
            }
        }

        // Map variables to stack slots
        let mut stack: [u8; 3] = [0; 3];
        let mut i = 0;
        for &v in movable.iter() {
            stack[i] = variable_map.get(v);
            i += 1;
        }
        stack[..i].sort();
        let mut slots: ConstMap<Variable, u8, 3> = Default::default();
        for (&u, &v) in immovable.iter() {
            stack[i] = variable_map.get(v);
            slots.insert(u, i as u8);
            i += 1;
        }

        let mut num_movable = movable.len() as u8;
        let mut num_immovable = immovable.len() as u8;
        if num_movable == 1 {
            // Having only one movable slot is pointless from the solver's perspective.
            num_movable -= 1;
            num_immovable += 1;
        }

        let input = ShuffleInput {
            reg: registers,
            stack,
            num_movable,
            num_immovable,
        };

        // Construct output state
        let stack_vars = Stack::new(anchored, &finl);

        let mut registers: [u8; 3] = [0; 3];
        for (r, u) in [Register::C, Register::X, Register::Y]
            .into_iter()
            .filter_map(|r| Some((r, finl.register(r)?.0)))
        {
            let var = variable_map.get(u);
            registers[r as usize - 1] = var as u8;
        }

        let mut stack: [u8; 3] = [0; 3];
        let mut i = 0;
        for (u, _) in initial.registers() {
            let var = variable_map.get(u);
            match stack_vars.location(u) {
                StackLocation::Floating => {}
                StackLocation::Movable => {
                    stack[i] = var;
                    i += 1;
                }
                StackLocation::Immovable(v) => {
                    let &slot = slots.get(&v).unwrap();
                    stack[slot as usize] = var;
                }
            }
        }

        let output = ShuffleState {
            reg: registers,
            stack: [stack[0], stack[1], stack[2], 0],
        };
        let output = output.canonicalize(input.num_movable);

        Solution {
            solver: self,
            input,
            output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_case {
        (
            solver: $solver:expr,
            in: [$($in_reg:expr),*$(,)?],
            out: [$($out_reg:expr),*$(,)?],
            anchored: [$($anchored:expr),*$(,)?],
            cost: $cost:expr,
            instructions: [$($instr:expr),*$(,)?]$(,)?
        ) => {{
            let anchored = [$($anchored),*];
            let initial = State { register_info: [$($in_reg),*] };
            let finl = State { register_info: [$($out_reg),*] };
            let soln = $solver.solve_inner(&anchored[..], &initial, &finl);
            assert_eq!(soln.cost(), $cost);
            let mut instrs: Vec<_> = soln.instructions().collect();
            instrs.reverse();
            assert_eq!(instrs, &[$($instr),*]);
        }}
    }

    #[test]
    fn test_shuffle() {
        use ShuffleInstr::*;
        use StackState::*;

        let solver = ShuffleSolver::new();

        let [v1, v2, v3, v4, v5, v6] = [0, 1, 2, 3, 4, 5].map(|x| Variable(0u16.into(), x));

        test_case! {
            solver: solver,
            in: [
                Some((v1, Floating)),
                Some((v2, Floating)),
                Some((v3, Floating)),
            ],
            out: [
                Some((v3, Floating)),
                Some((v1, Floating)),
                Some((v2, Floating)),
            ],
            anchored: [],
            cost: 24,
            instructions: [],
        }

        test_case! {
            solver: solver,
            in: [
                Some((v1, Floating)),
                Some((v2, Floating)),
                Some((v3, Floating)),
            ],
            out: [
                Some((v4, Floating)),
                Some((v5, Floating)),
                Some((v6, Floating)),
            ],
            anchored: [],
            cost: 40,
            instructions: [
                Swap(0, 1), Store(2, 3), Load(1, 2), Store(1, 1), Load(3, 1), Store(2, 3),
                Load(2, 2), Store(1, 2), Load(0, 1), Store(0, 0), Load(3, 0), DeleteStack(3),
            ],
        }

        test_case! {
            solver: solver,
            in: [
                Some((v1, Floating)),
                Some((v2, Floating)),
                Some((v3, Floating)),
            ],
            out: [
                Some((v4, Leased(v1))),
                Some((v5, Leased(v2))),
                Some((v6, Floating)),
            ],
            anchored: [v4, v5],
            cost: 42,
            instructions: [
                Store(1, 3), Load(0, 1), Store(2, 0), Load(3, 2), Store(1, 3), Load(2, 1),
                Store(2, 2), Copy(0, 2), Load(1, 0), Store(2, 1), Load(3, 2), DeleteStack(3),
            ],
        }

        test_case! {
            solver: solver,
            in: [
                Some((v1, Copied)),
                Some((v2, Floating)),
                Some((v3, Floating)),
            ],
            out: [
                Some((v4, Floating)),
                Some((v5, Floating)),
                Some((v1, Copied)),
            ],
            anchored: [],
            cost: 20,
            instructions: [
                Load(1, 0), Store(1, 1), Load(2, 1), Store(2, 2), Load(0, 2),
            ],
        }

        test_case! {
            solver: solver,
            in: [
                Some((v1, Floating)),
                Some((v2, Floating)),
                Some((v3, Floating)),
            ],
            out: [
                Some((v4, Floating)),
                Some((v5, Floating)),
                Some((v1, Floating)),
            ],
            anchored: [],
            cost: 24,
            instructions: [
                Store(0, 2), Load(0, 0), Store(1, 0), Load(1, 1), Store(2, 1), Load(2, 2), DeleteStack(2),
            ],
        }
    }
}

//! Optimal register shuffling solver and codegen for lookup tables.
// TODO: Handle 8-bit values. Keep in mind the accumulator is split into A and
// B registers.

use std::collections::BinaryHeap;
use std::{cmp::Reverse, io::Write};

use fnv::FnvHashMap as HashMap;
use serde_derive::{Deserialize, Serialize};

use crate::const_vec::{ConstMap, ConstVec};

#[derive(
    Clone, Copy, Debug, Default, Hash, Eq, Ord, PartialEq, PartialOrd, Deserialize, Serialize,
)]
pub struct State {
    pub reg: [u8; 3],
    pub stack: [u8; 4],
}

impl State {
    pub fn canonicalize(mut self, num_movable: u8) -> Self {
        if num_movable > 1 {
            if self.stack[0] > self.stack[1] {
                self.stack.swap(0, 1);
            }
        }
        if num_movable > 2 {
            if self.stack[0] > self.stack[2] {
                self.stack.swap(0, 2);
            }
            if self.stack[1] > self.stack[2] {
                self.stack.swap(1, 2);
            }
        }
        self
    }
}

#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
pub struct Input {
    pub reg: [u8; 3],
    pub stack: [u8; 3],
    /// Number of stack variables which may be relabeled.
    pub num_movable: u8,
    /// Number of stack variables which may not be relabeled.
    pub num_immovable: u8,
}

impl Input {
    fn state(&self) -> State {
        let [a, b, c] = self.stack;
        State {
            reg: self.reg,
            stack: [a, b, c, 0],
        }
    }

    fn stack_len(&self) -> usize {
        (self.num_immovable + self.num_movable) as usize
    }

    /// Returns the number of live variables present in the input.
    pub fn variables(&self) -> usize {
        let mut n = 0;
        for &v in self.reg.iter() {
            if v != 0 {
                n += 1;
            }
        }
        for &v in self.stack.iter() {
            if v != 0 {
                n += 1;
            }
        }
        n
    }
}

/// Instruction to use in code gen.
#[derive(Clone, Copy, Debug, Hash, Eq, Ord, PartialEq, PartialOrd, Deserialize, Serialize)]
pub enum ShuffleInstr {
    /// Load to reg from stack
    Load(u8, u8),
    /// Store reg to stack
    Store(u8, u8),
    /// Copy reg to reg
    Copy(u8, u8),
    /// Relabel two movable stack slots
    Swap(u8, u8),
    /// Replace register with 0
    DeleteReg(u8),
    /// Replace stack slot with 0
    DeleteStack(u8),
}

impl ShuffleInstr {
    pub const fn to_bytes(self) -> [u8; 3] {
        match self {
            ShuffleInstr::Load(src, dst) => [0, src, dst],
            ShuffleInstr::Store(src, dst) => [1, src, dst],
            ShuffleInstr::Copy(src, dst) => [2, src, dst],
            ShuffleInstr::Swap(src, dst) => [3, src, dst],
            ShuffleInstr::DeleteReg(src) => [4, src, 0],
            ShuffleInstr::DeleteStack(src) => [5, src, 0],
        }
    }

    pub const fn from_bytes(bytes: [u8; 3]) -> Option<Self> {
        match bytes[0] {
            0 => Some(ShuffleInstr::Load(bytes[1], bytes[2])),
            1 => Some(ShuffleInstr::Store(bytes[1], bytes[2])),
            2 => Some(ShuffleInstr::Copy(bytes[1], bytes[2])),
            3 => Some(ShuffleInstr::Swap(bytes[1], bytes[2])),
            4 => Some(ShuffleInstr::DeleteReg(bytes[1])),
            5 => Some(ShuffleInstr::DeleteStack(bytes[1])),
            _ => None,
        }
    }
}

// We use depth as a tie-breaker to eliminate unnecessary zero-cost moves
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
struct Entry {
    cost: u16,
    depth: u16,
    prev: Option<(State, ShuffleInstr)>,
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            cost: u16::MAX,
            depth: u16::MAX,
            prev: None,
        }
    }
}

const fn coeffs(k: usize) -> [usize; 6] {
    [
        k,
        k * k,
        k * k * k,
        k * k * k * k,
        k * k * k * k * k,
        k * k * k * k * k * k,
    ]
}

const COEFFS: [[usize; 6]; 6] = [
    coeffs(2),
    coeffs(3),
    coeffs(4),
    coeffs(5),
    coeffs(6),
    coeffs(7),
];

#[derive(Copy, Clone, Debug)]
pub struct EntryIndexer {
    num_vars: u8,
    num_slots: u8,
}

impl EntryIndexer {
    pub fn new(input: &Input) -> Self {
        let mut vars: ConstMap<u8, (), 7> = input
            .reg
            .iter()
            .chain(input.stack.iter())
            .map(|&x| (x, ()))
            .collect();
        vars.insert(0, ());
        let num_vars = vars.len() as u8;
        let num_slots = input.num_immovable + input.num_movable + 1;
        Self {
            num_vars,
            num_slots,
        }
    }

    pub fn len(&self) -> usize {
        (self.num_vars as usize).pow(self.num_slots as u32 + 3)
    }

    pub fn index(&self, state: &State) -> usize {
        let [k, k2, k3, k4, k5, k6] = COEFFS[self.num_vars as usize - 2];
        let [r0, r1, r2] = state.reg.map(|x| x as usize);
        let [s0, s1, s2, s3] = state.stack.map(|x| x as usize);
        match self.num_slots {
            1 => k3 * r0 + k2 * r1 + k * r2 + s0,
            2 => k4 * r0 + k3 * r1 + k2 * r2 + k * s0 + s1,
            3 => k5 * r0 + k4 * r1 + k3 * r2 + k2 * s0 + k * s1 + s2,
            4 => k6 * r0 + k5 * r1 + k4 * r2 + k3 * s0 + k2 * s1 + k * s2 + s3,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
struct EntryMap<T> {
    indexer: EntryIndexer,
    entries: Vec<T>,
}

impl<T: Clone + Default> EntryMap<T> {
    fn new(input: &Input) -> Self {
        let indexer = EntryIndexer::new(input);
        let v = vec![T::default(); indexer.len()];
        Self {
            indexer,
            entries: v,
        }
    }
}

impl<T> EntryMap<T> {
    fn get(&self, state: &State) -> &T {
        let idx = self.indexer.index(state);
        &self.entries[idx]
    }

    fn get_mut(&mut self, state: &State) -> &mut T {
        let idx = self.indexer.index(state);
        &mut self.entries[idx]
    }
}

fn insert_entry(
    entries: &mut EntryMap<Entry>,
    work: &mut BinaryHeap<Reverse<(u16, u16, State)>>,
    state: State,
    prev: State,
    instr: ShuffleInstr,
    cost: u16,
    depth: u16,
) {
    let entry = entries.get_mut(&state);
    if cost >= entry.cost {
        return;
    }
    if depth >= entry.depth {
        return;
    }
    *entry = Entry {
        cost,
        depth,
        prev: Some((prev, instr)),
    };
    work.push(Reverse((cost, depth, state)));
}

impl From<EntryMap<Entry>> for EntryMap<RegisterShuffleEntry> {
    fn from(value: EntryMap<Entry>) -> Self {
        EntryMap {
            indexer: value.indexer,
            entries: value
                .entries
                .into_iter()
                .map(|entry| entry.into())
                .collect(),
        }
    }
}

fn search(input: &Input) -> EntryMap<Entry> {
    let start = std::time::Instant::now();

    // Add 1 for the temp variable slot
    let stack_len = input.stack_len() + 1;
    let init = input.state();
    assert!(stack_len <= init.stack.len());
    let mut entries: EntryMap<Entry> = EntryMap::new(input);
    *entries.get_mut(&init) = Entry {
        cost: 0,
        depth: 0,
        prev: None,
    };
    let mut work: BinaryHeap<Reverse<(u16, u16, State)>> = BinaryHeap::new();
    work.push(Reverse((0, 0, init)));

    print!("input: {:?}, states: {}", input, entries.entries.len());
    std::io::stdout().flush().unwrap();

    while let Some(Reverse((c, d, state))) = work.pop() {
        let &Entry { cost, depth, .. } = entries.get(&state);
        if cost < c {
            continue;
        }
        if depth < d {
            continue;
        }
        let reg = &state.reg;
        let stack = &state.stack;

        for i in 0..reg.len() {
            if reg[i] == 0 {
                continue;
            }
            // Copy from register to register
            for j in 0..reg.len() {
                if reg[i] == reg[j] {
                    continue;
                }
                let mut new = state;
                new.reg[j] = reg[i];
                let new_cost = cost + 2;
                let instr = ShuffleInstr::Copy(i as u8, j as u8);
                insert_entry(&mut entries, &mut work, new, state, instr, new_cost, d + 1);
            }
            // Copy from register to stack
            for j in 0..stack_len {
                if reg[i] == stack[j] {
                    continue;
                }
                let mut new = state;
                new.stack[j] = reg[i];
                let new_cost = cost + 4;
                let instr = ShuffleInstr::Store(i as u8, j as u8);
                insert_entry(&mut entries, &mut work, new, state, instr, new_cost, d + 1);
            }
            // Delete from register
            let mut new = state;
            new.reg[i] = 0;
            let instr = ShuffleInstr::DeleteReg(i as u8);
            insert_entry(&mut entries, &mut work, new, state, instr, cost, d + 1);
        }

        for i in 0..stack_len {
            if stack[i] == 0 {
                continue;
            }
            // Copy from stack to register
            for j in 0..reg.len() {
                if stack[i] == reg[j] {
                    continue;
                }
                let mut new = state;
                new.reg[j] = stack[i];
                let new_cost = cost + 4;
                let instr = ShuffleInstr::Load(i as u8, j as u8);
                insert_entry(&mut entries, &mut work, new, state, instr, new_cost, d + 1);
            }
            // Delete from stack
            let mut new = state;
            new.stack[i] = 0;
            let instr = ShuffleInstr::DeleteStack(i as u8);
            insert_entry(&mut entries, &mut work, new, state, instr, cost, d + 1);
        }

        // Relabel movable slots
        for i in 0..input.num_movable as usize {
            for j in i + 1..input.num_movable as usize {
                let mut new = state;
                new.stack.swap(i, j);
                let instr = ShuffleInstr::Swap(i as u8, j as u8);
                insert_entry(&mut entries, &mut work, new, state, instr, cost, d + 1);
            }
        }
    }

    println!(", time elapsed: {:?}", start.elapsed());

    entries
}

/// Publicly consumable solution data
#[derive(Clone, Copy, Debug)]
pub struct RegisterShuffleEntry {
    pub prev: Option<(State, ShuffleInstr)>,
    pub cost: u16,
}

impl From<Entry> for RegisterShuffleEntry {
    fn from(value: Entry) -> Self {
        Self {
            prev: value.prev,
            cost: value.cost,
        }
    }
}

impl RegisterShuffleEntry {
    fn from_bytes(bytes: [u8; 12]) -> Option<Self> {
        let instr = ShuffleInstr::from_bytes([bytes[7], bytes[8], bytes[9]])?;
        let state = State {
            reg: [bytes[0], bytes[1], bytes[2]],
            stack: [bytes[3], bytes[4], bytes[5], bytes[6]],
        };
        let cost = u16::from_le_bytes([bytes[10], bytes[11]]);
        Some(Self {
            prev: Some((state, instr)),
            cost,
        })
    }

    fn to_bytes(self) -> [u8; 12] {
        let mut bytes = [0u8; 12];
        if let Some((state, instr)) = self.prev {
            bytes[0..3].copy_from_slice(&state.reg);
            bytes[3..7].copy_from_slice(&state.stack);
            let instr_bytes = instr.to_bytes();
            bytes[7..10].copy_from_slice(&instr_bytes);
        } else {
            // Mark as initial state
            bytes[7] = 255;
        }
        let cost_bytes = self.cost.to_le_bytes();
        bytes[10..12].copy_from_slice(&cost_bytes);
        bytes
    }
}

fn copy_from_iter<T>(slice: &mut [T], iter: impl Iterator<Item = T>) {
    for (i, v) in iter.enumerate() {
        slice[i] = v;
    }
}

fn visit_rec(
    output: &mut Vec<Input>,
    reg: [u8; 3],
    num_reg: u8,
    movable: ConstVec<u8, 3>,
    immovable: ConstVec<u8, 3>,
    i: u8,
) {
    if i <= num_reg {
        visit_rec(
            output,
            reg,
            num_reg,
            movable.clone(),
            immovable.clone(),
            i + 1,
        );
        let mut movable_next = movable.clone();
        movable_next.push(i);
        visit_rec(output, reg, num_reg, movable_next, immovable, i + 1);
        let mut immovable_next = immovable.clone();
        immovable_next.push(i);
        visit_rec(output, reg, num_reg, movable, immovable_next, i + 1);
    } else {
        let max_vars = 3 - movable.len() - immovable.len();
        for num_vars in 0..=max_vars {
            for num_movable in 0..=num_vars {
                let mut mv = movable.clone();
                let mut im = immovable.clone();
                for j in 0..num_movable {
                    mv.push(num_reg + j as u8 + 1);
                }
                for j in num_movable..num_vars {
                    im.push(num_reg + j as u8 + 1);
                }
                if mv.len() == 1 {
                    // There's no difference between having one movable slot and zero movable slots
                    continue;
                }
                mv.sort();
                im.sort();
                let mut stack = [0u8; 3];
                copy_from_iter(&mut stack, mv.iter().chain(im.iter()).cloned());
                output.push(Input {
                    reg,
                    stack,
                    num_movable: mv.len() as u8,
                    num_immovable: im.len() as u8,
                });
            }
        }
    }
}

pub(crate) fn build_shuffle_inputs() -> Vec<Input> {
    let mut output = Vec::new();
    let configs: &[(u8, [u8; 3])] = &[
        (1, [1, 0, 0]),
        (1, [0, 1, 0]),
        (1, [0, 0, 1]),
        (2, [1, 2, 0]),
        (2, [1, 0, 2]),
        (2, [0, 1, 2]),
        (3, [1, 2, 3]),
    ];
    for &(num_vars, reg) in configs.iter() {
        visit_rec(
            &mut output,
            reg,
            num_vars,
            Default::default(),
            Default::default(),
            1,
        );
    }
    output
}

pub fn solve_shuffles() -> HashMap<Input, Vec<RegisterShuffleEntry>> {
    let mut lut: HashMap<Input, Vec<RegisterShuffleEntry>> = Default::default();
    let inputs = build_shuffle_inputs();

    for input in inputs {
        let entries = search(&input);
        let entries = entries.entries.into_iter().map(From::from).collect();
        lut.insert(input, entries);
    }

    println!("done searching.");

    lut
}

fn entries_to_bytes(entries: &[RegisterShuffleEntry]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(entries.len() * 12);
    for entry in entries {
        bytes.extend_from_slice(&entry.to_bytes());
    }
    bytes
}

pub fn serialize_lut(lut: HashMap<Input, Vec<RegisterShuffleEntry>>, w: impl Write) {
    let lut: HashMap<_, _> = lut
        .into_iter()
        .map(|(input, entries)| (input, entries_to_bytes(&entries)))
        .collect();
    postcard::to_io(&lut, w).unwrap();
}

#![allow(dead_code)]

use std::collections::HashSet;

use crate::common::reg_alloc::AllowedRegisters;

use super::ssa_model::*;

/*
/// Computes uses of all variables in `function`, but does not compute
/// liveness.
fn compute_uses(function: &mut Function) {
    for b in 0..function.blocks.len() {
        for i in 0..function.blocks[b].instructions.len() {
            for o in 0..function.blocks[b].instructions[i].operands().len() {
                let op = function.blocks[b].instructions[i].operands()[o];
                let Some(var) = op.variable() else { continue };
                #[rustfmt::skip]
                function
                    .blocks[usize::from(var.block())]
                    .instructions[usize::from(var.instruction())]
                    .uses
                    .entry(Label(b as u16))
                    .or_default()
                    .push(i as u16);
            }
        }
    }
}
*/

/// Computes the set of live-in variables for each block
fn compute_liveness(function: &mut Function) {
    let blocks = &mut function.blocks;

    // Initialize live-ins
    for (b, block) in blocks.iter_mut().enumerate() {
        let live_in = &mut block.live_in;
        for (i, instr) in block.instructions.iter().enumerate().rev() {
            let var = Variable(Label::from(b), i as _);
            live_in.remove(&var);
            live_in.extend(instr.operands().iter().filter_map(|op| op.variable()));
        }
    }

    // Propagate live-ins backwards towards predecessors
    let mut work: HashSet<Label> = (0..blocks.len()).map(|b| Label::from(b)).collect();
    while !work.is_empty() {
        for b in std::mem::take(&mut work).into_iter() {
            let b = usize::from(b);
            let orig_size = blocks[b].live_in.len();

            let successors = blocks[b].successors().to_owned();
            for succ in successors {
                let succ = usize::from(succ);
                let vars = blocks[succ].live_in.clone();
                for var in vars.into_iter() {
                    if usize::from(var.block()) != b {
                        blocks[b].live_in.insert(var);
                    }
                }
            }

            if orig_size != blocks[b].live_in.len() {
                work.extend(blocks[b].predecessors());
            }
        }
    }
}

fn is_killed(function: &Function, block: Label, var: Variable) -> bool {
    !function.blocks[usize::from(block)]
        .successors()
        .iter()
        .any(|&b| function.blocks[usize::from(b)].live_in.contains(&var))
}

#[derive(Debug)]
pub struct LiveInterval {
    pub var: Variable,
    /// First instruction contained in the interval
    pub start: u16,
    /// One past last instruction contained in the interval
    pub end: u16,
    pub registers: AllowedRegisters,
}

/*
fn compute_live_intervals(function: &mut Function) {
    for b in 0..function.blocks.len() {
        let mut intervals: Vec<LiveInterval> = Vec::new();
        let mut live: HashMap<Variable, u16> =
            function.blocks[b].live_in.iter().map(|&v| (v, 0)).collect();

        {
            let block = &mut function.blocks[b];
            for (i, instr) in block.instructions.iter().enumerate() {
                let i = i as u16;
                for var in instr.uses() {
                    let start = live.get_mut(&var).unwrap();
                    if *start < i {
                        intervals.push(LiveInterval {
                            var,
                            start: *start,
                            end: i,
                            registers: Default::default(),
                        });
                    }
                    *start = i + 1;
                }
                live.insert(Variable(b.into(), i), i + 1);
            }
        }

        for (var, start) in live {
            let is_killed = is_killed(function, b.into(), var);
            let block = &mut function.blocks[usize::from(b)];
            if !is_killed && start < block.instructions.len() as u16 {
                intervals.push(LiveInterval {
                    var,
                    start,
                    end: block.instructions.len() as u16,
                    registers: Default::default(),
                });
            }
        }

        function.blocks[usize::from(b)].live_intervals = intervals;
    }
}

/// Eliminates registers from live intervals which would interfere with
/// overlapping instructions.
fn compute_allowed_registers(function: &mut Function) {
    for block in function.blocks.iter_mut() {
        let mut live: BinaryHeap<Reverse<(u16, u16)>> = BinaryHeap::new();
        let mut j: u16 = 0;
        for (i, instr) in block.instructions.iter_mut().enumerate() {
            let i = i as u16;
            while let Some(li) = block.live_intervals.get(j as usize)
                && li.start == i
            {
                live.push(Reverse((li.end, j)));
                j += 1;
            }
            while let Some(&Reverse((end, _))) = live.peek()
                && end < i
            {
                live.pop();
            }
            let allowed = instr.free_registers();
            for &Reverse((_, idx)) in live.iter() {
                let li = &mut block.live_intervals[idx as usize];
                li.registers = li.registers.and(allowed);
            }
        }
    }
    assert!(
        function.blocks[0].live_in.is_empty(),
        "variable used before definition: {}",
        function.blocks[0].live_in.iter().next().unwrap()
    );
}
*/

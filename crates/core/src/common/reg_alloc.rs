#![allow(dead_code)]

use std::iter::FusedIterator;

use super::macro_instr::*;
use super::ssa_model::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllowedRegisters {
    // Implicitly: none is always an option
    values: [bool; 3],
}

impl Default for AllowedRegisters {
    fn default() -> Self {
        Self { values: [true; 3] }
    }
}

impl AllowedRegisters {
    pub fn len(&self) -> usize {
        1 + self.values.into_iter().map(|b| b as usize).sum::<usize>()
    }

    pub fn c(&self) -> bool {
        self.values[0]
    }

    pub fn x(&self) -> bool {
        self.values[1]
    }

    pub fn y(&self) -> bool {
        self.values[2]
    }

    pub fn set_c(&mut self, value: bool) {
        self.values[0] = value
    }

    pub fn set_x(&mut self, value: bool) {
        self.values[1] = value
    }

    pub fn set_y(&mut self, value: bool) {
        self.values[2] = value
    }

    fn get(&self, mut index: usize) -> Option<Register> {
        for (reg, b) in [
            (None, true),
            (Some(Register::C), self.c()),
            (Some(Register::X), self.x()),
            (Some(Register::Y), self.y()),
        ] {
            if b {
                if index == 0 {
                    return reg;
                } else {
                    index -= 1;
                }
            }
        }
        panic!("index out of bounds");
    }

    pub fn iter(&self) -> impl Iterator<Item = Option<Register>> + FusedIterator + '_ {
        (0..4).filter_map(|i| {
            if i == 0 {
                return Some(None);
            }
            if !self.values[i - 1] {
                return None;
            }
            Some(Some(Register::try_from(i).unwrap()))
        })
    }

    pub fn and(self, other: Self) -> Self {
        let mut result = Self::default();
        for i in 0..3 {
            result.values[i] = self.values[i] & other.values[i];
        }
        result
    }

    pub fn or(self, other: Self) -> Self {
        let mut result = Self::default();
        for i in 0..3 {
            result.values[i] = self.values[i] | other.values[i];
        }
        result
    }
}

fn expand_commutative(
    expansions: &mut Vec<MacroInstr>,
    op1: Operand,
    op2: Operand,
    opcode: MacroOp,
) {
    if op1.is_variable() {
        expansions.push(MacroInstr {
            opcode,
            copy_c: Some(op1),
            operand: Some(op2),
            ..Default::default()
        });
    }
    if op2.is_variable() {
        expansions.push(MacroInstr {
            opcode,
            copy_c: Some(op2),
            operand: Some(op1),
            ..Default::default()
        });
    }
}

// FIXME: if two macro instructions operate on the same registers, must pick
// which one is fastest and push only that one. E.g.: ADC versus INC.
fn instr_expansions(inst: &InstrOp) -> Vec<MacroInstr> {
    let mut expansions = Vec::new();
    match *inst {
        InstrOp::Not(op) => {
            expansions.push(MacroInstr {
                opcode: MacroOp::Simple(MachineOpCode::Eor),
                copy_c: Some(op),
                operand: Some(Operand::Const(Const(-1))),
                ..Default::default()
            });
        }
        InstrOp::And([op1, op2]) => {
            expand_commutative(
                &mut expansions,
                op1,
                op2,
                MacroOp::Simple(MachineOpCode::And),
            );
        }
        InstrOp::Or([op1, op2]) => {
            expand_commutative(
                &mut expansions,
                op1,
                op2,
                MacroOp::Simple(MachineOpCode::Ora),
            );
        }
        InstrOp::Xor([op1, op2]) => {
            expand_commutative(
                &mut expansions,
                op1,
                op2,
                MacroOp::Simple(MachineOpCode::Eor),
            );
        }
        InstrOp::Neg(op) => {
            expansions.push(MacroInstr {
                opcode: MacroOp::EorFfInc(Const(1)),
                copy_c: Some(op),
                ..Default::default()
            });
        }
        InstrOp::Add([mut op1, mut op2]) => {
            if op1.is_constant() {
                std::mem::swap(&mut op1, &mut op2);
            }

            if op2.constant() != Some(Const(1)) {
                expansions.push(MacroInstr {
                    opcode: MacroOp::ClcAdc,
                    copy_c: Some(op1),
                    operand: Some(op2),
                    ..Default::default()
                });
            }

            if op2.is_variable() {
                expansions.push(MacroInstr {
                    opcode: MacroOp::ClcAdc,
                    copy_c: Some(op2),
                    operand: Some(op1),
                    ..Default::default()
                });
            }

            if let Operand::Const(c) = op2 {
                // INC
                expansions.push(MacroInstr {
                    opcode: MacroOp::Inc(c),
                    copy_c: Some(op2),
                    ..Default::default()
                });
                // INX
                expansions.push(MacroInstr {
                    opcode: MacroOp::Inx(c),
                    copy_x: Some(op2),
                    ..Default::default()
                });
                // INY
                expansions.push(MacroInstr {
                    opcode: MacroOp::Iny(c),
                    copy_y: Some(op2),
                    ..Default::default()
                });
            }
        }
        InstrOp::Sub([op1, op2]) => {
            // SEC
            // SBC op2
            let sec_sbc = MacroInstr {
                opcode: MacroOp::SecSbc,
                copy_c: Some(op1),
                operand: Some(op2),
                ..Default::default()
            };

            // EOR $ff
            // INC
            if let Operand::Const(c) = op1 {
                expansions.push(MacroInstr {
                    opcode: MacroOp::EorFfInc(c),
                    copy_c: Some(op2),
                    ..Default::default()
                });
            }

            if let Operand::Const(c) = op2 {
                if c.0.abs() > 1 {
                    expansions.push(sec_sbc);
                }

                // INC
                expansions.push(MacroInstr {
                    opcode: MacroOp::Inc(-c),
                    copy_c: Some(op1),
                    ..Default::default()
                });
                // INX
                expansions.push(MacroInstr {
                    opcode: MacroOp::Inx(-c),
                    copy_x: Some(op1),
                    ..Default::default()
                });
                // INY
                expansions.push(MacroInstr {
                    opcode: MacroOp::Iny(-c),
                    copy_y: Some(op1),
                    ..Default::default()
                });
            } else {
                expansions.push(sec_sbc);
            }
        }
    }

    expansions
}

fn generate_instr_expansions(function: &mut Function) {
    for block in function.blocks.iter_mut() {
        for instr in block.instructions.iter_mut() {
            instr.expansions = instr_expansions(&instr.operation);
        }
    }
}

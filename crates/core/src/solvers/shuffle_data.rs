use rcc_codegen::shuffle::ShuffleInstr::*;
use rcc_codegen::shuffle::{Input, RegisterShuffleEntry, State};

const SHUFFLE_DATA: &'static [u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shuffle_states.bin"));

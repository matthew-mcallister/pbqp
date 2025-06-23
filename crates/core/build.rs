use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use rcc_codegen::shuffle::*;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let path = out_dir.join("shuffle_states.bin");

    if path.exists() {
        println!("found shuffle states: {:?}", path);
    }

    let file = File::create(&path).unwrap();
    let file = BufWriter::new(file);

    let lut = solve_shuffles();

    println!("writing to {:?}", path);
    serialize_lut(lut, file);
}

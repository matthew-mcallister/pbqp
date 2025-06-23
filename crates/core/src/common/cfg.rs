//! CFG-related algorithms.

use std::num::NonZeroU16;

use smallvec::SmallVec;

use super::ssa_model::*;

pub fn compute_predecessors(f: &mut Function) {
    for i in 0..f.blocks.len() {
        let succ: SmallVec<[Label; 4]> = f.blocks[i].successors().iter().cloned().collect();
        for j in succ {
            f.blocks[usize::from(j)]
                .predecessors
                .push(Label::from(i as u16));
        }
    }
}

pub fn compute_junctions(f: &mut Function) {
    struct DisjointSets {
        parent: Vec<u16>,
    }

    impl DisjointSets {
        fn new(size: u16) -> Self {
            Self {
                parent: (0..size).collect(),
            }
        }

        fn find(&mut self, x: u16) -> u16 {
            if self.parent[x as usize] != x {
                self.parent[x as usize] = self.find(self.parent[x as usize]);
            }
            self.parent[x as usize]
        }

        fn union(&mut self, x: u16, y: u16) {
            let x_root = self.find(x);
            let y_root = self.find(y);
            if x_root < y_root {
                self.parent[y_root as usize] = x_root;
            } else {
                self.parent[x_root as usize] = y_root;
            }
        }

        fn flatten(&mut self) {
            for i in 0..self.parent.len() {
                self.parent[i] = self.parent[self.parent[i] as usize];
            }
        }
    }

    let blocks = &mut f.blocks;
    let mut sets = DisjointSets::new(2 * blocks.len() as u16);
    for b in 0..blocks.len() as u16 {
        for &pred in blocks[b as usize].predecessors.iter() {
            sets.union(2 * b, 2 * u16::from(pred) + 1);
        }
    }
    sets.flatten();

    for b in 0..blocks.len() {
        blocks[b].junction = NonZeroU16::new(sets.parent[2 * b] + 1);
    }
}

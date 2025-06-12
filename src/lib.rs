use std::collections::{HashMap, HashSet};

use crate::math::Matrix;

mod arena;
mod math;

/// Input to the PBQP solver.
///
/// Saturating addition is used to compute cost so u32::MAX is effectively infinity
// TODO: Arena memory allocation
#[derive(Debug)]
struct PbqpProblem {
    node_cost: Vec<Vec<u32>>,
    // Access like edge_cost[&(u, v)] where u < v
    edge_cost: HashMap<(u32, u32), Matrix<u32>>,
    adjacency: Vec<Vec<u32>>,
}

impl PbqpProblem {
    fn edge_matrix(&self, u: u32, v: u32) -> Matrix<u32> {
        if v < u {
            self.edge_cost[&(v, u)].transpose()
        } else {
            self.edge_cost[&(u, v)].clone()
        }
    }

    /// Returns the minimum edge cost between u and v for a given color of `u`
    fn min_edge_cost(&self, u: u32, u_color: usize, v: u32) -> u32 {
        let mut min = u32::MAX;
        for v_color in 0..self.node_cost[v as usize].len() {
            min = std::cmp::min(min, self.edge_matrix(u, v)[(u_color as usize, v_color)]);
        }
        min
    }
}

/// Lexicographic breadth-first search.
// adjacency[u] = list of nodes adjacent to node u
fn lex_bfs(adjacency: &Vec<Vec<u32>>) -> Vec<u32> {
    let n = adjacency.len();
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }

    // Each node starts in the same partition (all unvisited)
    let mut partitions: Vec<Vec<u32>> = vec![(0..n as u32).collect()];

    while let Some(mut part) = partitions.pop() {
        // Always pick the last node in the last partition (arbitrary but deterministic)
        let u = part.pop().unwrap();
        if !part.is_empty() {
            partitions.push(part);
        }
        out.push(u);

        // Refine partitions: for each partition, split into (adjacent to u, not adjacent to u)
        let mut new_partitions = Vec::with_capacity(partitions.len() * 2);
        let neighbors: HashSet<u32> = adjacency[u as usize].iter().copied().collect();
        for part in partitions.drain(..) {
            let (yes, no): (Vec<_>, Vec<_>) = part.iter().partition(|v| neighbors.contains(v));
            if !yes.is_empty() {
                new_partitions.push(yes);
            }
            if !no.is_empty() {
                new_partitions.push(no);
            }
        }
        partitions = new_partitions;
    }

    out.reverse();
    out
}

// Returns (colors, cost)
//
// lex_order[i] is the ith node in lexicographic order
// lex[u] is the position of node u in lex_order
fn solve_greedy(problem: &PbqpProblem, lex_order: &[u32], lex: &[u32]) -> (Vec<u8>, u32) {
    let n = problem.node_cost.len();
    let mut colors = vec![0u8; n];
    let mut total_cost = 0u32;

    for (lex_u, &u) in lex_order.iter().enumerate() {
        let u = u as usize;
        let k = problem.node_cost[u].len();
        let mut min_cost = u32::MAX;
        let mut best_color = 0u8;

        for color_u in 0..k {
            let mut cost = problem.node_cost[u][cu];

            for &v in &problem.adjacency[u] {
                let v = v as usize;
                if lex[v] < lex_u as u32 {
                    let color_v = colors[v] as usize;
                    let matrix = problem.edge_matrix(u as u32, v as u32);
                    cost = cost.saturating_add(matrix[(color_u, color_v)]);
                }
            }

            if cost < min_cost {
                min_cost = cost;
                best_color = color_u as u8;
            }
        }

        colors[u] = best_color;
        total_cost = total_cost.saturating_add(min_cost);
    }

    (colors, total_cost)
}

#[derive(Debug)]
struct PbqpSolver {
    problem: PbqpProblem,
    lex: Vec<u32>,
    lex_order: Vec<u32>,
    greedy: Vec<u8>,
    upper_bound: u32,
}

impl PbqpSolver {
    fn new(problem: PbqpProblem) -> Self {
        Self {
            problem,
            lex: Default::default(),
            lex_order: Default::default(),
            greedy: Default::default(),
            upper_bound: 0,
        }
    }

    // Computes a lower bound on the cost of a node + its lexicographically
    // backwards edges
    fn node_lower_bound(&self, u: u32) -> u32 {
        let mut min = u32::MAX;
        for color_u in 0..self.problem.node_cost[u as usize].len() {
            let mut cost = self.problem.node_cost[u as usize][color_u];
            for &v in &self.problem.adjacency[u as usize] {
                if self.lex[v as usize] < self.lex[u as usize] {
                    cost = cost.saturating_add(self.problem.min_edge_cost(u, color_u, v));
                }
            }
            min = std::cmp::min(min, cost);
        }
        min
    }

    fn solve(&mut self) {
        self.lex_order = lex_bfs(&self.problem.adjacency);
        self.lex = vec![0; self.lex_order.len()];
        for (i, &u) in self.lex_order.iter().enumerate() {
            self.lex[u as usize] = i as u32;
        }
        (self.greedy, self.upper_bound) = solve_greedy(&self.problem, &self.lex_order, &self.lex);
    }
}

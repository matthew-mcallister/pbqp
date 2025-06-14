// TODO maybe: Arena memory allocation

use std::collections::HashSet;

use crate::arena::{Arena, Id};
use crate::math::Matrix;

/// Input to the PBQP solver.
///
/// Saturating addition is used to compute cost so u32::MAX is effectively infinity
// TODO maybe: Make this a trait
#[derive(Debug)]
pub struct Problem {
    node_cost: Vec<Vec<u32>>,
    // List of back edges (and their cost) for each node
    edge_cost: Vec<Vec<(u32, Matrix<u32>)>>,
    // Forward edges are computed lazily and used for R0, R1, and R2 reductions
    forward_edges: Vec<Vec<u32>>,
    // After updating node order, contains a mapping from original index to
    // updated index
    remapping: Vec<u32>,
}

impl Problem {
    pub fn len(&self) -> usize {
        self.node_cost.len()
    }

    pub fn new(node_cost: Vec<Vec<u32>>, edge_cost: Vec<Vec<(u32, Matrix<u32>)>>) -> Self {
        assert_eq!(
            node_cost.len(),
            edge_cost.len(),
            "node_cost and edge_cost must have the same length"
        );
        Self {
            node_cost,
            edge_cost,
            remapping: Vec::new(),
            forward_edges: Vec::new(),
        }
    }

    // Computes the neighbors of `u` from its forward and backwards edges
    fn neighbors(&self, u: u32) -> impl Iterator<Item = u32> + '_ {
        assert!(!self.forward_edges.is_empty());
        self.edge_cost[u as usize]
            .iter()
            .map(|&(v, _)| v)
            .chain(self.forward_edges[u as usize].iter().copied())
    }

    /// Given a permutation of the nodes, rewrites the problem's internal data
    /// structures based on the new order.
    // TODO maybe: Write a RemappedProblem type that does the index remapping
    // transparently without rebuilding data structures
    fn remap(&mut self, remapping: Vec<u32>) {
        assert_eq!(
            remapping.len(),
            self.len(),
            "expected permutation length {}, got {}",
            self.len(),
            remapping.len()
        );
        assert!(
            self.remapping.is_empty(),
            // XXX: RemappedProblem would support this easily
            "multiple remappings not supported"
        );

        let n = remapping.len();

        let node_cost = std::mem::take(&mut self.node_cost);
        let mut new_node_cost = vec![Vec::new(); n];
        for (i, v) in node_cost.into_iter().enumerate() {
            new_node_cost[remapping[i] as usize] = v;
        }

        let edge_cost = std::mem::take(&mut self.edge_cost);
        let mut new_edge_cost: Vec<Vec<(u32, Matrix<u32>)>> = vec![Default::default(); n];
        for (u, edges) in edge_cost.into_iter().enumerate() {
            for (v, matrix) in edges.into_iter() {
                let new_u = remapping[u as usize];
                let new_v = remapping[v as usize];
                let (s, t) = if new_u < new_v {
                    (new_u, new_v)
                } else {
                    (new_v, new_u)
                };
                new_edge_cost[t as usize].push((s, matrix));
            }
        }

        self.remapping = remapping;
        self.node_cost = new_node_cost;
        self.edge_cost = new_edge_cost;
    }

    fn compute_forward_edges(&mut self) {
        todo!()
    }

    /// Computes a lower bound on the cost of a node + its backwards edges
    fn node_lower_bound(&self, u: u32) -> u32 {
        let mut min = u32::MAX;
        for color_u in 0..self.node_cost[u as usize].len() {
            let mut total = self.node_cost[u as usize][color_u];
            for (_, edge) in &self.edge_cost[u as usize] {
                let cost = edge[color_u].iter().copied().min().unwrap();
                total = total.saturating_add(cost);
            }
            min = std::cmp::min(min, total);
        }
        min
    }

    /// Computes a lower bound on the problem solution
    fn lower_bound(&self) -> u32 {
        let mut lower_bound: u32 = 0;
        for u in 0..self.len() {
            lower_bound = lower_bound.saturating_add(self.node_lower_bound(u as u32));
        }
        lower_bound
    }
}

/// Lexicographic breadth-first search.
fn lex_bfs(problem: &Problem) -> Vec<u32> {
    let n = problem.len();
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
        let neighbors: HashSet<u32> = problem.neighbors(u).collect();
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

/// Returns (colors, upper_bound). The problem should be in lexicographic order
/// for optimal results.
// TODO: Implement R1 and R2 reductions here
fn solve_greedy(problem: &Problem) -> Vec<u8> {
    let n = problem.node_cost.len();
    let mut colors = Vec::new();

    for u in 0..n {
        let k = problem.node_cost[u].len();
        let mut min_cost = u32::MAX;
        let mut best_color = 0u8;

        for color_u in 0..k {
            let mut cost = problem.node_cost[u][color_u];

            for &(v, ref matrix) in problem.edge_cost[u].iter() {
                let color_v = colors[v as usize] as usize;
                cost = cost.saturating_add(matrix[color_u][color_v]);
            }

            if cost < min_cost {
                min_cost = cost;
                best_color = color_u as u8;
            }
        }

        colors.push(best_color);
    }

    colors
}

/// Given a solution and a global lower bound, computes the lower bound on
/// solving subproblems generated by this solution. I.e. lower_bound[k] is the
/// lower bound on branching after coloring node k.
///
/// This is only needed for initializing the solver after computing the greedy
/// solution; we keep track of the lower bound of each branch while searching.
fn compute_lower_bound(problem: &Problem, solution: &[u8], global_lower_bound: u32) -> Vec<u32> {
    // Recurrence relation behind all lower bound calculations:
    //  lower_bound[0] = global_lower_bound
    //  lower_bound[k + 1] = lower_bound[k] + (cost_of_k_and_back_edges - node_lower_bound_of_k)
    let n = solution.len();
    let mut lower_bound = Vec::new();
    let mut prev = global_lower_bound;

    for k in 0..n {
        // Cost of node k and its back edges
        let mut cost = problem.node_cost[k][solution[k] as usize];
        for &(v, ref matrix) in &problem.edge_cost[k] {
            let color_v = solution[v as usize] as usize;
            cost = cost.saturating_add(matrix[solution[k] as usize][color_v]);
        }
        let node_lb = problem.node_lower_bound(k as u32);
        let lb = prev.saturating_add(cost - node_lb);
        prev = lb;
        lower_bound.push(lb);
    }

    lower_bound
}

fn invert_permutation(permutation: Vec<u32>) -> Vec<u32> {
    let n = permutation.len();
    let mut inverse = vec![0; n];
    for (i, &p) in permutation.iter().enumerate() {
        inverse[p as usize] = i as u32;
    }
    inverse
}

/// Node within branch-and-bound search.
#[derive(Debug)]
struct Instance {
    // Lower bound on solving the subproblem
    lower_bound: u32,
    // Index of node that was visited (i.e. the parent index + 1)
    index: u32,
    // Color that was chosen
    color: u8,
    // Skiplist implementation:
    // - skiplist.len() is the largest power of two that divides index
    // - skiplist[k] points 2^k nodes prior
    skiplist: Vec<Id>,
}

#[derive(Debug, Default)]
struct SearchTree {
    instances: Arena<Instance>,
}

impl SearchTree {
    /// Finds the ancestor of an instance where the given node index was
    /// visited.
    fn find_ancestor(&self, instance_id: Id, target_index: u32) -> Id {
        let instances = &self.instances;

        let mut cur_id = instance_id;
        let mut cur = &instances[cur_id];
        let mut cur_index = cur.index;
        while cur_index > target_index {
            let diff = cur_index - target_index;
            let k = std::cmp::min(diff.ilog2() as usize, cur.skiplist.len() - 1);
            cur_id = cur.skiplist[k];
            cur = &instances[cur_id];
            cur_index = cur.index;
        }

        cur_id
    }

    /// Inserts a new instance into the tree beneath the given parent instance.
    /// Its skiplist will be generated automatically.
    fn insert(&mut self, mut instance: Instance, parent: Id) -> Id {
        let instances = &self.instances;
        let mut cur_id = parent;
        let mut cur = &instances[cur_id];
        debug_assert_eq!(cur.index, instance.index + 1);
        let k = instance.index.trailing_zeros() as usize + 1;
        for _ in 0..k {
            instance.skiplist.push(cur_id);
            cur_id = cur.skiplist[k];
            cur = &instances[cur_id];
        }
        self.instances.insert(instance)
    }
}

#[derive(Debug)]
pub struct Solver {
    problem: Problem,
    global_lower_bound: u32,
    best_solution: Vec<u8>,
    solution_lower_bound: Vec<u32>,
    search_tree: SearchTree,
}

impl Solver {
    pub fn new(mut problem: Problem) -> Self {
        problem.compute_forward_edges();
        problem.remap(invert_permutation(lex_bfs(&problem)));
        let global_lower_bound = problem.lower_bound();
        let best_solution = solve_greedy(&problem);
        let solution_lower_bound =
            compute_lower_bound(&problem, &best_solution, global_lower_bound);
        Self {
            problem,
            global_lower_bound,
            best_solution,
            solution_lower_bound,
            search_tree: Default::default(),
        }
    }

    pub fn solve(&mut self) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::math::Matrix;

    #[test]
    fn test_greedy_solution() {
        let node_cost = vec![vec![0, 1, 0], vec![0, 1, 2], vec![1, 2]];
        let edge_cost = vec![
            vec![],
            vec![(
                0,
                Matrix::from_array([[u32::MAX, 0, 0], [0, u32::MAX, 0], [0, 0, 0]]),
            )],
            vec![
                (0, Matrix::from_array([[0, 3, 1], [3, 0, 2]])),
                (1, Matrix::from_array([[u32::MAX, 0, 0], [0, 0, 0]])),
            ],
        ];

        let problem = Problem::new(node_cost, edge_cost);
        assert_eq!(solve_greedy(&problem), &[0, 1, 0]);
    }
}

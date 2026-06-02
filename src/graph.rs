//! Simple undirected graph representation.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

/// Undirected graph with optional edge weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    /// Number of vertices.
    pub n: usize,
    /// Adjacency: vertex → set of neighbors.
    pub adj: Vec<BTreeSet<usize>>,
    /// Edge list: (u, v) with u < v.
    pub edges: Vec<(usize, usize)>,
    /// Edge weights (optional). Maps (u,v) → weight, defaults to 1.0.
    pub weights: HashMap<(usize, usize), f64>,
}

impl Graph {
    /// Create an empty graph with `n` vertices.
    pub fn new(n: usize) -> Self {
        Graph {
            n,
            adj: vec![BTreeSet::new(); n],
            edges: Vec::new(),
            weights: HashMap::new(),
        }
    }

    /// Add an undirected edge (u, v).
    pub fn add_edge(&mut self, u: usize, v: usize) {
        assert!(u < self.n && v < self.n && u != v, "invalid edge");
        if !self.adj[u].contains(&v) {
            self.adj[u].insert(v);
            self.adj[v].insert(u);
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            self.edges.push((a, b));
            self.weights.insert((a, b), 1.0);
        }
    }

    /// Add a weighted undirected edge.
    pub fn add_weighted_edge(&mut self, u: usize, v: usize, w: f64) {
        assert!(u < self.n && v < self.n && u != v, "invalid edge");
        if !self.adj[u].contains(&v) {
            self.adj[u].insert(v);
            self.adj[v].insert(u);
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            self.edges.push((a, b));
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        self.weights.insert((a, b), w);
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Degree of vertex v.
    pub fn degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }

    /// Get edge weight (1.0 if unweighted).
    pub fn weight(&self, u: usize, v: usize) -> f64 {
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        *self.weights.get(&(a, b)).unwrap_or(&1.0)
    }

    /// Oriented edge endpoints. Returns (tail, head) for edge index `ei`.
    /// Convention: edge i → (edges[i].0, edges[i].1).
    pub fn edge_endpoints(&self, ei: usize) -> (usize, usize) {
        self.edges[ei]
    }

    /// Build a complete graph on n vertices.
    pub fn complete(n: usize) -> Self {
        let mut g = Self::new(n);
        for u in 0..n {
            for v in (u + 1)..n {
                g.add_edge(u, v);
            }
        }
        g
    }

    /// Build a path graph on n vertices.
    pub fn path(n: usize) -> Self {
        let mut g = Self::new(n);
        for i in 0..n.saturating_sub(1) {
            g.add_edge(i, i + 1);
        }
        g
    }

    /// Build a cycle graph on n vertices.
    pub fn cycle(n: usize) -> Self {
        let mut g = Self::path(n);
        if n > 2 {
            g.add_edge(n - 1, 0);
        }
        g
    }

    /// Check if the graph is connected via BFS.
    pub fn is_connected(&self) -> bool {
        if self.n == 0 {
            return true;
        }
        let mut visited = vec![false; self.n];
        let mut queue = vec![0usize];
        visited[0] = true;
        let mut count = 1;
        while let Some(v) = queue.pop() {
            for &u in &self.adj[v] {
                if !visited[u] {
                    visited[u] = true;
                    count += 1;
                    queue.push(u);
                }
            }
        }
        count == self.n
    }

    /// Incidence matrix B (n × m) with orientation.
    /// B[v, e] = +1 if v is tail, -1 if v is head, 0 otherwise.
    pub fn incidence_matrix(&self) -> nalgebra::DMatrix<f64> {
        let m = self.num_edges();
        let mut b = nalgebra::DMatrix::zeros(self.n, m);
        for (ei, &(u, v)) in self.edges.iter().enumerate() {
            b[(u, ei)] = 1.0;
            b[(v, ei)] = -1.0;
        }
        b
    }

    /// Standard graph Laplacian L = D - A.
    pub fn laplacian(&self) -> nalgebra::DMatrix<f64> {
        let b = self.incidence_matrix();
        &b * &b.transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_complete_graph_edges() {
        let g = Graph::complete(4);
        assert_eq!(g.num_edges(), 6);
        assert_eq!(g.degree(0), 3);
    }

    #[test]
    fn test_path_graph() {
        let g = Graph::path(4);
        assert_eq!(g.num_edges(), 3);
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.degree(1), 2);
        assert!(g.is_connected());
    }

    #[test]
    fn test_cycle_graph() {
        let g = Graph::cycle(5);
        assert_eq!(g.num_edges(), 5);
        for v in 0..5 {
            assert_eq!(g.degree(v), 2);
        }
    }

    #[test]
    fn test_graph_laplacian() {
        let g = Graph::path(3);
        let l = g.laplacian();
        // L = [[1,-1,0],[-1,2,-1],[0,-1,1]]
        assert_relative_eq!(l[(0, 0)], 1.0);
        assert_relative_eq!(l[(0, 1)], -1.0);
        assert_relative_eq!(l[(1, 1)], 2.0);
        assert_relative_eq!(l[(2, 2)], 1.0);
    }

    #[test]
    fn test_connectivity() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(2, 3);
        assert!(!g.is_connected());
        g.add_edge(1, 2);
        assert!(g.is_connected());
    }
}

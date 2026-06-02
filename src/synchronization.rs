//! Synchronization on graphs via sheaf Laplacian eigenvectors.
//!
//! A graph supports synchronization if the sheaf has trivial H⁰
//! (no nontrivial global sections). The spectral gap of the sheaf
//! Laplacian measures how "synchronizable" the system is.

use nalgebra::DVector;

use crate::graph::Graph;
use crate::sheaf::Sheaf;
use crate::laplacian::SheafLaplacian;

/// Compute the synchronization gap of a sheaf on a graph.
/// Returns the spectral gap, or None if the sheaf has no nonzero eigenvalue.
pub fn synchronization_gap(sheaf: &Sheaf, graph: &Graph) -> Option<f64> {
    let sl = SheafLaplacian::build(sheaf, graph);
    sl.spectral_gap()
}

/// Check if a sheaf supports synchronization:
/// H⁰(𝓕) = 0 means the only harmonic section is the zero section.
pub fn is_synchronizable(sheaf: &Sheaf, graph: &Graph) -> bool {
    let sl = SheafLaplacian::build(sheaf, graph);
    sl.kernel_dim() == 0
}

/// Multi-agent consensus as sheaf synchronization.
///
/// Given agents on a graph with local states, compute the consensus
/// via the harmonic projection (limit of sheaf diffusion).
pub struct ConsensusProblem {
    pub graph: Graph,
    pub sheaf: Sheaf,
}

impl ConsensusProblem {
    /// Create a standard consensus problem with a constant sheaf.
    pub fn standard(graph: Graph, d: usize) -> Self {
        let sheaf = Sheaf::constant(&graph, d);
        ConsensusProblem { graph, sheaf }
    }

    /// Solve consensus: compute the harmonic projection of initial states.
    /// Returns the limit of the diffusion process.
    pub fn solve(&self, initial_states: &DVector<f64>) -> DVector<f64> {
        let sl = SheafLaplacian::build(&self.sheaf, &self.graph);

        // Project onto kernel of L
        let harmonic_cols: Vec<_> = sl
            .eigenvectors
            .column_iter()
            .zip(sl.eigenvalues.iter())
            .filter(|(_, &ev)| ev.abs() < 1e-8)
            .map(|(col, _)| col.into_owned())
            .collect();

        if harmonic_cols.is_empty() {
            return DVector::zeros(initial_states.len());
        }

        // Compute the harmonic projection
        let n = initial_states.len();
        let mut result = DVector::zeros(n);
        for col in &harmonic_cols {
            let coeff = col.dot(initial_states);
            result += coeff * col;
        }
        result
    }

    /// Check if agents have reached consensus (all states approximately equal).
    pub fn is_consensus(states: &DVector<f64>, d: usize, tol: f64) -> bool {
        let n = states.len() / d;
        if n <= 1 {
            return true;
        }
        let first: Vec<f64> = (0..d).map(|k| states[k]).collect();
        for i in 1..n {
            for k in 0..d {
                if (states[i * d + k] - first[k]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Compute disagreement: ||x - x̄||² where x̄ is the mean section.
    pub fn disagreement(&self, states: &DVector<f64>) -> f64 {
        let sl = SheafLaplacian::build(&self.sheaf, &self.graph);
        let lx = &sl.l * states;
        states.dot(&lx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_constant_sheaf_has_gap() {
        let g = Graph::complete(4);
        let s = Sheaf::constant(&g, 1);
        let gap = synchronization_gap(&s, &g);
        assert!(gap.is_some());
        assert!(gap.unwrap() > 0.0);
    }

    #[test]
    fn test_constant_sheaf_not_synchronizable() {
        // Constant sheaf always has harmonic sections (constant assignments).
        let g = Graph::path(3);
        let s = Sheaf::constant(&g, 1);
        assert!(!is_synchronizable(&s, &g));
    }

    #[test]
    fn test_consensus_constant_sheaf() {
        let g = Graph::path(3);
        let cp = ConsensusProblem::standard(g, 1);
        let x0 = DVector::from_vec(vec![1.0, 5.0, 3.0]);
        let consensus = cp.solve(&x0);
        // Should converge to mean = 3.0
        for i in 0..3 {
            assert_relative_eq!(consensus[i], 3.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_consensus_2d() {
        let g = Graph::complete(3);
        let cp = ConsensusProblem::standard(g, 2);
        let x0 = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let consensus = cp.solve(&x0);
        // Mean: [(1+3+5)/3, (2+4+6)/3] = [3, 4]
        for i in 0..3 {
            assert_relative_eq!(consensus[i * 2], 3.0, epsilon = 1e-6);
            assert_relative_eq!(consensus[i * 2 + 1], 4.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_is_consensus_check() {
        let states = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        assert!(ConsensusProblem::is_consensus(&states, 1, 0.1));
        let states = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(!ConsensusProblem::is_consensus(&states, 1, 0.1));
    }

    #[test]
    fn test_disagreement_decreases_with_diffusion() {
        use crate::diffusion::SheafDiffusion;
        let g = Graph::complete(4);
        let cp = ConsensusProblem::standard(g.clone(), 1);
        let x0 = DVector::from_vec(vec![10.0, -5.0, 3.0, -8.0]);
        let diff = SheafDiffusion::new(&cp.sheaf, &g);
        let x1 = diff.evolve(&x0, 0.5);
        let x2 = diff.evolve(&x0, 2.0);
        let d0 = cp.disagreement(&x0);
        let d1 = cp.disagreement(&x1);
        let d2 = cp.disagreement(&x2);
        assert!(d1 < d0);
        assert!(d2 < d1);
    }

    #[test]
    fn test_synchronization_gap_complete_grows_with_n() {
        let g3 = Graph::complete(3);
        let g5 = Graph::complete(5);
        let s3 = Sheaf::constant(&g3, 1);
        let s5 = Sheaf::constant(&g5, 1);
        let gap3 = synchronization_gap(&s3, &g3).unwrap();
        let gap5 = synchronization_gap(&s5, &g5).unwrap();
        assert!(gap5 > gap3);
    }
}

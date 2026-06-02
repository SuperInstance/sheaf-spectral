//! Sheaf Laplacian: L = D^T D where D is the coboundary operator.

use nalgebra::DMatrix;

use crate::graph::Graph;
use crate::sheaf::Sheaf;

/// Sheaf Laplacian and its spectral properties.
pub struct SheafLaplacian {
    /// The Laplacian matrix L = D^T D.
    pub l: DMatrix<f64>,
    /// Eigenvalues (sorted ascending).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors (columns), corresponding to eigenvalues.
    pub eigenvectors: DMatrix<f64>,
}

impl SheafLaplacian {
    /// Build the sheaf Laplacian from a sheaf on a graph.
    pub fn build(sheaf: &Sheaf, graph: &Graph) -> Self {
        let d = sheaf.coboundary(graph);
        let l = &d.transpose() * &d;
        let (eigenvalues, eigenvectors) = Self::sym_eigen(&l);
        SheafLaplacian {
            l,
            eigenvalues,
            eigenvectors,
        }
    }

    /// Compute symmetric eigenvalue decomposition.
    fn sym_eigen(m: &DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
        let n = m.nrows();
        if n == 0 {
            return (vec![], DMatrix::zeros(0, 0));
        }
        let eig = m.clone().symmetric_eigen();
        let mut pairs: Vec<(f64, nalgebra::DVector<f64>)> = eig
            .eigenvalues
            .iter()
            .zip(eig.eigenvectors.column_iter())
            .map(|(&val, col)| (val, col.into_owned()))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let eigenvalues: Vec<f64> = pairs.iter().map(|(v, _)| *v).collect();
        let cols: Vec<_> = pairs.iter().map(|(_, v)| v.clone()).collect();
        let eigenvectors = if cols.is_empty() {
            DMatrix::zeros(0, 0)
        } else {
            DMatrix::from_columns(&cols)
        };
        (eigenvalues, eigenvectors)
    }

    /// Smallest eigenvalue (should be ≥ 0 for PSD matrices).
    pub fn smallest_eigenvalue(&self) -> f64 {
        self.eigenvalues.first().copied().unwrap_or(0.0)
    }

    /// Spectral gap: smallest nonzero eigenvalue.
    pub fn spectral_gap(&self) -> Option<f64> {
        for &ev in &self.eigenvalues {
            if ev > 1e-10 {
                return Some(ev);
            }
        }
        None
    }

    /// Dimension of the kernel (= multiplicity of eigenvalue 0).
    pub fn kernel_dim(&self) -> usize {
        self.eigenvalues
            .iter()
            .filter(|&&ev| ev.abs() < 1e-8)
            .count()
    }

    /// Trace of the Laplacian.
    pub fn trace(&self) -> f64 {
        self.eigenvalues.iter().sum()
    }

    /// Fiedler value: second smallest eigenvalue (synonym for spectral gap on connected graphs).
    pub fn fiedler_value(&self) -> Option<f64> {
        let nonzero: Vec<f64> = self
            .eigenvalues
            .iter()
            .filter(|&&ev| ev > 1e-10)
            .copied()
            .collect();
        nonzero.first().copied()
    }

    /// Get the matrix reference.
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.l
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_constant_sheaf_laplacian_is_graph_laplacian() {
        // For a constant sheaf with d=1, the sheaf Laplacian equals the graph Laplacian.
        let g = Graph::path(3);
        let s = Sheaf::constant(&g, 1);
        let sl = SheafLaplacian::build(&s, &g);
        let gl = g.laplacian();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(sl.l[(i, j)], gl[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_psd_property() {
        let g = Graph::cycle(5);
        let s = Sheaf::constant(&g, 2);
        let sl = SheafLaplacian::build(&s, &g);
        for &ev in &sl.eigenvalues {
            assert!(ev >= -1e-8, "eigenvalue {} is negative", ev);
        }
    }

    #[test]
    fn test_kernel_dim_constant_connected() {
        // Constant sheaf d=2 on connected graph: kernel dim = 2.
        let g = Graph::path(4);
        let s = Sheaf::constant(&g, 2);
        let sl = SheafLaplacian::build(&s, &g);
        assert_eq!(sl.kernel_dim(), 2);
    }

    #[test]
    fn test_spectral_gap_positive_connected() {
        let g = Graph::complete(5);
        let s = Sheaf::constant(&g, 1);
        let sl = SheafLaplacian::build(&s, &g);
        assert!(sl.spectral_gap().unwrap() > 1e-6);
    }

    #[test]
    fn test_trace() {
        let g = Graph::path(3);
        let s = Sheaf::constant(&g, 1);
        let sl = SheafLaplacian::build(&s, &g);
        // Trace of path(3) Laplacian = 1 + 2 + 1 = 4
        assert_relative_eq!(sl.trace(), 4.0, epsilon = 1e-8);
    }

    #[test]
    fn test_spectral_gap_complete_vs_path() {
        // Complete graph should have larger spectral gap than path.
        let g_comp = Graph::complete(5);
        let g_path = Graph::path(5);
        let s1 = Sheaf::constant(&g_comp, 1);
        let s2 = Sheaf::constant(&g_path, 1);
        let sl_comp = SheafLaplacian::build(&s1, &g_comp);
        let sl_path = SheafLaplacian::build(&s2, &g_path);
        assert!(sl_comp.spectral_gap().unwrap() > sl_path.spectral_gap().unwrap());
    }

    #[test]
    fn test_laplacian_symmetric() {
        let g = Graph::cycle(6);
        let s = Sheaf::constant(&g, 3);
        let sl = SheafLaplacian::build(&s, &g);
        for i in 0..sl.l.nrows() {
            for j in 0..sl.l.ncols() {
                assert_relative_eq!(sl.l[(i, j)], sl.l[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_dimension_2_sheaf() {
        let g = Graph::path(3);
        let s = Sheaf::constant(&g, 2);
        let sl = SheafLaplacian::build(&s, &g);
        // Total vertex dim = 6, so Laplacian is 6×6
        assert_eq!(sl.l.nrows(), 6);
        assert_eq!(sl.l.ncols(), 6);
        // Kernel should be 2-dimensional (constant sections)
        assert_eq!(sl.kernel_dim(), 2);
    }
}

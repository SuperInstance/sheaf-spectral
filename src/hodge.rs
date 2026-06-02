//! Hodge decomposition of sheaf cohomology.
//!
//! For the sheaf Laplacian L = D^T D, we have the orthogonal decomposition:
//!   C⁰(𝓕) = ker(L) ⊕ im(D^T)
//!
//! The space ker(L) is the space of harmonic sections, isomorphic to H⁰(𝓕).

use nalgebra::DVector;
use nalgebra::DMatrix;

use crate::graph::Graph;
use crate::sheaf::Sheaf;
use crate::laplacian::SheafLaplacian;

/// Result of Hodge decomposition.
pub struct HodgeDecomposition {
    /// Dimension of harmonic space (H⁰).
    pub harmonic_dim: usize,
    /// Dimension of image space.
    pub image_dim: usize,
    /// Orthonormal basis for harmonic space (columns).
    pub harmonic_basis: DMatrix<f64>,
    /// Orthonormal basis for image space (columns).
    pub image_basis: DMatrix<f64>,
    /// Eigenvalues of the Laplacian.
    pub eigenvalues: Vec<f64>,
}

impl HodgeDecomposition {
    /// Compute the Hodge decomposition from the sheaf Laplacian.
    pub fn compute(sheaf: &Sheaf, graph: &Graph) -> Self {
        let sl = SheafLaplacian::build(sheaf, graph);
        Self::from_laplacian(&sl)
    }

    /// Compute from an already-built Laplacian.
    pub fn from_laplacian(sl: &SheafLaplacian) -> Self {
        let n = sl.eigenvalues.len();
        let mut harmonic_cols = Vec::new();
        let mut image_cols = Vec::new();

        for (i, &ev) in sl.eigenvalues.iter().enumerate() {
            let col = sl.eigenvectors.column(i).into_owned();
            if ev.abs() < 1e-8 {
                harmonic_cols.push(col);
            } else {
                image_cols.push(col);
            }
        }

        let harmonic_dim = harmonic_cols.len();
        let image_dim = image_cols.len();
        let harmonic_basis = if harmonic_cols.is_empty() {
            DMatrix::zeros(n, 0)
        } else {
            DMatrix::from_columns(&harmonic_cols)
        };
        let image_basis = if image_cols.is_empty() {
            DMatrix::zeros(n, 0)
        } else {
            DMatrix::from_columns(&image_cols)
        };

        HodgeDecomposition {
            harmonic_dim,
            image_dim,
            harmonic_basis,
            image_basis,
            eigenvalues: sl.eigenvalues.clone(),
        }
    }

    /// Project a section onto the harmonic subspace.
    pub fn project_harmonic(&self, x: &DVector<f64>) -> DVector<f64> {
        if self.harmonic_dim == 0 {
            return DVector::zeros(x.len());
        }
        &self.harmonic_basis * (&self.harmonic_basis.transpose() * x)
    }

    /// Project a section onto the image subspace.
    pub fn project_image(&self, x: &DVector<f64>) -> DVector<f64> {
        if self.image_dim == 0 {
            return DVector::zeros(x.len());
        }
        &self.image_basis * (&self.image_basis.transpose() * x)
    }

    /// Betti number b₀ = dim H⁰ = harmonic_dim.
    pub fn betti_0(&self) -> usize {
        self.harmonic_dim
    }

    /// Check if a vector is harmonic (in ker(L)).
    pub fn is_harmonic(&self, x: &DVector<f64>) -> bool {
        let proj = self.project_image(x);
        proj.iter().all(|&v| v.abs() < 1e-6)
    }

    /// Decompose a section into harmonic + image components.
    pub fn decompose(&self, x: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        (self.project_harmonic(x), self.project_image(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hodge_betti_0_constant_sheaf() {
        let g = Graph::path(4);
        let s = Sheaf::constant(&g, 2);
        let hodge = HodgeDecomposition::compute(&s, &g);
        // Constant sheaf on connected graph: b₀ = d = 2
        assert_eq!(hodge.betti_0(), 2);
    }

    #[test]
    fn test_hodge_direct_sum() {
        let g = Graph::cycle(4);
        let s = Sheaf::constant(&g, 1);
        let hodge = HodgeDecomposition::compute(&s, &g);
        let n = s.total_vertex_dim();
        let x = DVector::from_fn(n, |i, _| (i as f64 + 1.0).sin());
        let (h, i) = hodge.decompose(&x);
        // h + i should reconstruct x
        let sum = &h + &i;
        for k in 0..n {
            assert_relative_eq!(sum[k], x[k], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_orthogonality() {
        let g = Graph::complete(4);
        let s = Sheaf::constant(&g, 1);
        let hodge = HodgeDecomposition::compute(&s, &g);
        let n = s.total_vertex_dim();
        let x = DVector::from_fn(n, |i, _| (i as f64 * 0.7).cos());
        let (h, i) = hodge.decompose(&x);
        // h · i should be ~0
        let dot = h.dot(&i);
        assert_relative_eq!(dot, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_is_harmonic() {
        let g = Graph::path(3);
        let s = Sheaf::constant(&g, 1);
        let hodge = HodgeDecomposition::compute(&s, &g);
        // Constant section [1, 1, 1] should be harmonic
        let x = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        assert!(hodge.is_harmonic(&x));
    }

    #[test]
    fn test_harmonic_dim_matches_kernel() {
        let g = Graph::cycle(5);
        let s = Sheaf::constant(&g, 3);
        let sl = SheafLaplacian::build(&s, &g);
        let hodge = HodgeDecomposition::from_laplacian(&sl);
        assert_eq!(hodge.harmonic_dim, sl.kernel_dim());
    }

    #[test]
    fn test_dims_sum_to_total() {
        let g = Graph::path(5);
        let s = Sheaf::constant(&g, 2);
        let hodge = HodgeDecomposition::compute(&s, &g);
        assert_eq!(hodge.harmonic_dim + hodge.image_dim, s.total_vertex_dim());
    }
}

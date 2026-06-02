//! Sheaf diffusion: heat equation dx/dt = -Lx on sheaves.
//!
//! The solution x(t) = exp(-Lt) x(0) converges to the harmonic projection of x(0).

use nalgebra::{DMatrix, DVector};

use crate::graph::Graph;
use crate::sheaf::Sheaf;
use crate::laplacian::SheafLaplacian;

/// Sheaf diffusion simulator.
pub struct SheafDiffusion {
    pub laplacian: SheafLaplacian,
}

impl SheafDiffusion {
    /// Create a new diffusion from a sheaf on a graph.
    pub fn new(sheaf: &Sheaf, graph: &Graph) -> Self {
        let laplacian = SheafLaplacian::build(sheaf, graph);
        SheafDiffusion { laplacian }
    }

    /// Compute exp(-tL) using eigendecomposition.
    fn exp_minus_tl(&self, t: f64) -> DMatrix<f64> {
        let evals = &self.laplacian.eigenvalues;
        let evecs = &self.laplacian.eigenvectors;
        let n = evals.len();
        let mut diag_exp = DMatrix::zeros(n, n);
        for (i, &ev) in evals.iter().enumerate() {
            diag_exp[(i, i)] = (-ev * t).exp();
        }
        evecs * diag_exp * evecs.transpose()
    }

    /// Evolve section x by time t: x(t) = exp(-tL) x(0).
    pub fn evolve(&self, x0: &DVector<f64>, t: f64) -> DVector<f64> {
        let exp_mat = self.exp_minus_tl(t);
        &exp_mat * x0
    }

    /// Run discrete diffusion steps: x_{k+1} = (I - dt*L) x_k.
    /// Returns all intermediate states.
    pub fn step(&self, x0: &DVector<f64>, dt: f64, steps: usize) -> Vec<DVector<f64>> {
        let n = x0.len();
        let i_minus_dt_l = DMatrix::identity(n, n) - dt * &self.laplacian.l;
        let mut trajectory = Vec::with_capacity(steps + 1);
        let mut x = x0.clone();
        trajectory.push(x.clone());
        for _ in 0..steps {
            x = &i_minus_dt_l * &x;
            trajectory.push(x.clone());
        }
        trajectory
    }

    /// Check convergence to harmonic section after `steps` discrete steps.
    pub fn converge_to_harmonic(
        &self,
        x0: &DVector<f64>,
        dt: f64,
        steps: usize,
        tol: f64,
    ) -> (DVector<f64>, bool) {
        let traj = self.step(x0, dt, steps);
        let final_x = traj.last().unwrap().clone();

        // Check Lx ≈ 0
        let lx = &self.laplacian.l * &final_x;
        let residual = lx.iter().map(|v| v * v).sum::<f64>().sqrt();
        (final_x, residual < tol)
    }

    /// Compute the energy E(x) = x^T L x at each step.
    pub fn energy_trajectory(&self, x0: &DVector<f64>, dt: f64, steps: usize) -> Vec<f64> {
        let traj = self.step(x0, dt, steps);
        traj.iter()
            .map(|x| {
                let lx = &self.laplacian.l * x;
                x.dot(&lx)
            })
            .collect()
    }

    /// Spectral gap determines convergence rate.
    pub fn convergence_rate(&self) -> Option<f64> {
        self.laplacian.spectral_gap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_diffusion_converges_to_constant() {
        // Constant sheaf d=1 on path(3): diffusion of any vector → constant.
        let g = Graph::path(3);
        let s = Sheaf::constant(&g, 1);
        let diff = SheafDiffusion::new(&s, &g);

        let x0 = DVector::from_vec(vec![1.0, 3.0, -2.0]);
        let xt = diff.evolve(&x0, 100.0);
        // Should converge to mean = (1+3-2)/3 ≈ 0.667
        let mean = (1.0 + 3.0 + (-2.0)) / 3.0;
        for i in 0..3 {
            assert_relative_eq!(xt[i], mean, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_diffusion_energy_decreases() {
        let g = Graph::cycle(5);
        let s = Sheaf::constant(&g, 1);
        let diff = SheafDiffusion::new(&s, &g);
        let x0 = DVector::from_fn(5, |i, _| (i as f64 + 1.0).sin());
        let energies = diff.energy_trajectory(&x0, 0.05, 200);
        for i in 1..energies.len() {
            assert!(
                energies[i] <= energies[i - 1] + 1e-8,
                "Energy increased at step {}: {} > {}",
                i,
                energies[i],
                energies[i - 1]
            );
        }
    }

    #[test]
    fn test_discrete_convergence() {
        let g = Graph::complete(4);
        let s = Sheaf::constant(&g, 1);
        let diff = SheafDiffusion::new(&s, &g);
        let x0 = DVector::from_vec(vec![10.0, -5.0, 3.0, -8.0]);
        let (xf, converged) = diff.converge_to_harmonic(&x0, 0.01, 5000, 1e-3);
        assert!(converged, "Diffusion should converge");
        // All entries should be roughly equal
        let mean = x0.sum() / 4.0;
        for i in 0..4 {
            assert_relative_eq!(xf[i], mean, epsilon = 0.1);
        }
    }

    #[test]
    fn test_convergence_rate_exists() {
        let g = Graph::path(4);
        let s = Sheaf::constant(&g, 1);
        let diff = SheafDiffusion::new(&s, &g);
        assert!(diff.convergence_rate().is_some());
        assert!(diff.convergence_rate().unwrap() > 0.0);
    }

    #[test]
    fn test_exp_minus_tl_identity_at_zero() {
        let g = Graph::path(3);
        let s = Sheaf::constant(&g, 1);
        let diff = SheafDiffusion::new(&s, &g);
        let exp0 = diff.exp_minus_tl(0.0);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(exp0[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_preserves_harmonic() {
        let g = Graph::path(3);
        let s = Sheaf::constant(&g, 1);
        let diff = SheafDiffusion::new(&s, &g);
        // Harmonic section = constant [c, c, c]
        let x0 = DVector::from_vec(vec![2.0, 2.0, 2.0]);
        let xt = diff.evolve(&x0, 10.0);
        for i in 0..3 {
            assert_relative_eq!(xt[i], 2.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_multidim_diffusion() {
        let g = Graph::path(3);
        let s = Sheaf::constant(&g, 2);
        let diff = SheafDiffusion::new(&s, &g);
        let x0 = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let xt = diff.evolve(&x0, 50.0);
        // Each dimension should converge independently to its mean
        // dim 0: (1+3+5)/3 = 3, dim 1: (2+4+6)/3 = 4
        assert_relative_eq!(xt[0], 3.0, epsilon = 0.1);
        assert_relative_eq!(xt[2], 3.0, epsilon = 0.1);
        assert_relative_eq!(xt[4], 3.0, epsilon = 0.1);
        assert_relative_eq!(xt[1], 4.0, epsilon = 0.1);
    }
}

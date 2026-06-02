//! Connection Laplacian for vector bundles over graphs.
//!
//! For a vector bundle with transition maps (connections) on each edge,
//! the connection Laplacian is L_conn = Σ_{e=(u,v)} (I - ρ_{uv}⊗ρ_{vu}).

use nalgebra::DMatrix;

use crate::graph::Graph;

/// Connection Laplacian for a vector bundle on a graph.
pub struct ConnectionLaplacian {
    /// The connection Laplacian matrix.
    pub l: DMatrix<f64>,
    /// Eigenvalues (sorted).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors (columns).
    pub eigenvectors: DMatrix<f64>,
}

impl ConnectionLaplacian {
    /// Build from a graph with a fiber dimension d and connection maps.
    ///
    /// `connections` maps each edge index to a rotation/orthogonal matrix (d×d).
    /// For edge e = (u,v), the connection ρ_{uv} maps fiber_u → fiber_v.
    /// The reverse connection is ρ_{vu} = ρ_{uv}^T.
    ///
    /// L_conn = block matrix where block (u,u) = deg(u)*I,
    /// block (u,v) = -ρ_{vu} for each edge (u,v).
    pub fn build(graph: &Graph, d: usize, connections: Vec<DMatrix<f64>>) -> Self {
        let n = graph.n;
        let total = n * d;
        let mut l = DMatrix::zeros(total, total);

        // Build reverse connections
        let mut conn_rev: Vec<DMatrix<f64>> = vec![DMatrix::zeros(0, 0); connections.len()];
        for (ei, conn) in connections.iter().enumerate() {
            conn_rev[ei] = conn.transpose();
        }

        // Diagonal blocks: degree * I
        for v in 0..n {
            let deg = graph.degree(v) as f64;
            for k in 0..d {
                l[(v * d + k, v * d + k)] = deg;
            }
        }

        // Off-diagonal blocks
        for (ei, &(u, v)) in graph.edges.iter().enumerate() {
            let rho_vu = &conn_rev[ei];
            let rho_uv = &connections[ei];

            // Block (u, v) = -ρ_{vu}
            for i in 0..d {
                for j in 0..d {
                    l[(u * d + i, v * d + j)] = -rho_vu[(i, j)];
                }
            }
            // Block (v, u) = -ρ_{uv}
            for i in 0..d {
                for j in 0..d {
                    l[(v * d + i, u * d + j)] = -rho_uv[(i, j)];
                }
            }
        }

        let eig = l.clone().symmetric_eigen();
        let mut pairs: Vec<(f64, nalgebra::DVector<f64>)> = eig
            .eigenvalues
            .iter()
            .zip(eig.eigenvectors.column_iter())
            .map(|(&val, col)| (val, col.into_owned()))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let eigenvalues: Vec<f64> = pairs.iter().map(|(v, _)| *v).collect();
        let eigenvectors = if pairs.is_empty() {
            DMatrix::zeros(0, 0)
        } else {
            let cols: Vec<_> = pairs.iter().map(|(_, v)| v.clone()).collect();
            DMatrix::from_columns(&cols)
        };

        ConnectionLaplacian {
            l,
            eigenvalues,
            eigenvectors,
        }
    }

    /// Trivial connection (identity maps).
    pub fn trivial(graph: &Graph, d: usize) -> Self {
        let connections: Vec<DMatrix<f64>> = (0..graph.num_edges())
            .map(|_| DMatrix::identity(d, d))
            .collect();
        Self::build(graph, d, connections)
    }

    /// Spectral gap (smallest nonzero eigenvalue).
    pub fn spectral_gap(&self) -> Option<f64> {
        for &ev in &self.eigenvalues {
            if ev > 1e-8 {
                return Some(ev);
            }
        }
        None
    }

    /// Kernel dimension.
    pub fn kernel_dim(&self) -> usize {
        self.eigenvalues
            .iter()
            .filter(|&&ev| ev.abs() < 1e-8)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_trivial_connection_is_block_laplacian() {
        // Trivial connection with d=1 on path(3) should give graph Laplacian.
        let g = Graph::path(3);
        let cl = ConnectionLaplacian::trivial(&g, 1);
        let gl = g.laplacian();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(cl.l[(i, j)], gl[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_connection_laplacian_psd() {
        let g = Graph::cycle(4);
        let d = 2;
        let connections: Vec<DMatrix<f64>> = (0..g.num_edges())
            .map(|_| DMatrix::identity(d, d))
            .collect();
        let cl = ConnectionLaplacian::build(&g, d, connections);
        for &ev in &cl.eigenvalues {
            assert!(ev >= -1e-8, "negative eigenvalue: {}", ev);
        }
    }

    #[test]
    fn test_kernel_dim_trivial_connected() {
        let g = Graph::complete(3);
        let cl = ConnectionLaplacian::trivial(&g, 2);
        // Trivial connection on connected graph: kernel dim = d = 2
        assert_eq!(cl.kernel_dim(), 2);
    }

    #[test]
    fn test_symmetric() {
        let g = Graph::cycle(5);
        let cl = ConnectionLaplacian::trivial(&g, 3);
        let n = cl.l.nrows();
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(cl.l[(i, j)], cl.l[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_nontrivial_connection() {
        // Rotation connections on a cycle
        let g = Graph::cycle(4);
        let d = 2;
        let angle = std::f64::consts::PI / 4.0;
        let rot = DMatrix::from_row_slice(
            2,
            2,
            &[angle.cos(), -angle.sin(), angle.sin(), angle.cos()],
        );
        let connections: Vec<DMatrix<f64>> = (0..g.num_edges()).map(|_| rot.clone()).collect();
        let cl = ConnectionLaplacian::build(&g, d, connections);
        // Should still be PSD
        for &ev in &cl.eigenvalues {
            assert!(ev >= -1e-8);
        }
    }
}

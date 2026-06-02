//! Sheaf on a graph: assigns vector spaces to vertices and edges with restriction maps.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::graph::Graph;

/// A sheaf on a graph where each vertex and edge gets a vector space,
/// and restriction maps connect them.
///
/// We model a cellular sheaf: each vertex v gets stalk F(v) = R^{d_v},
/// and each edge e = (u,v) gets stalk F(e) = R^{d_e}, with
/// restriction maps F_u→e : F(u) → F(e) and F_v→e : F(v) → F(e).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sheaf {
    /// Dimension of the stalk at each vertex.
    pub vertex_dims: Vec<usize>,
    /// Dimension of the stalk at each edge.
    pub edge_dims: Vec<usize>,
    /// Restriction maps: restriction_maps[edge_index] = (map_from_tail, map_from_head).
    /// Each is a d_e × d_v matrix.
    pub restriction_maps: Vec<(DMatrix<f64>, DMatrix<f64>)>,
}

impl Sheaf {
    /// Create a constant sheaf: every vertex has dimension `d`, every edge has dimension `d`,
    /// and every restriction map is the identity.
    pub fn constant(graph: &Graph, d: usize) -> Self {
        let n = graph.n;
        let m = graph.num_edges();
        let vertex_dims = vec![d; n];
        let edge_dims = vec![d; m];
        let restriction_maps = graph
            .edges
            .iter()
            .map(|_| {
                let id = DMatrix::identity(d, d);
                (id.clone(), id)
            })
            .collect();
        Sheaf {
            vertex_dims,
            edge_dims,
            restriction_maps,
        }
    }

    /// Create a sheaf with custom restriction maps from a flat list of entries.
    /// Each restriction map is d_e × d_v, provided row-major.
    pub fn from_raw(
        vertex_dims: Vec<usize>,
        edge_dims: Vec<usize>,
        // For each edge: (tail_map_data, head_map_data), each row-major
        maps_data: Vec<(Vec<f64>, Vec<f64>)>,
    ) -> Self {
        let restriction_maps: Vec<(DMatrix<f64>, DMatrix<f64>)> = maps_data
            .into_iter()
            .enumerate()
            .map(|(ei, (tail_data, head_data))| {
                let de = edge_dims[ei];
                let tail_v = vertex_dims[0]; // simplified: assume uniform
                let head_v = vertex_dims[0];
                let tail_map = DMatrix::from_row_slice(de, tail_v, &tail_data);
                let head_map = DMatrix::from_row_slice(de, head_v, &head_data);
                (tail_map, head_map)
            })
            .collect();
        Sheaf {
            vertex_dims,
            edge_dims,
            restriction_maps,
        }
    }

    /// Total dimension of the vertex space (sum of all vertex stalk dims).
    pub fn total_vertex_dim(&self) -> usize {
        self.vertex_dims.iter().sum()
    }

    /// Total dimension of the edge space (sum of all edge stalk dims).
    pub fn total_edge_dim(&self) -> usize {
        self.edge_dims.iter().sum()
    }

    /// Build the coboundary operator D (total_edge_dim × total_vertex_dim).
    /// For edge e = (u, v): the block row has -F_{u→e} in column u and F_{v→e} in column v.
    /// (Orientation: tail gets negative sign.)
    pub fn coboundary(&self, graph: &Graph) -> DMatrix<f64> {
        let total_v = self.total_vertex_dim();
        let total_e = self.total_edge_dim();
        let mut d = DMatrix::zeros(total_e, total_v);

        // Precompute vertex column offsets.
        let v_offsets: Vec<usize> = self
            .vertex_dims
            .iter()
            .scan(0, |acc, &dim| {
                let off = *acc;
                *acc += dim;
                Some(off)
            })
            .collect();

        let mut e_row = 0;
        for (ei, &(u, v)) in graph.edges.iter().enumerate() {
            let de = self.edge_dims[ei];
            let du = self.vertex_dims[u];
            let dv = self.vertex_dims[v];
            let (ref tail_map, ref head_map) = self.restriction_maps[ei];

            // Block for tail (u): -tail_map
            for i in 0..de {
                for j in 0..du {
                    d[(e_row + i, v_offsets[u] + j)] = -tail_map[(i, j)];
                }
            }
            // Block for head (v): +head_map
            for i in 0..de {
                for j in 0..dv {
                    d[(e_row + i, v_offsets[v] + j)] = head_map[(i, j)];
                }
            }
            e_row += de;
        }
        d
    }
}

/// A vector sheaf: a sheaf with a specific section (assignment of vectors to stalks).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSheaf {
    pub sheaf: Sheaf,
    /// Section values for vertices: flattened, vertex v occupies indices [offset_v..offset_v+d_v).
    pub vertex_values: Vec<f64>,
}

impl VectorSheaf {
    /// Create a zero section.
    pub fn zero(sheaf: &Sheaf) -> Self {
        let total_v = sheaf.total_vertex_dim();
        VectorSheaf {
            sheaf: sheaf.clone(),
            vertex_values: vec![0.0; total_v],
        }
    }

    /// Get the section at vertex v.
    pub fn vertex_section(&self, v: usize) -> Vec<f64> {
        let offset: usize = self.sheaf.vertex_dims[..v].iter().sum();
        let d = self.sheaf.vertex_dims[v];
        self.vertex_values[offset..offset + d].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_sheaf_dims() {
        let g = crate::graph::Graph::path(3);
        let s = Sheaf::constant(&g, 2);
        assert_eq!(s.vertex_dims, vec![2, 2, 2]);
        assert_eq!(s.edge_dims, vec![2, 2]);
        assert_eq!(s.total_vertex_dim(), 6);
        assert_eq!(s.total_edge_dim(), 4);
    }

    #[test]
    fn test_coboundary_constant_sheaf() {
        // Path graph 0-1-2 with constant sheaf d=1.
        // D should be the oriented incidence matrix.
        let g = crate::graph::Graph::path(3);
        let s = Sheaf::constant(&g, 1);
        let d = s.coboundary(&g);
        // Edge (0,1): row 0 → [-1, +1, 0]
        // Edge (1,2): row 1 → [0, -1, +1]
        assert_relative_eq!(d[(0, 0)], -1.0);
        assert_relative_eq!(d[(0, 1)], 1.0);
        assert_relative_eq!(d[(1, 1)], -1.0);
        assert_relative_eq!(d[(1, 2)], 1.0);
    }

    use approx::assert_relative_eq;

    #[test]
    fn test_zero_section() {
        let g = crate::graph::Graph::complete(3);
        let s = Sheaf::constant(&g, 3);
        let v = VectorSheaf::zero(&s);
        assert_eq!(v.vertex_values.len(), 9);
        assert!(v.vertex_values.iter().all(|&x| x == 0.0));
    }
}

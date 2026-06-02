//! # sheaf-spectral
//!
//! Spectral sheaf theory: the intersection of sheaf cohomology and spectral graph theory.
//!
//! ## Core Concepts
//!
//! - **Sheaf Laplacian**: L = D^T D where D is the coboundary operator
//! - **Hodge Decomposition**: H⁰(𝓕) ≅ ker(L), exact/coexact splitting
//! - **Spectral Gaps**: Relation to synchronization on graphs
//! - **Sheaf Diffusion**: Heat equation dx/dt = -Lx converges to harmonic sections
//! - **Connection Laplacian**: For vector bundles over graphs
//! - **Sheaf Neural Networks**: Diffusion-based graph learning
//! - **Persistent Sheaf Cohomology**: Filtration → barcode

pub mod graph;
pub mod sheaf;
pub mod laplacian;
pub mod hodge;
pub mod diffusion;
pub mod connection;
pub mod neural;
pub mod synchronization;
pub mod persistent;

pub use graph::*;
pub use sheaf::*;
pub use laplacian::*;
pub use hodge::*;
pub use diffusion::*;
pub use connection::*;
pub use neural::*;
pub use synchronization::*;
pub use persistent::*;

pub mod prelude {
    pub use crate::graph::Graph;
    pub use crate::sheaf::{Sheaf, VectorSheaf};
    pub use crate::laplacian::SheafLaplacian;
    pub use crate::hodge::HodgeDecomposition;
    pub use crate::diffusion::SheafDiffusion;
    pub use crate::connection::ConnectionLaplacian;
    pub use crate::neural::SheafNNLayer;
    pub use crate::synchronization::synchronization_gap;
    pub use crate::persistent::PersistentSheaf;
}

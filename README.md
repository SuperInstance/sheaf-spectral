# sheaf-spectral

**Spectral sheaf theory in Rust. Where topology meets signal processing.**

`sheaf-spectral` sits at the intersection of **sheaf theory** and **spectral graph theory**. A cellular sheaf on a graph assigns vector spaces (stalks) to vertices and edges with linear restriction maps. The **sheaf Laplacian** L = DᵀD generalizes the graph Laplacian, and its spectral properties encode the global structure of the sheaf: harmonic sections, synchronization feasibility, diffusion behavior, and cohomology.

This crate provides the spectral toolkit for working with such sheaves — from computing eigenvalues to training neural networks that respect the sheaf structure.

## Key Idea

On a graph G with a sheaf F:

1. Each vertex v gets a vector space F(v) = R^{d_v}
2. Each edge e = (u,v) gets a vector space F(e) = R^{d_e}
3. Restriction maps F_{v→e}: F(v) → F(e) connect them

The **coboundary operator** D: C⁰(F) → C¹(F) encodes the signed restriction maps. The **sheaf Laplacian** L = DᵀD is the central object:

| Spectral property | Meaning |
|---|---|
| ker(L) = H⁰(F) | Space of harmonic (globally consistent) sections |
| Spectral gap λ₂ | Controls synchronization feasibility and diffusion speed |
| Eigenvalue multiplicity of 0 | Dimension of H⁰ (number of independent global sections) |
| Full spectrum | Determines diffusion, neural network behavior, and connectivity |

For a constant sheaf (identity restrictions, uniform dimension d), the sheaf Laplacian is just the Kronecker product L_graph ⊗ I_d, and H⁰ ≅ R^d (d copies of the constant functions).

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
sheaf-spectral = "0.1.0"
```

Requires **Rust 2021 edition**. Dependencies:

- [`nalgebra`](https://crates.io/crates/nalgebra) 0.33 (with `serde-serialize`) — linear algebra and eigendecomposition
- [`serde`](https://crates.io/crates/serde) 1 + [`serde_json`](https://crates.io/crates/serde_json) 1 — serialization
- [`approx`](https://crates.io/crates/approx) 0.5 (dev only) — approximate equality in tests

## Quick Start

### Build a graph and constant sheaf

```rust
use sheaf_spectral::prelude::*;

// Create graphs
let g = Graph::complete(5);
let path = Graph::path(4);
let cycle = Graph::cycle(6);

// Constant sheaf: every vertex gets R^d, every restriction is identity
let sheaf = Sheaf::constant(&g, 3); // 3D stalks
```

### Sheaf Laplacian and spectral analysis

```rust
let sl = SheafLaplacian::build(&sheaf, &g);

println!("Matrix size: {}×{}", sl.l.nrows(), sl.l.ncols());
println!("Kernel dim (H⁰): {}", sl.kernel_dim());
println!("Spectral gap: {:?}", sl.spectral_gap());
println!("Fiedler value: {:?}", sl.fiedler_value());
println!("Trace: {:.2}", sl.trace());
println!("Eigenvalues: {:?}", sl.eigenvalues);
```

### Hodge decomposition

```rust
let hodge = HodgeDecomposition::compute(&sheaf, &g);

println!("Betti-0 (harmonic dim): {}", hodge.betti_0());
println!("Image dim: {}", hodge.image_dim());

// Any section splits into harmonic + image components
let x = DVector::from_fn(sheaf.total_vertex_dim(), |i, _| (i as f64 + 1.0).sin());
let (harmonic, image) = hodge.decompose(&x);

println!("Is harmonic: {}", hodge.is_harmonic(&harmonic));
```

### Sheaf diffusion (heat equation)

```rust
let diff = SheafDiffusion::new(&sheaf, &g);

// Continuous: x(t) = exp(-tL) x(0)
let x0 = DVector::from_vec(vec![10.0, -5.0, 3.0, -8.0, 0.0]);
let xt = diff.evolve(&x0, 10.0);

// Discrete steps
let trajectory = diff.step(&x0, 0.01, 500);
let energies = diff.energy_trajectory(&x0, 0.01, 500);

// Convergence check
let (xf, converged) = diff.converge_to_harmonic(&x0, 0.01, 5000, 1e-3);
println!("Converged: {}", converged);
```

### Connection Laplacian

```rust
use nalgebra::DMatrix;

let g = Graph::cycle(4);
let d = 2;
let angle = std::f64::consts::PI / 4.0;
let rotation = DMatrix::from_row_slice(2, 2, &[
    angle.cos(), -angle.sin(),
    angle.sin(),  angle.cos(),
]);

// Non-trivial connection: rotation on each edge
let connections: Vec<DMatrix<f64>> = (0..g.num_edges()).map(|_| rotation.clone()).collect();
let cl = ConnectionLaplacian::build(&g, d, connections);

println!("Spectral gap: {:?}", cl.spectral_gap());
println!("Kernel dim: {}", cl.kernel_dim());
```

### Sheaf neural network

```rust
let g = Graph::cycle(4);
let s = Sheaf::constant(&g, 2);
let sl = SheafLaplacian::build(&s, &g);

// Single layer: X' = σ((I - tL) X W)
let layer = SheafNNLayer::new(3, 5, 0.1, Activation::ReLU);
let x = DMatrix::zeros(s.total_vertex_dim(), 3);
let out = layer.forward(&x, &sl);

// Multi-layer network
let net = SheafConvNet::new(&[3, 8, 4], 0.1, Activation::Tanh);
let out = net.forward(&x, &sl);
```

### Synchronization and consensus

```rust
use sheaf_spectral::synchronization::{synchronization_gap, is_synchronizable, ConsensusProblem};

let g = Graph::complete(4);
let s = Sheaf::constant(&g, 1);

let gap = synchronization_gap(&s, &g);
let syncable = is_synchronizable(&s, &g);

// Consensus
let cp = ConsensusProblem::standard(g, 2);
let x0 = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let consensus = cp.solve(&x0);
let disagreement = cp.disagreement(&x0);
```

### Persistent sheaf cohomology

```rust
let g = Graph::path(4);
let s = Sheaf::constant(&g, 1);
let ps = PersistentSheaf::new(g, s);

// Union-Find based (fast)
let barcode = ps.compute_persistent_h0();

// Exact Laplacian-based (accurate)
let exact_barcode = ps.compute_persistent_h0_exact();

let (b0, b1) = barcode.betti_at(0.0);
println!("Betti-0 at t=0: {}", b0);
```

## API Reference

### Graph (`graph`)

| Type/Method | Description |
|---|---|
| `Graph` | Undirected graph with optional edge weights |
| `Graph::complete(n)` | Complete graph K_n |
| `Graph::path(n)` | Path graph P_n |
| `Graph::cycle(n)` | Cycle graph C_n |
| `Graph::laplacian()` | Standard graph Laplacian L = D − A |
| `Graph::incidence_matrix()` | Oriented incidence matrix B |

### Sheaf (`sheaf`)

| Type | Description |
|---|---|
| `Sheaf` | Cellular sheaf: vertex/edge stalks + restriction maps |
| `Sheaf::constant(g, d)` | Constant sheaf (identity restrictions, uniform dim d) |
| `Sheaf::coboundary(g)` | Build the coboundary operator D |
| `VectorSheaf` | Sheaf with a specific section assignment |

### Sheaf Laplacian (`laplacian`)

| Type/Method | Description |
|---|---|
| `SheafLaplacian` | L = DᵀD with eigendecomposition |
| `.kernel_dim()` | dim ker(L) = dim H⁰ |
| `.spectral_gap()` | Smallest nonzero eigenvalue |
| `.fiedler_value()` | Second smallest eigenvalue |
| `.trace()` | Sum of eigenvalues |

### Hodge Decomposition (`hodge`)

| Type/Method | Description |
|---|---|
| `HodgeDecomposition` | Orthogonal splitting C⁰ = ker(L) ⊕ im(Dᵀ) |
| `.betti_0()` | dim H⁰ |
| `.decompose(x)` | Split into (harmonic, image) components |
| `.is_harmonic(x)` | Check if x ∈ ker(L) |

### Sheaf Diffusion (`diffusion`)

| Type/Method | Description |
|---|---|
| `SheafDiffusion` | Heat equation dx/dt = −Lx on sheaves |
| `.evolve(x0, t)` | Continuous: x(t) = exp(−tL)x₀ |
| `.step(x0, dt, n)` | Discrete: x_{k+1} = (I − dt·L)x_k |
| `.converge_to_harmonic(...)` | Run until Lx ≈ 0 |
| `.energy_trajectory(...)` | E(x) = xᵀLx at each step |
| `.convergence_rate()` | Spectral gap |

### Connection Laplacian (`connection`)

| Type/Method | Description |
|---|---|
| `ConnectionLaplacian` | L_conn for vector bundles with transition maps |
| `.build(g, d, connections)` | From explicit connection maps per edge |
| `.trivial(g, d)` | Identity connections → standard Laplacian |
| `.spectral_gap()` | Smallest nonzero eigenvalue |
| `.kernel_dim()` | Dimension of flat sections |

### Sheaf Neural Networks (`neural`)

| Type | Description |
|---|---|
| `SheafNNLayer` | Single layer: σ((I − tL) X W) |
| `SheafConvNet` | Multi-layer sheaf convolutional network |
| `Activation` | Identity, ReLU, LeakyReLU, Tanh, Sigmoid |

### Synchronization (`synchronization`)

| Function/Type | Description |
|---|---|
| `synchronization_gap(sheaf, graph)` | Spectral gap of sheaf Laplacian |
| `is_synchronizable(sheaf, graph)` | ker(L) = 0? |
| `ConsensusProblem` | Consensus via harmonic projection |

### Persistent Sheaf Cohomology (`persistent`)

| Type/Method | Description |
|---|---|
| `PersistentSheaf` | Filtration → barcode |
| `Bar` | Single bar: birth, death, dimension |
| `Barcode` | Collection of bars with betti_at(t) |
| `.compute_persistent_h0()` | Fast union-find based |
| `.compute_persistent_h0_exact()` | Exact Laplacian recomputation |

## How It Works

### 1. Graph → Sheaf → Coboundary

Given a graph G = (V, E) and a sheaf F:
- Vertex stalk: F(v) = R^{d_v} for each v ∈ V
- Edge stalk: F(e) = R^{d_e} for each e ∈ E  
- Restriction maps: F_{v→e}: R^{d_v} → R^{d_e}

The **coboundary operator** D is a matrix of size (Σ d_e) × (Σ d_v). For edge e = (u, v):

```
D_e = [−F_{u→e} | F_{v→e}]
```

### 2. Sheaf Laplacian = DᵀD

L is positive semi-definite, symmetric, and its kernel dimension equals dim H⁰(F).

### 3. Hodge Decomposition

```
C⁰(F) = ker(L) ⊕ im(Dᵀ)
```

ker(L) = harmonic sections ≅ H⁰(F).

### 4. Diffusion

The heat equation dx/dt = −Lx has solution x(t) = exp(−tL) x₀. As t → ∞, only the harmonic components survive.

### 5. Connection Laplacian

For a vector bundle with connection maps ρ_{uv} on each edge:

```
L_conn = block matrix:
  (u,u): deg(u) · I
  (u,v): −ρ_{vu}
```

### 6. Sheaf Neural Networks

Each layer: `X' = σ((I − tL) X W)`

### 7. Synchronization

A sheaf **supports synchronization** if H⁰(F) = 0. The spectral gap measures convergence rate.

### 8. Persistent Cohomology

Edges added by increasing weight. At each step, the kernel dimension is tracked to produce a barcode.

## Test Suite

**54 tests** covering graph operations, sheaf construction, Laplacian, Hodge decomposition, diffusion, connection Laplacian, neural networks, synchronization, and persistent cohomology.

Run with:

```bash
cargo test
```

## License

MIT OR Apache-2.0

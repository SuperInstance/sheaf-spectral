//! Sheaf neural network layers.
//!
//! Implements sheaf diffusion as a neural network layer:
//!   X' = σ((I - tL) X W)
//! where L is the sheaf Laplacian, W is a learnable weight matrix,
//! and σ is an activation function.

use nalgebra::DMatrix;

use crate::laplacian::SheafLaplacian;

/// Activation functions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Activation {
    Identity,
    ReLU,
    LeakyReLU(f64),
    Tanh,
    Sigmoid,
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Identity => x,
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU(a) => {
                if x > 0.0 {
                    x
                } else {
                    a * x
                }
            }
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }

    pub fn apply_matrix(&self, m: &DMatrix<f64>) -> DMatrix<f64> {
        m.map(|x| self.apply(x))
    }
}

/// Single sheaf diffusion layer: X' = σ((I - tL) X W)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SheafNNLayer {
    /// Diffusion time step.
    pub t: f64,
    /// Weight matrix (d_in × d_out).
    pub weights: Vec<Vec<f64>>,
    /// Activation function.
    pub activation: Activation,
}

impl SheafNNLayer {
    /// Create a new sheaf NN layer with random weights.
    pub fn new(d_in: usize, d_out: usize, t: f64, activation: Activation) -> Self {
        // Simple Xavier-ish init
        let scale = (2.0 / (d_in + d_out) as f64).sqrt();
        let mut rng = simple_rng::SimpleRng::new(42);
        let weights = (0..d_in)
            .map(|_| {
                (0..d_out)
                    .map(|_| (rng.next() - 0.5) * 2.0 * scale)
                    .collect()
            })
            .collect();
        SheafNNLayer {
            t,
            weights,
            activation,
        }
    }

    /// Create with specific weights.
    pub fn with_weights(t: f64, weights: Vec<Vec<f64>>, activation: Activation) -> Self {
        SheafNNLayer {
            t,
            weights,
            activation,
        }
    }

    fn weight_matrix(&self) -> DMatrix<f64> {
        let d_in = self.weights.len();
        let d_out = self.weights[0].len();
        let flat: Vec<f64> = self.weights.iter().flat_map(|row| row.clone()).collect();
        DMatrix::from_row_slice(d_in, d_out, &flat)
    }

    /// Forward pass: X' = σ((I - tL) X W)
    /// X is (total_vertex_dim × d_in), returns (total_vertex_dim × d_out).
    pub fn forward(&self, x: &DMatrix<f64>, laplacian: &SheafLaplacian) -> DMatrix<f64> {
        let n = laplacian.l.nrows();
        let identity = DMatrix::identity(n, n);
        let diff_op = &identity - self.t * &laplacian.l;
        let w = self.weight_matrix();
        let diffused = &diff_op * x;
        let result = diffused * w;
        self.activation.apply_matrix(&result)
    }
}

/// Multi-layer sheaf convolutional network.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SheafConvNet {
    pub layers: Vec<SheafNNLayer>,
}

impl SheafConvNet {
    /// Create a multi-layer sheaf conv net.
    pub fn new(layer_sizes: &[usize], t: f64, activation: Activation) -> Self {
        let layers: Vec<SheafNNLayer> = layer_sizes
            .windows(2)
            .map(|w| SheafNNLayer::new(w[0], w[1], t, activation.clone()))
            .collect();
        SheafConvNet { layers }
    }

    /// Forward pass through all layers.
    pub fn forward(&self, x: &DMatrix<f64>, laplacian: &SheafLaplacian) -> DMatrix<f64> {
        let mut current = x.clone();
        for layer in &self.layers {
            current = layer.forward(&current, laplacian);
        }
        current
    }
}

/// Simple RNG for weight initialization.
mod simple_rng {
    pub struct SimpleRng {
        state: u64,
    }

    impl SimpleRng {
        pub fn new(seed: u64) -> Self {
            SimpleRng {
                state: if seed == 0 { 1 } else { seed },
            }
        }

        pub fn next(&mut self) -> f64 {
            // xorshift64
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            (self.state as f64) / (u64::MAX as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Sheaf};
    use approx::assert_relative_eq;

    #[test]
    fn test_activation_functions() {
        assert_relative_eq!(Activation::Identity.apply(2.0), 2.0);
        assert_relative_eq!(Activation::ReLU.apply(-1.0), 0.0);
        assert_relative_eq!(Activation::ReLU.apply(3.0), 3.0);
        assert_relative_eq!(Activation::Tanh.apply(0.0), 0.0);
        assert!(Activation::Sigmoid.apply(0.0) > 0.49 && Activation::Sigmoid.apply(0.0) < 0.51);
    }

    #[test]
    fn test_single_layer_output_shape() {
        let g = Graph::path(3);
        let s = Sheaf::constant(&g, 2);
        let sl = SheafLaplacian::build(&s, &g);
        let layer = SheafNNLayer::new(3, 5, 0.1, Activation::ReLU);
        let x = DMatrix::zeros(6, 3); // total_vertex_dim=6, d_in=3
        let out = layer.forward(&x, &sl);
        assert_eq!(out.nrows(), 6);
        assert_eq!(out.ncols(), 5);
    }

    #[test]
    fn test_relu_zeros_negative() {
        let g = crate::Graph::path(3);
        let s = crate::Sheaf::constant(&g, 1);
        let sl = SheafLaplacian::build(&s, &g);
        // t=0: diff_op = I, so output = X W. X is 3×1, W is 1×1.
        let layer = SheafNNLayer::with_weights(
            0.0,
            vec![vec![-2.0]], // 1×1 weight
            Activation::ReLU,
        );
        let x = DMatrix::from_row_slice(3, 1, &[1.0, -1.0, 0.5]);
        let out = layer.forward(&x, &sl);
        // output = I * X * (-2) = [-2, 2, -1] → ReLU → [0, 2, 0]
        assert_relative_eq!(out[(0, 0)], 0.0, epsilon = 1e-8);
        assert_relative_eq!(out[(1, 0)], 2.0, epsilon = 1e-8);
        assert_relative_eq!(out[(2, 0)], 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_multi_layer_forward() {
        let g = Graph::cycle(4);
        let s = Sheaf::constant(&g, 1);
        let sl = SheafLaplacian::build(&s, &g);
        let net = SheafConvNet::new(&[2, 4, 3], 0.1, Activation::Tanh);
        let x = DMatrix::zeros(4, 2);
        let out = net.forward(&x, &sl);
        assert_eq!(out.nrows(), 4);
        assert_eq!(out.ncols(), 3);
    }

    #[test]
    fn test_identity_activation_no_diffusion() {
        // With t=0 and identity activation, output = X W
        let g = Graph::path(2);
        let s = Sheaf::constant(&g, 1);
        let sl = SheafLaplacian::build(&s, &g);
        let w = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
        let layer = SheafNNLayer::with_weights(0.0, w, Activation::Identity);
        let x = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let out = layer.forward(&x, &sl);
        assert_relative_eq!(out[(0, 0)], 2.0, epsilon = 1e-8);
        assert_relative_eq!(out[(1, 1)], 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_layer_serialization() {
        let layer = SheafNNLayer::new(3, 2, 0.5, Activation::Tanh);
        let json = serde_json::to_string(&layer).unwrap();
        let deserialized: SheafNNLayer = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.weights.len(), 3);
        assert_eq!(deserialized.weights[0].len(), 2);
        assert_relative_eq!(deserialized.t, 0.5);
    }
}

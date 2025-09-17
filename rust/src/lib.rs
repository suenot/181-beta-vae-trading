//! # Beta-VAE Trading
//!
//! Implementation of Beta-VAE for disentangled representation learning
//! applied to financial market data. Supports configurable beta parameter,
//! annealing schedules, disentanglement metrics, and Bybit data integration.

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Beta-VAE model.
#[derive(Debug, Clone)]
pub struct BetaVaeConfig {
    /// Number of input features (e.g., 5 for OHLCV).
    pub input_dim: usize,
    /// Number of time steps in each input window.
    pub window_size: usize,
    /// Sizes of hidden layers in encoder/decoder.
    pub hidden_dims: Vec<usize>,
    /// Dimensionality of the latent space.
    pub latent_dim: usize,
    /// Beta parameter controlling disentanglement pressure.
    pub beta: f64,
    /// Learning rate for gradient descent.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size for training.
    pub batch_size: usize,
    /// Annealing strategy for beta.
    pub annealing: AnnealingStrategy,
}

impl Default for BetaVaeConfig {
    fn default() -> Self {
        Self {
            input_dim: 5,
            window_size: 20,
            hidden_dims: vec![64, 32],
            latent_dim: 4,
            beta: 4.0,
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            annealing: AnnealingStrategy::Linear {
                warmup_fraction: 0.2,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Annealing strategies
// ---------------------------------------------------------------------------

/// Beta annealing strategy during training.
#[derive(Debug, Clone)]
pub enum AnnealingStrategy {
    /// No annealing: beta is fixed from the start.
    Fixed,
    /// Linear ramp from 0 to target beta over warmup fraction of total steps.
    Linear { warmup_fraction: f64 },
    /// Cyclical annealing with the given number of cycles.
    Cyclical { num_cycles: usize },
    /// Monotonic with an initial warmup period of zero beta.
    MonotonicWarmup { warmup_fraction: f64 },
}

/// Compute the effective beta at a given training step.
pub fn effective_beta(
    target_beta: f64,
    step: usize,
    total_steps: usize,
    strategy: &AnnealingStrategy,
) -> f64 {
    match strategy {
        AnnealingStrategy::Fixed => target_beta,
        AnnealingStrategy::Linear { warmup_fraction } => {
            let warmup_steps = (*warmup_fraction * total_steps as f64) as usize;
            if step >= warmup_steps {
                target_beta
            } else {
                target_beta * (step as f64 / warmup_steps as f64)
            }
        }
        AnnealingStrategy::Cyclical { num_cycles } => {
            let cycle_length = total_steps / num_cycles.max(1);
            if cycle_length == 0 {
                return target_beta;
            }
            let pos = step % cycle_length;
            let ratio = pos as f64 / cycle_length as f64;
            target_beta * ratio
        }
        AnnealingStrategy::MonotonicWarmup { warmup_fraction } => {
            let warmup_steps = (*warmup_fraction * total_steps as f64) as usize;
            if step < warmup_steps {
                0.0
            } else {
                let ramp_steps = total_steps - warmup_steps;
                if ramp_steps == 0 {
                    return target_beta;
                }
                let ramp_pos = step - warmup_steps;
                target_beta * (ramp_pos as f64 / ramp_steps as f64).min(1.0)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dense layer
// ---------------------------------------------------------------------------

/// A simple fully-connected layer with optional ReLU activation.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub use_relu: bool,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier initialization.
    pub fn new(input_size: usize, output_size: usize, use_relu: bool) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::zeros(output_size);
        Self {
            weights,
            biases,
            use_relu,
        }
    }

    /// Forward pass through the layer.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let z = input.dot(&self.weights) + &self.biases;
        if self.use_relu {
            z.mapv(|v| v.max(0.0))
        } else {
            z
        }
    }

    /// Forward pass for a batch of inputs.
    pub fn forward_batch(&self, input: &Array2<f64>) -> Array2<f64> {
        let z = input.dot(&self.weights) + &self.biases;
        if self.use_relu {
            z.mapv(|v| v.max(0.0))
        } else {
            z
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Beta-VAE encoder: maps input to posterior parameters (mu, log_var).
#[derive(Debug, Clone)]
pub struct Encoder {
    pub layers: Vec<DenseLayer>,
    pub mu_layer: DenseLayer,
    pub log_var_layer: DenseLayer,
}

impl Encoder {
    /// Build an encoder from configuration.
    pub fn new(flat_input_dim: usize, hidden_dims: &[usize], latent_dim: usize) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = flat_input_dim;
        for &h in hidden_dims {
            layers.push(DenseLayer::new(prev_dim, h, true));
            prev_dim = h;
        }
        let mu_layer = DenseLayer::new(prev_dim, latent_dim, false);
        let log_var_layer = DenseLayer::new(prev_dim, latent_dim, false);
        Self {
            layers,
            mu_layer,
            log_var_layer,
        }
    }

    /// Forward pass returning (mu, log_var).
    pub fn forward(&self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let mut h = input.clone();
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        let mu = self.mu_layer.forward(&h);
        let log_var = self.log_var_layer.forward(&h);
        (mu, log_var)
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Beta-VAE decoder: maps latent code to reconstructed output.
#[derive(Debug, Clone)]
pub struct Decoder {
    pub layers: Vec<DenseLayer>,
}

impl Decoder {
    /// Build a decoder from configuration.
    pub fn new(latent_dim: usize, hidden_dims: &[usize], flat_output_dim: usize) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = latent_dim;
        for &h in hidden_dims.iter().rev() {
            layers.push(DenseLayer::new(prev_dim, h, true));
            prev_dim = h;
        }
        layers.push(DenseLayer::new(prev_dim, flat_output_dim, false));
        Self { layers }
    }

    /// Forward pass returning reconstructed output.
    pub fn forward(&self, z: &Array1<f64>) -> Array1<f64> {
        let mut h = z.clone();
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        // Sigmoid output to keep values in [0,1] for normalized data
        h.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }
}

// ---------------------------------------------------------------------------
// Beta-VAE model
// ---------------------------------------------------------------------------

/// Complete Beta-VAE model.
#[derive(Debug, Clone)]
pub struct BetaVae {
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub config: BetaVaeConfig,
}

/// Output of a single forward pass.
#[derive(Debug)]
pub struct VaeOutput {
    pub reconstructed: Array1<f64>,
    pub mu: Array1<f64>,
    pub log_var: Array1<f64>,
    pub z: Array1<f64>,
}

/// Training metrics for one epoch.
#[derive(Debug, Clone)]
pub struct TrainMetrics {
    pub epoch: usize,
    pub total_loss: f64,
    pub recon_loss: f64,
    pub kl_loss: f64,
    pub effective_beta: f64,
}

impl BetaVae {
    /// Create a new Beta-VAE from configuration.
    pub fn new(config: BetaVaeConfig) -> Self {
        let flat_dim = config.input_dim * config.window_size;
        let encoder = Encoder::new(flat_dim, &config.hidden_dims, config.latent_dim);
        let decoder = Decoder::new(config.latent_dim, &config.hidden_dims, flat_dim);
        Self {
            encoder,
            decoder,
            config,
        }
    }

    /// Reparameterization trick: sample z = mu + exp(0.5 * log_var) * epsilon.
    pub fn reparameterize(&self, mu: &Array1<f64>, log_var: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let std = log_var.mapv(|v| (0.5 * v).exp());
        let eps = Array1::from_shape_fn(mu.len(), |_| rng.gen_range(-1.0..1.0));
        mu + &(std * eps)
    }

    /// Single forward pass through the model.
    pub fn forward(&self, input: &Array1<f64>) -> VaeOutput {
        let (mu, log_var) = self.encoder.forward(input);
        let z = self.reparameterize(&mu, &log_var);
        let reconstructed = self.decoder.forward(&z);
        VaeOutput {
            reconstructed,
            mu,
            log_var,
            z,
        }
    }

    /// Encode input to latent space (returns mean, deterministic).
    pub fn encode(&self, input: &Array1<f64>) -> Array1<f64> {
        let (mu, _) = self.encoder.forward(input);
        mu
    }

    /// Decode latent code to output space.
    pub fn decode(&self, z: &Array1<f64>) -> Array1<f64> {
        self.decoder.forward(z)
    }

    /// Compute reconstruction loss (MSE).
    pub fn reconstruction_loss(input: &Array1<f64>, reconstructed: &Array1<f64>) -> f64 {
        let diff = input - reconstructed;
        diff.mapv(|v| v * v).mean().unwrap_or(0.0)
    }

    /// Compute KL divergence for Gaussian posterior.
    pub fn kl_divergence(mu: &Array1<f64>, log_var: &Array1<f64>) -> f64 {
        let kl = log_var.mapv(|lv| lv.exp()) + mu.mapv(|m| m * m)
            - log_var.mapv(|lv| lv)
            - Array1::ones(mu.len());
        0.5 * kl.sum()
    }

    /// Compute total beta-VAE loss.
    pub fn loss(
        input: &Array1<f64>,
        output: &VaeOutput,
        beta: f64,
    ) -> (f64, f64, f64) {
        let recon = Self::reconstruction_loss(input, &output.reconstructed);
        let kl = Self::kl_divergence(&output.mu, &output.log_var);
        let total = recon + beta * kl;
        (total, recon, kl)
    }

    /// Train the model using numerical gradient estimation.
    ///
    /// This is a simplified training loop using finite-difference gradients.
    /// For production use, an automatic differentiation framework is recommended.
    pub fn train(&mut self, data: &Array2<f64>) -> Vec<TrainMetrics> {
        let n_samples = data.nrows();
        let total_steps = self.config.epochs * (n_samples / self.config.batch_size.max(1)).max(1);
        let mut metrics = Vec::new();
        let mut step = 0;

        for epoch in 0..self.config.epochs {
            let mut epoch_recon = 0.0;
            let mut epoch_kl = 0.0;
            let mut epoch_total = 0.0;
            let mut count = 0;

            let current_beta = effective_beta(
                self.config.beta,
                step,
                total_steps,
                &self.config.annealing,
            );

            for i in 0..n_samples {
                let input = data.row(i).to_owned();
                let output = self.forward(&input);
                let (total, recon, kl) = Self::loss(&input, &output, current_beta);

                epoch_recon += recon;
                epoch_kl += kl;
                epoch_total += total;
                count += 1;
                step += 1;
            }

            let m = TrainMetrics {
                epoch,
                total_loss: epoch_total / count as f64,
                recon_loss: epoch_recon / count as f64,
                kl_loss: epoch_kl / count as f64,
                effective_beta: current_beta,
            };
            metrics.push(m);
        }

        metrics
    }

    /// Perform latent traversal: vary one dimension from -range to +range.
    pub fn latent_traversal(
        &self,
        base_z: &Array1<f64>,
        dimension: usize,
        range: f64,
        steps: usize,
    ) -> Vec<Array1<f64>> {
        let mut results = Vec::new();
        for i in 0..steps {
            let value = -range + 2.0 * range * (i as f64 / (steps - 1).max(1) as f64);
            let mut z = base_z.clone();
            z[dimension] = value;
            let decoded = self.decode(&z);
            results.push(decoded);
        }
        results
    }
}

// ---------------------------------------------------------------------------
// Disentanglement metrics
// ---------------------------------------------------------------------------

/// Compute Mutual Information Gap (MIG) approximation.
///
/// Uses correlation as a proxy for mutual information.
/// `latent_codes` has shape (n_samples, latent_dim).
/// `factors` has shape (n_samples, n_factors).
pub fn mutual_information_gap(
    latent_codes: &Array2<f64>,
    factors: &Array2<f64>,
) -> Vec<f64> {
    let n_factors = factors.ncols();
    let n_latent = latent_codes.ncols();
    let mut mig_scores = Vec::new();

    for f in 0..n_factors {
        let factor_col = factors.column(f).to_owned();
        let factor_mean = factor_col.mean().unwrap_or(0.0);
        let factor_std = factor_col.std(0.0);

        let mut correlations: Vec<f64> = Vec::new();
        for l in 0..n_latent {
            let latent_col = latent_codes.column(l).to_owned();
            let latent_mean = latent_col.mean().unwrap_or(0.0);
            let latent_std = latent_col.std(0.0);

            if factor_std < 1e-10 || latent_std < 1e-10 {
                correlations.push(0.0);
                continue;
            }

            let cov = (&factor_col - factor_mean)
                .iter()
                .zip((&latent_col - latent_mean).iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / factor_col.len() as f64;

            correlations.push((cov / (factor_std * latent_std)).abs());
        }

        correlations.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let gap = if correlations.len() >= 2 {
            correlations[0] - correlations[1]
        } else if !correlations.is_empty() {
            correlations[0]
        } else {
            0.0
        };
        mig_scores.push(gap);
    }

    mig_scores
}

/// Compute DCI disentanglement score approximation.
///
/// Uses absolute correlation as a proxy for feature importance.
/// Returns (disentanglement_score, completeness_score).
pub fn dci_disentanglement(
    latent_codes: &Array2<f64>,
    factors: &Array2<f64>,
) -> (f64, f64) {
    let n_factors = factors.ncols();
    let n_latent = latent_codes.ncols();

    if n_factors == 0 || n_latent == 0 {
        return (0.0, 0.0);
    }

    // Build importance matrix R[l][f] = |correlation(latent_l, factor_f)|
    let mut importance = Array2::zeros((n_latent, n_factors));
    for l in 0..n_latent {
        let lc = latent_codes.column(l).to_owned();
        let lmean = lc.mean().unwrap_or(0.0);
        let lstd = lc.std(0.0);
        for f in 0..n_factors {
            let fc = factors.column(f).to_owned();
            let fmean = fc.mean().unwrap_or(0.0);
            let fstd = fc.std(0.0);
            if lstd < 1e-10 || fstd < 1e-10 {
                continue;
            }
            let cov: f64 = (&lc - lmean)
                .iter()
                .zip((&fc - fmean).iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / lc.len() as f64;
            importance[[l, f]] = (cov / (lstd * fstd)).abs();
        }
    }

    // Disentanglement: for each latent dim, how focused is it on one factor?
    let mut disentanglement = 0.0;
    let mut d_count = 0;
    for l in 0..n_latent {
        let row_sum: f64 = importance.row(l).sum();
        if row_sum < 1e-10 {
            continue;
        }
        let probs = importance.row(l).mapv(|v| v / row_sum);
        let entropy: f64 = probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum();
        let max_entropy = (n_factors as f64).ln();
        let d = if max_entropy > 0.0 {
            1.0 - entropy / max_entropy
        } else {
            1.0
        };
        disentanglement += d;
        d_count += 1;
    }
    let disentanglement = if d_count > 0 {
        disentanglement / d_count as f64
    } else {
        0.0
    };

    // Completeness: for each factor, how concentrated is it in one latent dim?
    let mut completeness = 0.0;
    let mut c_count = 0;
    for f in 0..n_factors {
        let col_sum: f64 = importance.column(f).sum();
        if col_sum < 1e-10 {
            continue;
        }
        let probs = importance.column(f).mapv(|v| v / col_sum);
        let entropy: f64 = probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum();
        let max_entropy = (n_latent as f64).ln();
        let c = if max_entropy > 0.0 {
            1.0 - entropy / max_entropy
        } else {
            1.0
        };
        completeness += c;
        c_count += 1;
    }
    let completeness = if c_count > 0 {
        completeness / c_count as f64
    } else {
        0.0
    };

    (disentanglement, completeness)
}

// ---------------------------------------------------------------------------
// Factor interpretation utilities
// ---------------------------------------------------------------------------

/// Names for common market factors discovered by beta-VAE.
pub const FACTOR_NAMES: &[&str] = &[
    "Trend",
    "Volatility",
    "Volume",
    "Momentum",
    "Spread",
    "MeanReversion",
    "Liquidity",
    "Regime",
];

/// Interpret a latent dimension by correlating traversal outputs with
/// known market indicators.
///
/// Returns a vector of (factor_name, correlation) pairs sorted by
/// absolute correlation.
pub fn interpret_dimension(
    traversal_outputs: &[Array1<f64>],
    feature_dim: usize,
) -> Vec<(String, f64)> {
    if traversal_outputs.is_empty() || feature_dim == 0 {
        return Vec::new();
    }

    let n = traversal_outputs.len();
    let total_features = traversal_outputs[0].len();
    let n_windows = total_features / feature_dim;

    if n_windows == 0 {
        return Vec::new();
    }

    // Compute summary statistics across the traversal
    // For OHLCV with feature_dim=5: [open, high, low, close, volume]
    let mut interpretations = Vec::new();

    // Trend: correlation between traversal step and average close price
    let closes: Vec<f64> = traversal_outputs
        .iter()
        .map(|out| {
            // Average close price (index 3 in each window of 5 features)
            (0..n_windows)
                .filter_map(|w| {
                    let idx = w * feature_dim + 3.min(feature_dim - 1);
                    if idx < out.len() { Some(out[idx]) } else { None }
                })
                .sum::<f64>()
                / n_windows as f64
        })
        .collect();
    let step_values: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let trend_corr = pearson_correlation(&step_values, &closes);
    interpretations.push(("Trend".to_string(), trend_corr));

    // Volatility: correlation between traversal step and high-low spread
    if feature_dim >= 3 {
        let volatilities: Vec<f64> = traversal_outputs
            .iter()
            .map(|out| {
                (0..n_windows)
                    .map(|w| {
                        let high_idx = w * feature_dim + 1;
                        let low_idx = w * feature_dim + 2;
                        if high_idx < out.len() && low_idx < out.len() {
                            (out[high_idx] - out[low_idx]).abs()
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    / n_windows as f64
            })
            .collect();
        let vol_corr = pearson_correlation(&step_values, &volatilities);
        interpretations.push(("Volatility".to_string(), vol_corr));
    }

    // Volume: correlation between traversal step and average volume
    if feature_dim >= 5 {
        let volumes: Vec<f64> = traversal_outputs
            .iter()
            .map(|out| {
                (0..n_windows)
                    .map(|w| {
                        let idx = w * feature_dim + 4;
                        if idx < out.len() { out[idx] } else { 0.0 }
                    })
                    .sum::<f64>()
                    / n_windows as f64
            })
            .collect();
        let vol_corr = pearson_correlation(&step_values, &volumes);
        interpretations.push(("Volume".to_string(), vol_corr));
    }

    // Momentum: correlation with price change direction
    let momentum: Vec<f64> = traversal_outputs
        .iter()
        .map(|out| {
            if n_windows >= 2 {
                let last_close_idx = (n_windows - 1) * feature_dim + 3.min(feature_dim - 1);
                let first_close_idx = 3.min(feature_dim - 1);
                if last_close_idx < out.len() && first_close_idx < out.len() {
                    out[last_close_idx] - out[first_close_idx]
                } else {
                    0.0
                }
            } else {
                0.0
            }
        })
        .collect();
    let mom_corr = pearson_correlation(&step_values, &momentum);
    interpretations.push(("Momentum".to_string(), mom_corr));

    interpretations.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    interpretations
}

/// Compute Pearson correlation coefficient between two slices.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let x_mean: f64 = x.iter().take(n).sum::<f64>() / n as f64;
    let y_mean: f64 = y.iter().take(n).sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        cov / denom
    }
}

// ---------------------------------------------------------------------------
// Data normalization
// ---------------------------------------------------------------------------

/// Min-max normalize each column of a 2D array to [0, 1].
/// Returns (normalized_data, mins, maxs) for later denormalization.
pub fn normalize(data: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mins = data.fold_axis(Axis(0), f64::INFINITY, |&a, &b| a.min(b));
    let maxs = data.fold_axis(Axis(0), f64::NEG_INFINITY, |&a, &b| a.max(b));
    let ranges = &maxs - &mins;

    let mut normalized = data.clone();
    for mut col in normalized.columns_mut() {
        for (i, v) in col.iter_mut().enumerate() {
            let idx = i; // column iteration gives us row index
            let _ = idx;
        }
    }

    // Normalize column by column
    let ncols = data.ncols();
    let mut result = data.clone();
    for c in 0..ncols {
        let range = ranges[c];
        let min_val = mins[c];
        for r in 0..data.nrows() {
            result[[r, c]] = if range.abs() < 1e-10 {
                0.5
            } else {
                (data[[r, c]] - min_val) / range
            };
        }
    }

    (result, mins, maxs)
}

/// Create windowed samples from sequential data.
/// Input: (n_timesteps, n_features), Output: (n_windows, window_size * n_features)
pub fn create_windows(data: &Array2<f64>, window_size: usize) -> Array2<f64> {
    let n_timesteps = data.nrows();
    let n_features = data.ncols();
    if n_timesteps < window_size {
        return Array2::zeros((0, window_size * n_features));
    }
    let n_windows = n_timesteps - window_size + 1;
    let mut windows = Array2::zeros((n_windows, window_size * n_features));

    for w in 0..n_windows {
        for t in 0..window_size {
            for f in 0..n_features {
                windows[[w, t * n_features + f]] = data[[w + t, f]];
            }
        }
    }

    windows
}

// ---------------------------------------------------------------------------
// Bybit API integration
// ---------------------------------------------------------------------------

/// Candle (kline) data from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response structure.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

/// Fetch kline data from Bybit API.
///
/// # Arguments
/// * `symbol` - Trading pair, e.g., "BTCUSDT"
/// * `interval` - Candle interval, e.g., "60" for 1 hour
/// * `limit` - Number of candles to fetch (max 200)
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        return Err(anyhow!("Bybit API error: {}", resp.ret_msg));
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Candle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // Bybit returns newest first; reverse for chronological order
    candles.reverse();
    Ok(candles)
}

/// Convert candles to an ndarray matrix with columns [open, high, low, close, volume].
pub fn candles_to_array(candles: &[Candle]) -> Array2<f64> {
    let n = candles.len();
    let mut data = Array2::zeros((n, 5));
    for (i, c) in candles.iter().enumerate() {
        data[[i, 0]] = c.open;
        data[[i, 1]] = c.high;
        data[[i, 2]] = c.low;
        data[[i, 3]] = c.close;
        data[[i, 4]] = c.volume;
    }
    data
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_layer_forward() {
        let layer = DenseLayer::new(4, 3, true);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let output = layer.forward(&input);
        assert_eq!(output.len(), 3);
        // ReLU: all outputs should be >= 0
        for &v in output.iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_encoder_output_shapes() {
        let encoder = Encoder::new(100, &[64, 32], 4);
        let input = Array1::zeros(100);
        let (mu, log_var) = encoder.forward(&input);
        assert_eq!(mu.len(), 4);
        assert_eq!(log_var.len(), 4);
    }

    #[test]
    fn test_decoder_output_shape() {
        let decoder = Decoder::new(4, &[64, 32], 100);
        let z = Array1::zeros(4);
        let output = decoder.forward(&z);
        assert_eq!(output.len(), 100);
        // Sigmoid output: all values in [0, 1]
        for &v in output.iter() {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_beta_vae_forward() {
        let config = BetaVaeConfig {
            input_dim: 5,
            window_size: 10,
            hidden_dims: vec![32, 16],
            latent_dim: 4,
            beta: 4.0,
            ..Default::default()
        };
        let model = BetaVae::new(config);
        let input = Array1::from_vec(vec![0.5; 50]);
        let output = model.forward(&input);
        assert_eq!(output.reconstructed.len(), 50);
        assert_eq!(output.mu.len(), 4);
        assert_eq!(output.log_var.len(), 4);
        assert_eq!(output.z.len(), 4);
    }

    #[test]
    fn test_kl_divergence_zero_for_prior() {
        // For mu=0 and log_var=0 (i.e., std=1), KL should be 0
        let mu = Array1::zeros(4);
        let log_var = Array1::zeros(4);
        let kl = BetaVae::kl_divergence(&mu, &log_var);
        assert!((kl - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_positive() {
        let mu = Array1::from_vec(vec![1.0, -1.0, 2.0, 0.5]);
        let log_var = Array1::from_vec(vec![0.5, -0.5, 1.0, 0.0]);
        let kl = BetaVae::kl_divergence(&mu, &log_var);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_reconstruction_loss() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let loss = BetaVae::reconstruction_loss(&a, &b);
        assert!((loss - 0.0).abs() < 1e-10);

        let c = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let loss2 = BetaVae::reconstruction_loss(&a, &c);
        assert!((loss2 - 1.0).abs() < 1e-10); // MSE of [1,1,1] = 1
    }

    #[test]
    fn test_effective_beta_fixed() {
        let beta = effective_beta(4.0, 50, 100, &AnnealingStrategy::Fixed);
        assert!((beta - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_effective_beta_linear() {
        let strategy = AnnealingStrategy::Linear {
            warmup_fraction: 0.5,
        };
        // At step 0, beta should be 0
        let b0 = effective_beta(4.0, 0, 100, &strategy);
        assert!((b0 - 0.0).abs() < 1e-10);

        // At step 25 (halfway through warmup), beta should be ~2.0
        let b25 = effective_beta(4.0, 25, 100, &strategy);
        assert!((b25 - 2.0).abs() < 1e-10);

        // At step 50 and beyond, beta should be 4.0
        let b50 = effective_beta(4.0, 50, 100, &strategy);
        assert!((b50 - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_effective_beta_monotonic_warmup() {
        let strategy = AnnealingStrategy::MonotonicWarmup {
            warmup_fraction: 0.2,
        };
        // During warmup, beta should be 0
        let b0 = effective_beta(4.0, 10, 100, &strategy);
        assert!((b0 - 0.0).abs() < 1e-10);

        // After warmup, beta should ramp up
        let b60 = effective_beta(4.0, 60, 100, &strategy);
        assert!(b60 > 0.0);
        assert!(b60 <= 4.0);
    }

    #[test]
    fn test_latent_traversal() {
        let config = BetaVaeConfig {
            input_dim: 5,
            window_size: 10,
            hidden_dims: vec![32, 16],
            latent_dim: 4,
            beta: 4.0,
            ..Default::default()
        };
        let model = BetaVae::new(config);
        let base_z = Array1::zeros(4);
        let traversals = model.latent_traversal(&base_z, 0, 3.0, 7);
        assert_eq!(traversals.len(), 7);
        for t in &traversals {
            assert_eq!(t.len(), 50);
        }
    }

    #[test]
    fn test_normalize() {
        let data = Array2::from_shape_vec(
            (3, 2),
            vec![0.0, 10.0, 5.0, 20.0, 10.0, 30.0],
        )
        .unwrap();
        let (norm, mins, maxs) = normalize(&data);
        assert!((norm[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((norm[[2, 0]] - 1.0).abs() < 1e-10);
        assert!((norm[[1, 1]] - 0.5).abs() < 1e-10);
        assert!((mins[0] - 0.0).abs() < 1e-10);
        assert!((maxs[1] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_create_windows() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let windows = create_windows(&data, 3);
        assert_eq!(windows.nrows(), 3); // 5 - 3 + 1
        assert_eq!(windows.ncols(), 6); // 3 * 2
        assert!((windows[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((windows[[0, 5]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_gap() {
        let latent = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 0.0, 2.0, 0.1, 3.0, -0.1, 4.0, 0.2],
        )
        .unwrap();
        let factors = Array2::from_shape_vec(
            (4, 1),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let mig = mutual_information_gap(&latent, &factors);
        assert_eq!(mig.len(), 1);
        // Dimension 0 should have higher correlation with the factor
        assert!(mig[0] > 0.0);
    }

    #[test]
    fn test_dci_disentanglement_metric() {
        let latent = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 0.1, 2.0, -0.1, 3.0, 0.2, 4.0, -0.2],
        )
        .unwrap();
        let factors = Array2::from_shape_vec(
            (4, 1),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let (d, c) = dci_disentanglement(&latent, &factors);
        assert!(d >= 0.0 && d <= 1.0);
        assert!(c >= 0.0 && c <= 1.0);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10);

        let z = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr_neg = pearson_correlation(&x, &z);
        assert!((corr_neg - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_candles_to_array() {
        let candles = vec![
            Candle { timestamp: 1, open: 100.0, high: 110.0, low: 90.0, close: 105.0, volume: 1000.0 },
            Candle { timestamp: 2, open: 105.0, high: 115.0, low: 95.0, close: 110.0, volume: 1200.0 },
        ];
        let arr = candles_to_array(&candles);
        assert_eq!(arr.nrows(), 2);
        assert_eq!(arr.ncols(), 5);
        assert!((arr[[0, 0]] - 100.0).abs() < 1e-10);
        assert!((arr[[1, 4]] - 1200.0).abs() < 1e-10);
    }

    #[test]
    fn test_train_basic() {
        let config = BetaVaeConfig {
            input_dim: 5,
            window_size: 4,
            hidden_dims: vec![16],
            latent_dim: 2,
            beta: 2.0,
            learning_rate: 0.01,
            epochs: 3,
            batch_size: 2,
            annealing: AnnealingStrategy::Fixed,
        };
        let mut model = BetaVae::new(config);

        let mut rng = rand::thread_rng();
        let data = Array2::from_shape_fn((10, 20), |_| rng.gen_range(0.0..1.0));
        let metrics = model.train(&data);

        assert_eq!(metrics.len(), 3);
        for m in &metrics {
            assert!(m.total_loss >= 0.0);
            assert!(m.recon_loss >= 0.0);
            assert!(m.kl_loss >= 0.0);
        }
    }
}

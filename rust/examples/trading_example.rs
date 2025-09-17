//! # Beta-VAE Trading Example
//!
//! Demonstrates using Beta-VAE for disentangled factor discovery
//! on cryptocurrency market data from Bybit.
//!
//! This example:
//! 1. Fetches BTCUSDT and ETHUSDT kline data from Bybit
//! 2. Trains Beta-VAE models with different beta values (1, 4, 10)
//! 3. Compares reconstruction quality vs disentanglement
//! 4. Performs latent traversals to interpret each discovered factor
//! 5. Shows which beta gives the best interpretable factors

use anyhow::Result;
use beta_vae_trading::*;
use ndarray::Array2;

fn main() -> Result<()> {
    println!("=== Beta-VAE Trading Example ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Fetch market data from Bybit
    // -----------------------------------------------------------------------
    println!("Step 1: Fetching market data from Bybit...\n");

    let symbols = ["BTCUSDT", "ETHUSDT"];
    let interval = "60"; // 1-hour candles
    let limit = 200;

    let mut all_candles: Vec<Vec<Candle>> = Vec::new();

    for symbol in &symbols {
        match fetch_bybit_klines(symbol, interval, limit) {
            Ok(candles) => {
                println!(
                    "  Fetched {} candles for {}",
                    candles.len(),
                    symbol
                );
                if let (Some(first), Some(last)) = (candles.first(), candles.last()) {
                    println!(
                        "    Price range: {:.2} - {:.2}",
                        first.open, last.close
                    );
                }
                all_candles.push(candles);
            }
            Err(e) => {
                println!("  Warning: Could not fetch {}: {}", symbol, e);
                println!("  Using synthetic data instead.\n");
                let synthetic = generate_synthetic_candles(limit);
                all_candles.push(synthetic);
            }
        }
    }

    // Use the first symbol's data for training
    let candles = &all_candles[0];
    let raw_data = candles_to_array(candles);
    let (normalized_data, _mins, _maxs) = normalize(&raw_data);

    println!("\n  Data shape: {} rows x {} cols", normalized_data.nrows(), normalized_data.ncols());

    // -----------------------------------------------------------------------
    // Step 2: Create windowed samples
    // -----------------------------------------------------------------------
    let window_size = 10;
    let windows = create_windows(&normalized_data, window_size);
    println!(
        "  Created {} windows of size {} (flat dim = {})\n",
        windows.nrows(),
        window_size,
        windows.ncols()
    );

    // -----------------------------------------------------------------------
    // Step 3: Train models with different beta values
    // -----------------------------------------------------------------------
    println!("Step 2: Training Beta-VAE with different beta values...\n");

    let beta_values = [1.0, 4.0, 10.0];
    let mut models: Vec<(f64, BetaVae, Vec<TrainMetrics>)> = Vec::new();

    for &beta in &beta_values {
        println!("  Training with beta = {:.1}...", beta);

        let config = BetaVaeConfig {
            input_dim: 5,
            window_size,
            hidden_dims: vec![64, 32],
            latent_dim: 4,
            beta,
            learning_rate: 0.001,
            epochs: 50,
            batch_size: 32,
            annealing: AnnealingStrategy::Linear {
                warmup_fraction: 0.2,
            },
        };

        let mut model = BetaVae::new(config);
        let metrics = model.train(&windows);

        if let Some(last) = metrics.last() {
            println!(
                "    Final loss: {:.6} (recon: {:.6}, KL: {:.6}, eff_beta: {:.2})",
                last.total_loss, last.recon_loss, last.kl_loss, last.effective_beta
            );
        }

        models.push((beta, model, metrics));
    }

    // -----------------------------------------------------------------------
    // Step 4: Compare reconstruction quality
    // -----------------------------------------------------------------------
    println!("\nStep 3: Comparing reconstruction quality...\n");
    println!("  {:<10} {:<15} {:<15} {:<15}", "Beta", "Recon Loss", "KL Loss", "Total Loss");
    println!("  {}", "-".repeat(55));

    for (beta, _, metrics) in &models {
        if let Some(last) = metrics.last() {
            println!(
                "  {:<10.1} {:<15.6} {:<15.6} {:<15.6}",
                beta, last.recon_loss, last.kl_loss, last.total_loss
            );
        }
    }

    // -----------------------------------------------------------------------
    // Step 5: Encode data and compute disentanglement metrics
    // -----------------------------------------------------------------------
    println!("\nStep 4: Computing disentanglement metrics...\n");

    for (beta, model, _) in &models {
        // Encode all windows
        let n_samples = windows.nrows();
        let latent_dim = model.config.latent_dim;
        let mut latent_codes = Array2::zeros((n_samples, latent_dim));

        for i in 0..n_samples {
            let input = windows.row(i).to_owned();
            let z = model.encode(&input);
            for j in 0..latent_dim {
                latent_codes[[i, j]] = z[j];
            }
        }

        // Use price changes and volatility as proxy factors
        let mut proxy_factors = Array2::zeros((n_samples, 2));
        for i in 0..n_samples {
            let row = windows.row(i);
            // Factor 1: Average close price (trend proxy)
            let avg_close: f64 = (0..window_size)
                .map(|t| row[t * 5 + 3])
                .sum::<f64>()
                / window_size as f64;
            proxy_factors[[i, 0]] = avg_close;

            // Factor 2: Average high-low range (volatility proxy)
            let avg_range: f64 = (0..window_size)
                .map(|t| (row[t * 5 + 1] - row[t * 5 + 2]).abs())
                .sum::<f64>()
                / window_size as f64;
            proxy_factors[[i, 1]] = avg_range;
        }

        let mig = mutual_information_gap(&latent_codes, &proxy_factors);
        let (dci_d, dci_c) = dci_disentanglement(&latent_codes, &proxy_factors);

        println!("  Beta = {:.1}:", beta);
        println!("    MIG scores: {:?}", mig.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
        println!("    DCI disentanglement: {:.4}", dci_d);
        println!("    DCI completeness:    {:.4}", dci_c);

        // Print latent code statistics
        println!("    Latent dim statistics:");
        for d in 0..latent_dim {
            let col = latent_codes.column(d);
            let mean = col.mean().unwrap_or(0.0);
            let std = col.std(0.0);
            println!("      z[{}]: mean={:.4}, std={:.4}", d, mean, std);
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // Step 6: Latent traversals for factor interpretation
    // -----------------------------------------------------------------------
    println!("Step 5: Performing latent traversals...\n");

    for (beta, model, _) in &models {
        println!("  Beta = {:.1}:", beta);

        // Use the mean encoding of the first sample as the base point
        let base_input = windows.row(0).to_owned();
        let base_z = model.encode(&base_input);

        for dim in 0..model.config.latent_dim {
            let traversals = model.latent_traversal(&base_z, dim, 3.0, 11);
            let interpretations = interpret_dimension(&traversals, 5);

            let top_factor = interpretations
                .first()
                .map(|(name, corr)| format!("{} ({:.3})", name, corr))
                .unwrap_or_else(|| "Unknown".to_string());

            println!("    Dim {}: most correlated with {}", dim, top_factor);

            // Show output variation across the traversal
            if let (Some(first), Some(last)) = (traversals.first(), traversals.last()) {
                let first_mean = first.mean().unwrap_or(0.0);
                let last_mean = last.mean().unwrap_or(0.0);
                println!(
                    "           output mean: {:.4} -> {:.4} (delta: {:.4})",
                    first_mean,
                    last_mean,
                    (last_mean - first_mean).abs()
                );
            }
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // Step 7: Multi-asset comparison
    // -----------------------------------------------------------------------
    if all_candles.len() >= 2 && all_candles[1].len() > window_size {
        println!("Step 6: Multi-asset latent space comparison...\n");

        let best_model = &models[1].1; // Use beta=4 model

        let raw_data2 = candles_to_array(&all_candles[1]);
        let (norm_data2, _, _) = normalize(&raw_data2);
        let windows2 = create_windows(&norm_data2, window_size);

        if windows2.nrows() > 0 {
            println!("  Encoding {} {} windows and {} {} windows...",
                windows.nrows(), symbols[0],
                windows2.nrows(), symbols[1]
            );

            // Compare average latent codes between assets
            let mut avg_z1 = ndarray::Array1::zeros(best_model.config.latent_dim);
            for i in 0..windows.nrows() {
                let z = best_model.encode(&windows.row(i).to_owned());
                avg_z1 = avg_z1 + z;
            }
            avg_z1 /= windows.nrows() as f64;

            let mut avg_z2 = ndarray::Array1::zeros(best_model.config.latent_dim);
            for i in 0..windows2.nrows() {
                let z = best_model.encode(&windows2.row(i).to_owned());
                avg_z2 = avg_z2 + z;
            }
            avg_z2 /= windows2.nrows() as f64;

            println!("\n  Average latent codes:");
            println!("    {}: {:?}", symbols[0],
                avg_z1.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>()
            );
            println!("    {}: {:?}", symbols[1],
                avg_z2.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>()
            );

            let diff = &avg_z1 - &avg_z2;
            println!("    Difference: {:?}",
                diff.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n=== Summary ===\n");
    println!("  Beta=1:  Standard VAE, best reconstruction, entangled factors");
    println!("  Beta=4:  Good balance of reconstruction and disentanglement");
    println!("  Beta=10: Strongest disentanglement, higher reconstruction error");
    println!("\n  For trading, beta=4 typically provides the best interpretable");
    println!("  factors while maintaining sufficient reconstruction quality");
    println!("  for meaningful generation and scenario analysis.");
    println!("\n=== Done ===");

    Ok(())
}

/// Generate synthetic candle data for testing when API is unavailable.
fn generate_synthetic_candles(count: usize) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(count);
    let mut price = 50000.0_f64;

    for i in 0..count {
        let change = rng.gen_range(-0.02..0.02);
        let volatility = rng.gen_range(0.005..0.02);

        let open = price;
        let close = price * (1.0 + change);
        let high = open.max(close) * (1.0 + volatility);
        let low = open.min(close) * (1.0 - volatility);
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp: 1700000000 + (i as u64 * 3600),
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}

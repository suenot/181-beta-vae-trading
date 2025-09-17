# Chapter 234: Beta-VAE Trading

## 1. Introduction

Variational Autoencoders (VAEs) have become a foundational tool in generative modeling, but their standard formulation often produces entangled latent representations where individual dimensions lack clear interpretable meaning. In financial markets, where understanding the driving factors behind price movements is as important as prediction itself, this opacity is a serious limitation.

Beta-VAE addresses this by introducing a single hyperparameter, beta (written as the Greek letter in mathematical notation), that controls the degree of disentanglement in the learned latent space. By increasing beta beyond 1, the model is encouraged to find representations where each latent dimension captures an independent, interpretable factor of variation. When applied to trading, this means we can learn latent spaces where individual dimensions correspond to recognizable market forces such as trend direction, volatility regime, trading volume dynamics, or momentum signals.

The key insight of beta-VAE is elegantly simple: by strengthening the pressure on the latent distribution to match an isotropic Gaussian prior, we force the model to use each dimension efficiently and independently. This pressure, controlled by beta, creates a tunable tradeoff between reconstruction fidelity and representation quality. For trading applications, this tradeoff translates directly into a choice between capturing every nuance of market microstructure versus obtaining clean, actionable factor decompositions.

This chapter explores how beta-VAE can be deployed as a factor discovery and generation tool for systematic trading. We will develop the mathematical foundations, discuss practical considerations for financial data, implement a complete system in Rust, and demonstrate its application to cryptocurrency markets using live Bybit exchange data.

## 2. Mathematical Foundation

### The Standard VAE Objective

A standard VAE optimizes the Evidence Lower Bound (ELBO):

```
L_VAE = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
```

The first term is the reconstruction likelihood, encouraging the decoder to faithfully reproduce the input. The second term is the Kullback-Leibler divergence, regularizing the approximate posterior q(z|x) toward the prior p(z), typically a standard normal N(0, I).

### The Beta-VAE Modification

Beta-VAE modifies this objective by introducing a weighting factor on the KL term:

```
L_beta-VAE = E_q(z|x)[log p(x|z)] - beta * KL(q(z|x) || p(z))
```

When beta = 1, we recover the standard VAE. When beta > 1, the model faces stronger pressure to match the prior, which encourages statistical independence between latent dimensions. When beta < 1, the model relaxes the prior constraint, allowing richer but potentially entangled representations.

### Information Bottleneck Perspective

The beta parameter can be understood through the lens of information theory. The KL divergence term bounds the mutual information between the input x and the latent code z:

```
I(x; z) <= KL(q(z|x) || p(z))
```

By increasing beta, we tighten this information bottleneck, forcing the model to transmit only the most essential information through the latent channel. Each dimension must justify its information capacity, naturally leading to a decomposition where each dimension carries a distinct, non-redundant signal.

### Total Correlation Decomposition

The KL divergence in the VAE objective can be decomposed into three terms:

```
KL(q(z|x) || p(z)) = I(x; z) + KL(q(z) || prod_j q(z_j)) + sum_j KL(q(z_j) || p(z_j))
```

The middle term, known as the total correlation (TC), measures the statistical dependence between latent dimensions. Beta-VAE's effectiveness comes partly from its implicit penalization of total correlation. Variants like beta-TC-VAE explicitly target this term, but the original beta-VAE formulation remains practical and effective, especially with careful beta selection.

### Reparameterization and Gradient Estimation

The encoder outputs parameters mu and log_var for a Gaussian posterior. Sampling uses the reparameterization trick:

```
z = mu + exp(0.5 * log_var) * epsilon,  where epsilon ~ N(0, I)
```

This allows gradients to flow through the sampling operation, enabling end-to-end training. The KL divergence for Gaussian posteriors has a closed-form solution:

```
KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
```

## 3. Disentanglement for Finance

### Why Disentanglement Matters in Trading

Financial markets are driven by multiple overlapping factors: macroeconomic trends, sector rotations, volatility regimes, liquidity conditions, and momentum effects. Standard dimensionality reduction methods like PCA provide orthogonal factors but lack the nonlinear expressiveness needed to capture complex market dynamics. Standard VAEs can capture these dynamics but produce latent spaces where factors are hopelessly entangled.

Beta-VAE offers a middle path: nonlinear factor discovery with interpretability. Each latent dimension can be probed independently through latent traversals, where we vary one dimension while holding others fixed and observe the effect on reconstructed market data.

### Interpretable Market Factors

In practice, beta-VAE applied to OHLCV (Open, High, Low, Close, Volume) financial data tends to discover latent dimensions that align with recognizable market concepts:

- **Trend dimension**: Varying this dimension shifts the overall direction of price movement, from bearish through neutral to bullish.
- **Volatility dimension**: Controls the spread between high and low prices, capturing regime shifts between calm and turbulent markets.
- **Volume dimension**: Captures trading activity levels, distinguishing between high-conviction moves and low-participation drift.
- **Momentum dimension**: Encodes the persistence or mean-reversion tendency of recent price action.
- **Spread/range dimension**: Controls the typical candle body size relative to the overall range.

The exact factors discovered depend on the data, the latent dimensionality, and critically, the value of beta. Higher beta values tend to produce cleaner separation but may miss subtle interactions between factors.

### Measuring Disentanglement

Several metrics quantify how well a representation is disentangled:

- **Mutual Information Gap (MIG)**: For each true generative factor, compute the mutual information with all latent dimensions. The gap between the top two is the MIG score. Higher is better.
- **DCI Disentanglement**: Uses a feature importance matrix from predicting true factors from latent codes. Measures both disentanglement (each code dimension predicts only one factor) and completeness (each factor is captured by only one code dimension).
- **Factor VAE metric**: Trains a simple classifier to identify which factor was varied in a traversal. Accuracy measures disentanglement quality.

In finance, where ground-truth factors are not available, we use proxy measures: correlation of latent dimensions with known indicators (RSI, ATR, OBV, moving averages), stability of traversal interpretations across time periods, and prediction performance of individual dimensions for specific outcomes.

## 4. Trading Applications

### Interpretable Factor Models

Beta-VAE provides a data-driven alternative to hand-crafted factor models. Rather than specifying factors a priori (value, momentum, quality, etc.), the model discovers factors from raw market data. With sufficient disentanglement, each factor can be analyzed, backtested, and traded independently.

A typical workflow involves training the beta-VAE on rolling windows of multi-asset OHLCV data, performing latent traversals to interpret each dimension, then constructing factor portfolios by going long assets with high values on a given dimension and short those with low values.

### Controlled Generation

The generative nature of beta-VAE enables scenario analysis. By manipulating specific latent dimensions, traders can generate synthetic market scenarios that vary along a single factor while holding others constant. This supports stress testing: "What happens to my portfolio if volatility doubles but trend direction remains unchanged?"

### Factor-Specific Hedging

Once factors are identified, hedging becomes more targeted. Instead of hedging against broad market moves, a trader can identify which latent factor poses the greatest risk to their portfolio and construct hedges that specifically neutralize that exposure. For example, if a portfolio is primarily exposed to the volatility factor dimension, the hedge should target volatility instruments rather than directional ones.

### Regime Detection

The latent space naturally segments into regions corresponding to different market regimes. Clustering in the disentangled latent space provides regime labels that are more stable and interpretable than those from raw feature clustering. Transitions between regimes can be monitored in real time by tracking the portfolio's position in latent space.

## 5. Beta Selection

### The Reconstruction-Disentanglement Tradeoff

The choice of beta fundamentally controls the tension between two objectives:

- **Low beta (near 1)**: Better reconstruction, richer representations, but entangled factors. Suitable when prediction accuracy matters more than interpretability.
- **High beta (e.g., 4-10)**: Stronger disentanglement, cleaner factors, but worse reconstruction. Suitable when understanding market drivers matters more than precise prediction.
- **Very high beta (>10)**: May cause posterior collapse, where the model ignores the input and produces near-prior outputs. Latent dimensions become meaningless.

### Annealing Strategies

Rather than fixing beta from the start, annealing strategies gradually increase beta during training:

- **Linear annealing**: beta increases linearly from 0 to the target value over a specified number of training steps. This allows the model to first learn a good reconstruction, then gradually impose disentanglement.
- **Cyclical annealing**: beta oscillates between 0 and the target value in cycles. This repeatedly allows the model to "breathe" and reorganize its latent space.
- **Monotonic with warmup**: beta stays at 0 for an initial warmup period, then ramps up. This ensures the encoder and decoder are well-initialized before disentanglement pressure is applied.

In practice, linear annealing with a warmup period of approximately 20% of total training steps works well for financial data. The target beta should be validated by examining traversal quality and downstream trading performance.

### Practical Beta Selection for Trading

A pragmatic approach to beta selection involves training models at several beta values (e.g., 1, 2, 4, 8, 16), computing both reconstruction error and disentanglement metrics, and selecting the beta at the "elbow" of the tradeoff curve. For trading applications, one should also evaluate the predictive power of individual latent dimensions for future returns, as the ultimate goal is actionable insight rather than abstract disentanglement.

## 6. Implementation Walkthrough

Our Rust implementation consists of several key components:

### Encoder Network

The encoder maps input market data (OHLCV features over a lookback window) to the parameters of a Gaussian posterior. We use a multi-layer architecture with configurable hidden sizes. The final layer splits into two heads: one for means (mu) and one for log-variances (log_var).

### Decoder Network

The decoder maps sampled latent codes back to reconstructed market data. It mirrors the encoder architecture but in reverse. The output layer uses a linear activation to allow reconstruction of arbitrary price values.

### Beta-Weighted ELBO

The loss function computes the standard reconstruction error (MSE for continuous financial data) and the KL divergence, applying the beta weighting to the KL term. During annealing, beta varies according to the selected schedule.

### Latent Traversals

To interpret learned factors, we encode a reference input, then systematically vary each latent dimension from -3 to +3 standard deviations while holding others fixed. The decoded outputs reveal how each dimension influences the reconstructed market data.

### Bybit Integration

The implementation includes a complete Bybit API client for fetching historical kline (candlestick) data. This provides real market data for training and evaluation without requiring external data pipelines.

See `rust/src/lib.rs` for the complete implementation and `rust/examples/trading_example.rs` for a working demonstration.

## 7. Bybit Data Integration

The Bybit exchange provides a comprehensive REST API for accessing historical market data. Our implementation uses the public kline endpoint to fetch OHLCV data at configurable intervals.

### Data Pipeline

1. **Fetch**: Request kline data for specified symbols (e.g., BTCUSDT, ETHUSDT) and timeframes (1m, 5m, 1h, 1d).
2. **Parse**: Deserialize the JSON response into structured candle data with open, high, low, close, and volume fields.
3. **Normalize**: Scale features to [0, 1] range using rolling min-max normalization to maintain temporal validity.
4. **Window**: Create overlapping windows of consecutive candles as input samples for the beta-VAE.

### API Considerations

- Bybit's public API has rate limits; the implementation includes appropriate delays between requests.
- Historical data may have gaps; the pipeline handles missing candles gracefully.
- Data is fetched in reverse chronological order and reversed for temporal consistency.

## 8. Key Takeaways

1. **Beta-VAE extends standard VAEs** by introducing a single hyperparameter, beta, that controls the tradeoff between reconstruction quality and latent space disentanglement.

2. **Disentangled representations are valuable for trading** because they provide interpretable, independently controllable market factors without requiring manual factor engineering.

3. **The mathematical foundation** centers on strengthening the information bottleneck in the VAE objective, which implicitly penalizes statistical dependence (total correlation) between latent dimensions.

4. **Practical trading applications** include interpretable factor models, controlled scenario generation, factor-specific hedging, and regime detection.

5. **Beta selection requires balancing** reconstruction fidelity against factor interpretability. Annealing strategies help the model converge to better solutions than fixed-beta training.

6. **Implementation in Rust** provides the performance characteristics necessary for processing large volumes of financial data and enables integration into low-latency trading systems.

7. **Disentanglement metrics** such as MIG and DCI provide quantitative assessment of representation quality, though in finance these should be supplemented with domain-specific evaluations like factor predictive power and traversal interpretability.

8. **Live market data from Bybit** enables direct application to cryptocurrency trading, where 24/7 markets and high volatility provide rich training signals for factor discovery.

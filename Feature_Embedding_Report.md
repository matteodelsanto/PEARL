Feature Embedding from Token-wise Sequences

(Surprisal, Entropy, ΔSurprisal, ΔEntropy)

⸻

Note on the {p} notation

In what follows, {p} denotes a feature prefix identifying the source sequence from which the embedding is computed.

In practice, {p} is replaced by strings such as:
	•	S_HC → surprisal computed with the healthy-control language model
	•	S_AD → surprisal computed with the patient (AD) language model
	•	S_DELTA → difference S_HC − S_AD
	•	H_HC → token-wise entropy (healthy-control model)
	•	H_AD → token-wise entropy (patient model)
	•	H_DELTA → difference H_HC − H_AD

Examples:
	•	{p}_mean → S_HC_mean, H_DELTA_mean, …
	•	{p}_hist_03 → S_AD_hist_03, …

This naming scheme allows one to:
	•	reuse the same feature definitions across multiple model views (HC / AD / Δ)
	•	keep feature names consistent and interpretable
	•	concatenate multiple embeddings within a single classifier

⸻

General context

Given a tokenized text:

W = (w_1, w_2, \dots, w_N)

we derive a token-wise numeric sequence:

X = (x_2, x_3, \dots, x_N)

where x_i may represent:
	•	surprisal
	•	entropy
	•	the difference between two models (Δ)

The goal is to transform X (variable length) into a fixed-dimensional feature vector suitable for classification or regression.

⸻

1. Global Summary Pooling

These features describe the global behavior of the sequence, ignoring temporal order.

Features

Feature	Description	Computation
{p}_n	Sequence length	N-1
{p}_mean	Mean value	\frac{1}{N-1}\sum_i x_i
{p}_std	Sample standard deviation	\sqrt{\frac{1}{N-2}\sum (x_i-\mu)^2}
{p}_min	Minimum value	\min_i x_i
{p}_max	Maximum value	\max_i x_i
{p}_q10	10th percentile	Q_{0.10}(X)
{p}_q25	25th percentile	Q_{0.25}(X)
{p}_q50	Median	Q_{0.50}(X)
{p}_q75	75th percentile	Q_{0.75}(X)
{p}_q90	90th percentile	Q_{0.90}(X)
{p}_q95	95th percentile	Q_{0.95}(X)

Interpretation
These features capture overall level, dispersion, and tail behavior.

⸻

2. Temporal Dynamics Features

These features characterize temporal evolution without using explicit sequential models.

Feature	Description	Computation
{p}_slope	Global temporal trend	linear regression coefficient of x_i over index i
{p}_autocorr_lag1	Lag-1 autocorrelation	\text{corr}(x_i, x_{i+1})

Interpretation
	•	positive slope → progressive increase in unpredictability
	•	low autocorrelation → local instability

⸻

3. Burstiness Features

These features quantify local, rare events of high surprisal or entropy.

Feature	Description	Computation
{p}_burst_fraction_above	Fraction of anomalous values	$begin:math:text$
{p}_burst_max_run_above	Maximum consecutive burst	longest run above threshold

(with z = 2 by default)

Interpretation
Captures local predictive breakdowns not visible in global averages or perplexity.

⸻

4. Window-based Pooling

The sequence is split into K contiguous temporal windows (typically K = 5).

For each window w:

Feature	Description
{p}_w{w}_n	Number of tokens in the window
{p}_w{w}_mean	Mean value
{p}_w{w}_std	Standard deviation
{p}_w{w}_q50	Median
{p}_w{w}_q90	90th percentile

Interpretation
Allows differentiation between early, middle, and late portions of the discourse, capturing fatigue or progressive deterioration effects.

⸻

5. Histogram / Distribution Embedding

The sequence is represented as a normalized distribution over B bins.

Feature	Description
{p}_hist_00 … {p}_hist_{B-1}	Normalized bin frequencies

Computation
	1.	define a global value range [a, b] from the training set
	2.	divide the range into B intervals
	3.	count and normalize:

h_j = \frac{|\{x_i \in \text{bin}_j\}|}{N-1}

Interpretation
Captures the full distributional shape (spread, tails, multimodality), even when mean and variance are similar.

⸻

6. Differential (Δ) Features

Given two models (HC and AD):

\Delta x_i = x_i^{HC} - x_i^{AD}

All features described above are also computed on ΔX.

Advantages
	•	reduction of non-informative variability
	•	enhanced sensitivity to model-specific cognitive differences

⸻

7. Relation to Perplexity

Perplexity (PPL) is a single global feature:

\text{PPL} = \exp\left(\frac{1}{N-1}\sum_i S_i\right)

It can be used jointly with:
	•	{p}_mean (mean surprisal)
	•	{p}_std, burstiness, and histogram features

which capture information that is collapsed and lost when using perplexity alone.
# ctabpfn-a-v3
Conformalized TabPFN, additive variant: splits the train part 90/10 at the seed, reads the model's native quantile grid once through the shared content-addressed cache, and shifts the native `alpha/2` and `1-alpha/2` quantiles outward by the exact conformal order statistic of the CQR score, a margin that may be negative and tighten the interval. Writes the native median as `center`.
- paper: none, novel to this project; the additive score coincides with van Leeuwen's conformalized TabPFN note, https://openreview.net/forum?id=TnZcC7GXI5
- source: ctabpfn pinned to git commit `48dbc27` (tag v0.1.0), tabpfn 8.5.0 with v3 weights
- fast: ignored by the calibration; disables the shared inference cache so tests recompute
- knobs: `TABPFN_DEVICE` selects the inference device, default `cpu`; `CTABPFN_CACHE` moves the shared inference cache or, set empty, disables it
- threading: one TabPFN inference over torch's machine-default thread pool; phase workers two to four; the trio shares cached inferences, so run this phase first and the other two ride the cache
- deviations: none, the ctabpfn package is the reference implementation

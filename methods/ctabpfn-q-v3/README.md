# ctabpfn-q-v3
Conformalized TabPFN, quantile-level variant: splits the train part 90/10 at the seed, reads the model's native quantile grid once through the shared content-addressed cache, and calibrates the quantile level itself, picking the deepest symmetric grid level whose intervals miss at most the allowed share of calibration points, distributional conformal prediction on the model's own CDF. Writes the native median as `center`.
- paper: none, novel to this project; the score follows distributional conformal prediction, https://arxiv.org/abs/1909.07889
- source: ctabpfn pinned to git commit `48dbc27` (tag v0.1.0), tabpfn 8.5.0 with v3 weights
- fast: ignored by the calibration; disables the shared inference cache so tests recompute
- knobs: `TABPFN_DEVICE` selects the inference device, default `cpu`; `CTABPFN_CACHE` moves the shared inference cache or, set empty, disables it
- threading: one TabPFN inference over torch's machine-default thread pool, all cache hits after a ctabpfn-a-v3 phase; phase workers two to four
- deviations: none, the ctabpfn package is the reference implementation

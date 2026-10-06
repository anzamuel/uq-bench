# ctabpfn-m-v3
Conformalized TabPFN, multiplicative variant: splits the train part 90/10 at the seed, reads the model's native quantile grid once through the shared content-addressed cache, and scales the two half widths around the native median by the exact conformal order statistic of the ratio score, so the correction follows the native local width and may shrink it. Writes the native median as `center`.
- paper: none, novel to this project
- source: ctabpfn pinned to git commit `48dbc27` (tag v0.1.0), tabpfn 8.5.0 with v3 weights
- fast: ignored by the calibration; disables the shared inference cache so tests recompute
- knobs: `TABPFN_DEVICE` selects the inference device, default `cpu`; `CTABPFN_CACHE` moves the shared inference cache or, set empty, disables it
- threading: one TabPFN inference over torch's machine-default thread pool, all cache hits after a ctabpfn-a-v3 phase; phase workers two to four
- deviations: none, the ctabpfn package is the reference implementation

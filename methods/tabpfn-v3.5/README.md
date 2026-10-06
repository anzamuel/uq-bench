# tabpfn-v3.5
Fits the pretrained TabPFN foundation model and reads the interval directly off its native predictive-distribution quantiles at `alpha/2` and `1-alpha/2`, with no conformal layer, so coverage reflects the model's own calibration. Writes the native median as `center`.
- paper: https://www.nature.com/articles/s41586-024-08328-7
- source: tabpfn 9.0.0 pinned, v3.5 weights, the library default
- fast: ignored
- knobs: `TABPFN_DEVICE` selects the inference device, default `cpu` for reproducibility
- threading: one inference over torch's machine-default thread pool; phase workers two to four
- deviations: none

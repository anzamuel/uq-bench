# Method contract
Each directory here is one benchmark method: a standalone `uv` project with its own `pyproject.toml`, `uv.lock`, a `main.py` entry point, and a `README.md` in the format below. The runner invokes it as `uv run --directory <method> main.py <split> <preds> <coverage> <seed>` in a clean environment, so a method never shares dependencies with the harness or with other methods.

## Arguments
1. `split`: path to an `npz` file with `X_train`, `y_train`, and `X_test`.
2. `preds`: path where the method writes its `npz` output.
3. `coverage`: target coverage in `(0, 1)`, e.g. `0.9`.
4. `seed`: integer seeding everything random, including the method's own calibration split.

## Output
The method writes `preds` as an `npz` with one row per test point: `lower` and `upper` interval bounds, and optionally `center` with its point predictions. When `center` is present the harness anchors the NCIW rescaling on it instead of the interval midpoint.

## Determinism
Same seed, same bytes: two runs with identical arguments must write identical arrays, and a different seed must change them. `tests/test_determinism.py` discovers every method here and enforces this, so never draw from unseeded global state and avoid parallel reductions at predict time, their summation order is nondeterministic.

## Environment
`UQ_BENCH_FAST` is the one universal knob: set to any non-empty value it requests the method's own cheap settings, used by the test suite. Each method defines what it maps to, or ignores it, in its README; method-specific knobs always take precedence over it.

## Per-method README
Every method's `README.md` follows this fixed format so methods stay comparable at a glance. A `# <name>` title matching the directory name, a description of at most three sentences covering what the method does, how it forms its intervals, and what it wraps, then exactly five label lines in this order:
- `paper:` link to the implemented paper, or `none`.
- `source:` dependency provenance, a pinned package, an editable checkout path, or `original`.
- `fast:` what `UQ_BENCH_FAST` changes, or `ignored`.
- `knobs:` method-specific env vars with effect and default, semicolon separated, or `none`.
- `deviations:` deliberate differences from the reference implementation, semicolon separated, or `none`.

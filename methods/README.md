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
Same seed, same machine, same environment, same bytes: two runs with identical arguments must write identical arrays, and a different seed must change them. Bit-identity across machines is out of scope, different BLAS backends sum floats differently. `tests/test_determinism.py` discovers every method here and enforces the per-machine contract, so never draw from unseeded global state and avoid parallel reductions whose summation order depends on scheduling.

## Scheduling
The grid runs one method at a time, one `run_grid` call per method, so each phase has a homogeneous resource shape. The guiding rule is threads times workers equals cores: methods whose cells are single threaded run at `workers` equal to the CPU count, the machine-derived default, while methods with internal pools run at fewer workers sized so the products match. Worker counts and pool sizes are pure scheduling and never affect results; thread counts inside numeric kernels can affect bytes and are therefore fixed constants inside each method. Each method states its shape under `threading` in its README. The root `benchmark.py` encodes the paper grid's phases with these worker sizes. Methods may share state across directories only through content-addressed caches, like the ctabpfn trio's inference cache; order such phases so the first fills what the rest reuse.

## Environment
`UQ_BENCH_FAST` is the one universal knob: set to any non-empty value it requests the method's own cheap settings, used by the test suite. Each method defines what it maps to, or ignores it, in its README; method-specific knobs always take precedence over it.

## Per-method README
Every method's `README.md` follows this fixed format so methods stay comparable at a glance. A `# <name>` title matching the directory name, a description of at most three sentences covering what the method does, how it forms its intervals, and what it wraps, then exactly six label lines in this order:
- `paper:` link to the implemented paper, or `none`.
- `source:` dependency provenance, a pinned package, an editable checkout path, or `original`.
- `fast:` what `UQ_BENCH_FAST` changes, or `ignored`.
- `knobs:` method-specific env vars with effect and default, semicolon separated, or `none`.
- `threading:` the cell's internal thread and process usage plus the recommended phase workers.
- `deviations:` deliberate differences from the reference implementation, semicolon separated, or `none`.

## Versioning
A method that wraps a versioned model carries the model generation in its directory name, `tabpfn-v3`, `ctabpfn-q-v3.5`, and pins that generation inside its own environment, for the ctabpfn family through one pinned git commit of the `ctabpfn` package, which in turn pins one tabpfn release and one set of weights. A new model generation is a new directory with new pins, existing directories never move, so every row in `results.csv` stays reproducible under the method name it was produced with.

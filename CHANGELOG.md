# 3.0.0
- Breaking change: API of qbench module now uses simplified pass-through arguments to configure
    benchmarking. Now it's possible to just write
    `qb.benchmark(funcs, iters=50, reps=12, alpha=0.01, title="My benchmark")`
    to run a benchmark of `funcs` tuple while directly passing `iters=50, reps=12` to `qb.bench()`
    function, and passing `alpha=0.01` to `compareStats()` with `title="My benchmark"` to
    `renderComparisonResults()` through `qb.showBench()`
- `renderComparisonResults()` got parameter `show_percent_diff:bool = True` to calculate percentage
    difference between the two sets. `--percent_diff` and `no-percent_diff` are corresponding CLI
    options.
- slightly tweaked alternating row color for dark theme

# 2.0.0
- Backward compatibility breaking renaming of fields in data types of `compare` module.
- `compare.compareStats()` received an `alt_delimiter` parameter and a whole different mode
of pooling benchmarks to compare. This mode is not supported by CLI yet.
- A new module `qbench` is introduced for quick and simple benchmarking of Python callables.

# 1.1.0
- Add `--metric_precision` CLI argument.
- Ensure benchmark name column isn't wrapped (doesn't seem possible for all columns without risk of
cropping, though)

# 1.0.3
Tiny insignificant patch to ensure Python 3.10 compatibility (3.10.16 was tested)

# 1.0.1 - .2
No code changes, version bump due to minor updates to the docs

# 1.0.0
initial public release
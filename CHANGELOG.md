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
# Statistical Testing for Benchmark Results Comparison

`benchstats` is a Python 3.10+ package that lets you compare two sets of benchmark results<sup>*</sup> using proper statistical tests and make a readable report on that. This lets remove guesswork and WAGs from interpretation of benchmarking results, and obtain a solid<sup>**</sup> answer to a question "is the algorithm implementation A is faster than the algorithm implementation B".

Code that read input data, perform statistical testing and then visualize results is coupled only by a shared data types and it could easily be used separately if needed (for example, to build a performance regression testing pipeline).

<sup>*</sup> - while currently `benchstats` has a built-in parser only for [Google Benchmark](https://github.com/google/benchmark)-produced `.json` files, it's very simple to make and use a parser for any other data source as long as you know how to interpret it. See a "Custom data sources" paragraph and `--files_parser` CLI argument below.

<sup>**</sup> - the solidness of the answer still depends heavily on how well the underlying hardware and OS are quiesced, and how inherently non-deterministic the algorithms under benchmark are. It isn't covered here how to achieve these benchmarking basics, however, `benchstats` does have a mode to alleviate testing if the benchmarking machine is stable enough, see `--expect_same` CLI argument below.

## Rationale: why should you even bother to use statistical tests to compare benchmarking results?

\- *Really, why bother? Can't you just measure and compare the run time of code variant A to the run time of code variant B?*

Well, you can, but to see that this is meaningless, just run your benchmark several times. The numbers will never be the same, they'll fluctuate. Which particular number should you use for the comparison then?

\- *Ok, then why not take an average number over a certain amount of benchmark executions and compare these averages?*

If you repeat that experiment several times, you'll see this suffers from the very same problem as above - the number you get still fluctuates, even though usually it fluctuates less. You'll obtain two such numbers for benchmark alternatives and one will be bigger then the other. How would you know that the difference is large enough to be attributed to real differences in the code instead of a random non-determinism?

\- *Ah, but that's simple! In addition to the average value, also compute a standard deviation and then if averages are within 1-2-whatever standard deviations apart one from the other, declare the results aren't different. Otherwise one is significantly less or greater than the other. If you want to be extra sure, go with a "6 sigma golden standard" just like they do it in Physics! Google Benchmark even computes standard deviation on its own when you run benchmark with repetitions! Easy!*

... and even more wrong than you'd be, should you just compare two random measurements directly. [Standard deviation](https://en.wikipedia.org/wiki/Standard_deviation) is a measure of [variance](https://en.wikipedia.org/wiki/Variance) applicable (read: that make sense to apply) only to random variables that are instances of [Normal/Gaussian distribution](https://en.wikipedia.org/wiki/Normal_distribution). Measurements you get when benchmarking are anything but normally distributed, and a value computed as their "standard deviation" just doesn't describe what is expected - a real variance of distribution of measurements.

None of above is a good idea. But what is? Let's step back and think what do we want to learn from comparing benchmark results? We want to learn if a code variant A is faster, i.e. have smaller run time, than a code variant B. But we can't just measure their run times and compare them directly, because each measurement is essentially a random variable. Even if algorithms are perfectly deterministic, the hardware inherently isn't, so instead of measuring a run time precisely, we always sample a random value from some unknown distribution instead. Hence, the only correct way to go is to sample a lot of these random variables for distributions A and B, and then compare these sets of samples against each other. If it's more likely to find a smaller value in the set A than in the set B, then A is faster than B.

But how to do that? It's a well known problem in Mathematics and several solutions exists, such as [Mann–Whitney U test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test) and a newer more generic [Brunner Munzel Test](https://en.wikipedia.org/wiki/Brunner_Munzel_Test). `benchstats` is essentially a convenience wrapper around these tests with fully customizable inputs and results visualization/exporting, letting you get the answers effortlessly.

## Installation

```bash
pip install benchstats
```

For command line use see section "Examples" and "Command Line Interface Reference" below. For API overview see a corresponding section below.

## Examples of Benchmark Comparison

*(I'll use a Google Benchmark-derived binary as a data source for the examples, but just reminding that any data source could be easily plugged-in and used instead. For conciseness, I'll shorten "Google Benchmark" to simply "GBench")*

Note that in all cases, you should run at least 10 repetitions of a benchmark (`--benchmark_repetitions=10`). Even 2 repetitions would work, but the more data you have, the more reliable results are, and at least 10 is a reasonable minimum. `--benchmark_enable_random_interleaving` flag is highly recommended too.

Lets consider two use-cases to illustrate how to use `benchstats`.

### How to compare two different implementations of the same benchmark?

For example, you already have a benchmark in the `main` code branch and want to ensure that a `feature` branch, which modifies some underlying code, doesn't contain unexpected performance regressions. How to do that?

1. Run the benchmark from the `main` branch with `--benchmark_out=./my_main.json` argument to save results to `./my_main.json` file:

```bash
benchmark_binary --benchmark_repetitions=10 \
    --benchmark_enable_random_interleaving \
    --benchmark_out=./my_main.json
```

2. Checkout the `feature` code branch and similarly run the same benchmark saving results to `./my_feature.json` file.

3. Let the `benchstats` analyze the results:

```bash
benchstats ./my_main.json ./my_feature.json real_time
```

The first two positional arguments of `benchstats` module are just file paths and the third argument is a json field name that contain a metric value for comparison. (`real_time` is the default field/metric, so it doesn't even have to be specified, btw). GBench typically saves two metrics: `real_time` (shown as just `Time` in reports) which is an average time of one iteration, and a `cpu_time` (shown as `CPU` in reports) which is an average time CPU spent executing the user-mode code. GBench also supports custom timers, and you can use them here as well. Typically, custom timers have the same field name as the name that was given to a counter (in case of doubts, just peek into a `.json` file).

If the `benchmark_binary` had only a single benchmark, the result of the command could look something like this:

<img src="https://github.com/Arech/benchstats/blob/main/docs/imgs/simple1.svg?raw=true" alt="Sample benchmark comparison results for a single benchmark" width=600>

`benchstats` present results as a table: the first column correspond to a benchmark name and all other columns shows results of applying a statistical test to a corresponding metric (yes, you can supply many metrics at once and all of them will be analyzed. Try to append `cpu_time` to the command line above).

Here's what a data cell of the `real_time` column mean: there were 10 samples of an alternative A (from the first positional argument - `my_main.json`) and 10 samples of an alternative B (from the second positional argument `my_feature.json`), - hence this `(10 vs 10)` text in the end. Mean value (hence `(means)` in the column title) for alternative A over its 10 samples was 516.5 microseconds, while mean for B was 502.8 microseconds. The test was using significance level 0.001 (hence the string `alpha=0.00100` in the table title) and on that level test declared that both sets aren't different from each other (hence symbol `~` between the mean values). Actual [p-value](https://en.wikipedia.org/wiki/P-value) for tests that didn't find a difference is not shown by default, only blank space is allocated to report it (so no text shift happens when some benchmark has differences and others don't) but you can pass `--always_show_pvalues` flag to show it.

To show how `benchstats` report could look like when a significant difference is found, I'll just rerun the same analysis with less restrictive significance level (and will use the opportunity to show effects of some above-mentioned flags)

```bash
benchstats ./my_main.json ./my_feature.json real_time cpu_time \
    --always_show_pvalues --alpha=0.03
```

![Benchmark comparison results when a difference is detected](https://github.com/Arech/benchstats/blob/main/docs/imgs/simple2.svg?raw=true)

The main change is that now the test reports that according to `real_time` metric, the benchmark that generated `./my_main.json` run significantly slower than the benchmark that made `./my_feature.json`. Due to that the script has exited with exit code 1, letting one to detect that a human intervention might be needed. But what are these "main metrics" that the warning string refers to? Main metrics are such metrics, any difference in which makes the script to exit with error code 1. By default, a metric mentioned the first in the command line list of metrics is the main metric, but this could be changed with `--main_metrics` argument, which accepts a list of indices in the metrics list. Let's make `cpu_time` the main metric and see what would change:

```bash
benchstats ./my_main.json ./my_feature.json real_time cpu_time \
    --always_show_pvalues --alpha=0.03 --main_metrics 1
```

![Benchmark comparison results when a difference in non-main metric is detected](https://github.com/Arech/benchstats/blob/main/docs/imgs/simple3.svg?raw=true)

Now `cpu_time` is the main metric and the difference in it isn't significant. `real_time` now is a secondary metric, which is only reported, but doesn't influence the script exit code.

### How to compare two different benchmarks against each other?

For example, we are optimizing a function and want to compare how a new implementation compares to the old. We've created an individual benchmark for both implementations and for convenience have placed them in the same binary. That implies the benchmark functions must have different names. Now, if we run the benchmark binary and obtain `./my.json` results file and then will follow the approach above blindly, we won't get what we need. `benchstats` compares a benchmark from the source 1 to a benchmark with the same name in the source 2, so we'll end up comparing old vs old in one row, and new vs new in the other, just like this:

```bash
benchstats ./my.json ./my.json real_time
```
<img src="https://github.com/Arech/benchstats/blob/main/docs/imgs/simple4_bad.svg?raw=true" alt="Wrong benchmark comparison" width=700>

To solve that, `benchstats` has two features based on the Python standard [regular expression](https://docs.python.org/3.10/library/re.html#regular-expression-syntax) engine:

1. filtering benchmarks by name when reading a source file using a regexp in `--filter1` or `--filter2` flags correspondingly.
2. a benchmark name rewrite feature, based on regexp search and replace. `--from` (also `--from1` for source 1 only and `--from2` for source 2 only) flag sets a regular expression pattern to find in each benchmark name, and an optional `--to` (also `--to1` and `--to2`) flag sets replacement string to substitute the pattern found.

This allows us to make a plan:

1. source 1 will represent the new implementation. For that we'll read `./my.json` and take only benchmark names that doesn't have `Old` substring using a regexp `^(?!.*?Old).*$`
2. source 2 will represent the old implementation. We'll read `./my.json` and take only benchmarks with names that contain `Old` substring. The regexp is trivial literal string `Old`.
3. For source 2 we'll rewrite benchmark names to remove the `Old` prefix

```bash
benchstats ./my.json ./my.json real_time \
    --filter1='^(?!.*?Old).*$' --filter2=Old \
    --from2='(?:Old)(.*?)' --to2=\\1
```

Note the single `'` quotes around `--filter1` and `--from2` regexps. They contain special characters that bash will try interpret spawning an error, if we won't escape them. Single quotes prevent bash expansion (double quotes `"` don't!) and pass strings as is to the program. Also note that string replacement pattern for `--to2` has a back slash escaped for that very reason too: `\\1` instead of `\1`. Single quotes is an option here too.

Now we get a proper result:

<img src="https://github.com/Arech/benchstats/blob/main/docs/imgs/simple4.svg?raw=true" alt="Correct benchmark comparison when a single source file is used" width=700>

## Custom Data Sources

While `benchstats` today support one Google Benchmark -produced `.json` files out of the box, it's intended to be able to run statistical tests for any data sources. If you know how to parse your custom file format, connect to a remote database, generate data on the fly, or obtain data any other way, you can easily let `benchstats` to run statistical tests on it by writing a custom parser.

Any python class (a parser) named X, that:
- inherits from `benchstats.common.ParserBase` class
- can be instantiated with the following 4 parameters:
    1. `source_id` - a string identifier, describing a data source within parser's controlled namespace. It's is obtained as the first or the second positional argument of `benchstats` module CLI.
        - In the examples above, for the built-in `parser_GbenchJson` this is a source file name.
    2. `filter` - a string identifier of a filter, used to restrict a set of benchmarks to load. Obtained as a value of `--filter1` or `--filter2` CLI flags.
        - for the built-in `parser_GbenchJson` it's a string with regular expression to use for filtering benchmark names.
    3. `metrics` - is a non-empty list or a tuple of string identifiers, corresponding to metrics to read for benchmarks. Obtained from third and subsequent CLI positional arguments.
    4. `debug_log` - an instance of `benchstats.common.LoggingConsole` class that provides a
    console logging interface to report errors if any <small>(built-in parsers should also support `None`
    values and `bool` values and instantiate a `LoggingConsole` on its own, but support for this
    isn't needed for a custom parser)</small>
- has a `def getStats(self)` method returning a dictionary that maps strings describing benchmark names to dictionaries linking metric names with `np.ndarray()` of metric values.
- is saved to a file `X.py`

can be used as a custom data source for `benchstats` CLI. Just pass a path to the `X.py` as value of `--files_parser` argument (or `--file1_parser` or `--file2_parser` if you want to use different parsers for source1 and source2)

### The simplest example of a custom data source

Imagine you have a one-column CSV with data on which you want to run a statistical significance test. Save the following to `./myCSV.py`:

```python
import numpy as np
from benchstats.common import ParserBase

class myCSV(ParserBase):
    def __init__(self, fpath, filter, metrics, debug_log=None) -> None:
        self.stats = np.loadtxt(fpath, dtype=np.float64)

    def getStats(self) -> dict[str, dict[str, np.ndarray]]:
        return {"bm": {"real_time": self.stats}}
```

There's always only one benchmark name `bm` hardcoded, and only one metric `real_time` that is read from CSV, but you get the idea. It's that simple.

Now just run `benchstats` passing a path to the parser in `--files_parser` flag:

```bash
benchstats ./src1.csv ./src2.csv --files_parser ./myCSV.py
```

If you make a parser that might be usable by other people, please consider adding it to the project's built-in parsers set by opening a thread with a suggestion in https://github.com/Arech/benchstats/issues or by making a PR into the repo.

## Command Line Interface Reference

You can always get actual CLI help output by passing `--help` flag to the invocation of `benchstats`. Note that `--help` output has precedence over the description below.

```
usage: __main__.py [-h] [--show_debug | --no-show_debug]
    [--files_parser <files parser class or path>]
    [--file1_parser <file1 parser class or path>]
    [--filter1 <reg expr>]
    [--file2_parser <file2 parser class or path>]
    [--filter2 <reg expr>]
    [--from <reg expr>] [--to <replacement string>]
    [--from1 <reg expr>] [--to1 <replacement string>]
    [--from2 <reg expr>] [--to2 <replacement string>]
    [--main_metrics <metric idx> [<metric idx> ...]]
    [--method <method id>]
    [--alpha <positive float less than 0.5>]
    [--bonferroni]
    [--always_show_pvalues]
    [--sample_sizes | --no-sample_sizes]
    [--sample_stats <stat ids> [<stat ids> ...]]
    [--expect_same]
    [--colors | --no-colors]
    [--multiline | --no-multiline]
    [--export_to <path/to/export_file>]
    [--export_fmt <format id>]
    [--export_light]
    <path/to/file1> <path/to/file2> [metrics ...]
```
### Options
- `-h, --help` show this help message and exit
- `--show_debug, --no-show_debug` Shows some additional debugging info. Default: True

#### Input data:
Arguments describing how a source data sets are obtained and are optionally
transformed.

- `<path/to/file1>` Path to the first data file with benchmark results.
                    See also `--file1_parser` and `--filter1` arguments.
- `<path/to/file2>` Path to the second data file with benchmark results.
                    See also `--file2_parser` and `--filter2` arguments
- `--files_parser <files parser class or path>` Sets files parser class identifier, if a built-in parser is used (options are: `GbenchJson`). Or sets a path to `.py` file defining a custom parser, inherited from `benchstats.common.ParserBase` class. The parser class name must be the same as the file name.
- `--file1_parser <file1 parser class or path>` Same as `--files_parser`, but only applies to `file1`.
- `--filter1 <reg expr>` If specified, sets a Python regular expression to select benchmarks by name from `<file1>`
- `--file2_parser <file2 parser class or path>` Same as --files_parser, but only applies to `file2`.
- `--filter2 <reg expr>` Same as `--filter1`, but for `<file2>`
- `--from <reg expr>` Arguments `--from` and `--to` works in a pair and are used to describe benchmark names transformation. `--from` sets a regular expression (Python re module flavor) to define a pattern to replace to `--to` replacement string. Typical use-cases include removing unnecessary parts of benchmark names (for example, Google Benchmark adds some suffixes that might not convey useful information) or "glueing" results of different benchmarks so they become comparable by the tool (for example, you have two benchmarks, the first is 'old_foo' with old algorithm implementation and the other is 'foo' with a new competing algorithm implementation, - to compare their performance against each other, you need to remove 'old_' prefix from the first benchmark with `--from old_` and apply `--filter1` and `--filter2` to restrict loading only corresponding data for old or new variations). Remember to escape characters that have special meaning for the shell.
- `--to <replacement string>` See help for `--from`.
- `--from1 <reg expr>`  Like `--from`, but for `<file1>` only.
- `--to1 <replacement string>` Like `--to`, but for `<file1>` only
- `--from2 <reg expr>`  Like `--from`, but for `<file2>` only
- `--to2 <replacement string>` Like `--to`, but for `<file2>` only
- `metrics`             List of metric identifiers to use in tests for each benchmark. Default: `real_time`. For deterministic algorithms highly recommend to measure and to use a minimum latency per repetition.
- `--main_metrics <metric idx> [<metric idx> ...]` Indexes in `metrics` list specifying the main metrics. These are displayed differently and differences detected cause script to exit(1).
Default: `0`.

#### Statistical settings:

- `--method <method id>`  Selects a method of statistical testing. Possible values are are: `brunnermunzel` for [Brunner Munzel test](https://en.wikipedia.org/wiki/Brunner_Munzel_Test) and `mannwhitneyu` for [Mann-Whitney U test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test). Default is `brunnermunzel`
- `--alpha <positive float less than 0.5>` Set statistical significance level (a desired mistake probability level). Default `0.001`
    - Note! This is an actual probability of a mistake only when all preconditions of the chosen statistical test are met. One of the most important preconditions like independence of individual measurements, or constancy of underlying distribution parameters are almost never met in a real-life benchmarking, even on a properly quiesced hardware. Real mistake rates are higher. 
- `--bonferroni`          If set, applies a [Bonferroni multiple comparisons correction](https://en.wikipedia.org/wiki/Bonferroni_correction).

#### Rendering:
Controls how results are rendered

- `--always_show_pvalues` If set, always show pvalues. By default it shows pvalues only for significant differences. Note that when two sets are compared stochastically
same (`~`), or more precisely, not stochastically less and not stochastically greater (see [Stochastic ordering](https://en.wikipedia.org/wiki/Stochastic_ordering)),
pvalue shown is a minimum of two pvalues for less and greater comparison.
- `--sample_sizes, --no-sample_sizes` Controls whether to show sizes of datasets used in a test. Default `True`
- `--sample_stats <stat ids> [<stat ids> ...]` Sets which additional statistics about compared samples sets to show. Could be any sequence of floats (must be in range `[0,100]`, designating a
percentile value to calculate), or aliases `extremums`, `median`, `iqr` (interquartile range), `std` (standard deviation). Shortenings are accepted. Use of `std`, though isn't recommended as standard deviation doesn't really mean anything for most distributions beyond Normal.
- `--expect_same` If set, assumes that distributions are the same (i.e. `H0` hypothesis is true) and shows some additional statistics useful for ensuring that a
benchmark code is stable enough, or the machine is quiesced enough. One good example is when two different source `<file1>` and `<file2>` were made with exactly the same binary running on exactly the same machine in the same state - on a machine with a proper setup tests shouldn't find a difference.
- `--colors, --no-colors` Controls if the output should be colored. Default: `True`
- `--multiline, --no-multiline` Controls if benchmark results could span several lines. Default: `False`

#### Export:
Controls how to export results.

- `--export_to <path/to/export_file>` Path to file to store comparison results to.
- `--export_fmt <format id>` Format of the export file. Options are: `txt`, `svg`, `html`. If not set, inferred from `--export_to` file extension.
- `--export_light` If set, uses light theme instead of dark

## API overview

The code is pretty trivial, so I'll post only a high-level overview: there are 3 major components interacting in `benchstats` when it runs in CLI mode:
1. one or two source data parsers, depending on value of `--files_parser`, `file1_parser` or `file2_parser`.
    - built-in parsers are stored in files starting with `parser_` prefix. What goes after the prefix is a parser identifier for CLI `--file?_parser` flag.
    - all parsers obey requirements listed above in "Custom Data Sources" section.
    - essentially a parser could be just a single function returning `dict[str, dict[str, np.ndarray]]` mapping from benchmark names to metric names to metric values. It's a class only for the simplicity of writing actual code and validating user-supplied parsers.
2. a comparison method `compareStats()` from `benchstats.compare`. It:
    - takes two dictionaries `dict[str, dict[str, np.ndarray]]` with source data plus some control arguments, run stat tests on all metric values of the dictionaries that have the same benchmark name (key of the outer-most dictionary) and a metric name (key of the inner-most dictionary), and then
    - produces `benchstats.compare.CompareStatsResult` namedtuple object with comparison results for that benchmarks+metrics, common to both source data sets.
3. a rendering method `renderComparisonResults()` from `benchstats.render`:
    - takes `benchstats.compare.CompareStatsResult` namedtuple produced by `compareStats()`, plus various control parameters, and renders it according to the rules and styles.

## Stability and Changelog

The project uses semantic versioning scheme.

Depending on a feedback the project or its individual components might get breaking changes, so prefer to use version pinning to prevent unexpected breakages. See [CHANGELOG.md](https://github.com/Arech/benchstats/blob/main/CHANGELOG.md) for details.


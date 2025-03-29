import argparse
from .common import kAvailableFormats
from .compare import kMethods, kDefaultAlpha
from .parsers import getBuiltinParsers
from .render import kPossibleStatNames


def makeParser():
    parser = argparse.ArgumentParser(
        description="A tool to compare two sets of the same benchmarks with repetitions.\n"
        "Homepage: https://github.com/Arech/benchstats\n",
        epilog="On custom parsers:\n"
        "You could supply virtually any source data to the tool by utilizing --files_parser (or "
        "--file1_parser or --file2_parser) command line argument. Each of these arguments in "
        "addition to built-in parser identifiers also accepts a path to a Python file that "
        "defines a custom parser. The simplest possible parser to read a single one column CSV "
        "file is this:\n\n"
        """# save to ./myCSV.py
import numpy as np
from benchstats.common import ParserBase

class myCSV(ParserBase):
    def __init__(self, fpath, filter, metrics, debug_log=None) -> None:
        self.stats = np.loadtxt(fpath, dtype=np.float64)

    def getStats(self) -> dict[str, dict[str, np.ndarray]]:
        return {"bm": {"real_time": self.stats}}

"""
        "It doesn't support filtering, different benchmarks and the metric is hardcoded - but you "
        "get the idea. It's that simple.\n"
        "Now to compare two datasets in csvs, just run:\n"
        "python -m benchstats ./src1.csv ./src2.csv --files_parser ./myCSV.py\n\n"
        "If you'll make a parser that could be useful to other people, please consider adding it "
        "to the project's built-in parsers set by opening a thread with a suggestion in "
        "https://github.com/Arech/benchstats/issues or by making a PR into the repo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--show_debug",
        help="Shows some additional debugging info. Default: %(default)s",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    g_inputs = parser.add_argument_group(
        "Input data",
        "Arguments describing how a source data sets are obtained and are optionally transformed.",
    )

    g_inputs.add_argument(
        "file1",
        help="Path to the first data file with benchmark results. See also --file1_parser and "
        "--filter1 arguments",
        metavar="<path/to/file1>",
    )

    g_inputs.add_argument(
        "file2",
        help="Path to the second data file with benchmark results. See also --file2_parser and "
        "--filter2 arguments",
        metavar="<path/to/file2>",
    )

    g_inputs.add_argument(
        "--files_parser",
        help="Sets files parser class identifier, if a built-in parser is used (options are: "
        f"{', '.join(getBuiltinParsers())}). Or sets a path to .py file defining a custom parser, "
        "inherited from 'benchstats.common.ParserBase' class. The parser class name must be the "
        "same as the file name. See below for an example.",
        default="GbenchJson",
        metavar="<files parser class or path>",
    )

    g_inputs.add_argument(
        "--file1_parser",
        help="Same as --files_parser, but only applies to file1.",
        default=None,
        metavar="<file1 parser class or path>",
    )

    g_inputs.add_argument(
        "--filter1",
        help="If specified, sets a Python regular expression (see "
        "https://docs.python.org/3/howto/regex.html#regex-howto) to select benchmarks by name "
        "from <file1>",
        metavar="<reg expr>",
        default=None,
    )

    g_inputs.add_argument(
        "--file2_parser",
        help="Same as --files_parser, but only applies to file2.",
        default=None,
        metavar="<file2 parser class or path>",
    )

    g_inputs.add_argument(
        "--filter2",
        help="Same as --filter1, but for <file2>",
        metavar="<reg expr>",
        default=None,
    )

    g_inputs.add_argument(
        "--from",
        help="--from and --to works in a pair and are used to describe benchmark names transformation. "
        "--from sets a regular expression (Python re module flavor) to define a pattern to replace to --to replacement "
        "string. "
        "Typical use-cases include removing unnecessary parts of benchmark names (for example, Google Benchmark adds "
        "some suffixes that might not convey useful information) or \"glueing\" results of different benchmarks "
        "so they become comparable by the tool (for example, you have two benchmarks, the first is 'old_foo' with old algorithm "
        "implementation and the other is 'foo' with a new competing algorithm implementation, - to compare their performance "
        "against each other, you need to remove 'old_' prefix from the first benchmark with `--from old_`"
        " and apply --filter1 and --filter2 to restrict loading only corresponding data for old or new variations). "
        "Remember to escape characters that have special meaning for the shell.",
        metavar="<reg expr>",
        default=None,
    )

    g_inputs.add_argument(
        "--to",
        help="See help for --from.",
        metavar="<replacement string>",
        default=None,
    )

    g_inputs.add_argument(
        "--from1",
        help="Like --from, but for <file1> only",
        metavar="<reg expr>",
        default=None,
    )

    g_inputs.add_argument(
        "--to1",
        help="Like --to, but for <file1> only",
        metavar="<replacement string>",
        default=None,
    )

    g_inputs.add_argument(
        "--from2",
        help="Like --from, but for <file2> only",
        metavar="<reg expr>",
        default=None,
    )

    g_inputs.add_argument(
        "--to2",
        help="Like --to, but for <file2> only",
        metavar="<replacement string>",
        default=None,
    )

    g_inputs.add_argument(
        "metrics",
        help="List of metric identifiers to use in tests for each benchmark. Default: %(default)s. "
        "For deterministic algorithms highly recommend to measure and to use a minimum latency per "
        "repetition.",
        nargs="*",
        default=["real_time"],
    )

    g_inputs.add_argument(
        "--main_metrics",
        help="Indexes in 'metrics' list specifying the main metrics. These are displayed differently and differences "
        "detected cause script to exit(1). Default: %(default)s.",
        nargs="+",
        metavar="<metric idx>",
        type=int,
        default=[0],
    )

    g_stats = parser.add_argument_group("Statistical settings")

    g_stats.add_argument(
        "--method",
        help=(
            "Selects a method of statistical testing. Possible values are are: "
            + ", ".join(
                [
                    "'" + id + f"' for {descr['name']} ({descr['url'].replace('%','%%')})"
                    for (id, descr) in kMethods.items()
                ]
            )
            + ". Default is %(default)s"
        ),
        metavar="<method id>",
        choices=kMethods.keys(),
        default=list(kMethods.keys())[0],
    )

    g_stats.add_argument(
        "--alpha",
        help="Set statistical significance level (a desired mistake probability level)\n"
        "Note! This is an actual probability of a mistake only when all preconditions of the "
        "chosen statistical test are met. One of the most important preconditions like independence of "
        "individual measurements, or constancy of underlying distribution parameters are almost "
        "never met in a real-life benchmarking, even on a properly quiesced hardware. Real "
        "mistake rates are higher. Default %(default)s",
        metavar="<positive float less than 0.5>",
        type=float,
        default=kDefaultAlpha,
    )

    g_stats.add_argument(
        "--bonferroni",
        help="If set, applies a Bonferroni multiple comparisons correction "
        "(https://en.wikipedia.org/wiki/Bonferroni_correction).",
        action="store_true",
        default=False,
    )

    g_rendering=parser.add_argument_group("Rendering", "Controls how results are rendered")

    g_rendering.add_argument(
        "--always_show_pvalues",
        help="If set, always show pvalues. By default it shows pvalues only for significant differences. Note that "
        "when two sets are compared stochastically same (~), or more precisely, not stochastically less and not "
        "stochastically greater (see https://en.wikipedia.org/wiki/Stochastic_ordering), pvalue shown is a minimum of "
        "two pvalues for less and greater comparison.",
        action="store_true",
        default=False,
    )

    g_rendering.add_argument(
        "--sample_sizes",
        help="Controls whether to show sizes of datasets used in a test. Default %(default)s",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # py3.11 doesn't accept backslashes in f-strings, so preparing that in advance
    stat_options = "', '".join(kPossibleStatNames.keys())
    g_rendering.add_argument(
        "--sample_stats",
        help="Sets which additional statistics about compared samples sets to show. Could be any sequence of floats "
        "(must be in range [0,100], designating a percentile value to calculate), or aliases '"
        f"{stat_options}' (shortenings are accepted). Use of 'std', though isn't "
        "recommended as standard deviation doesn't really mean anything for most distributions "
        "beyond Normal.",
        nargs="+",  # default value doesn't work with *, while it must
        metavar="<stat ids>",
        default=None,
    )

    g_rendering.add_argument(
        "--expect_same",
        help="If set, assumes that distributions are the same (i.e. H0 hypothesis is true) and shows some additional "
        "statistics useful for ensuring that a benchmark code is stable enough, or the machine is quiesced enough. "
        "One good example is when <file1> and <file2> were made with exactly the same binary running on exactly the "
        "same machine in the same state - on a machine with a proper setup tests shouldn't find a difference.",
        action="store_true",
        default=False,
    )

    g_rendering.add_argument(
        "--colors",
        help="Controls if the output should be colored. Default: %(default)s",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    g_rendering.add_argument(
        "--multiline",
        help="Controls if benchmark results could span several lines. Default: %(default)s",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    g_export = parser.add_argument_group("Export","Controls how to export results.")

    g_export.add_argument(
        "--export_to",
        help="Path to file to store comparison results to.",
        metavar="<path/to/export_file>",
    )

    g_export.add_argument(
        "--export_fmt",
        help=f"Format of the export file. Options are: {', '.join(kAvailableFormats)}. If not set, "
        "inferred from --export_to file extension",
        choices=kAvailableFormats,
        default=None,
        metavar="<format id>",
    )

    g_export.add_argument(
        "--export_light",
        help="If set, uses light theme instead of dark",
        action="store_true",
        default=False,
    )

    return parser

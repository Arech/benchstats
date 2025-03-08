import argparse
from .compare import kMethods, kDefaultAlpha
from .parsers import getBuiltinParsers
from .render import kAvailableFormats


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
    def __init__(self, fpath, filter, metrics, debug_log=True) -> None:
        self.stats = np.loadtxt(fpath, dtype=np.float64)

    def getStats(self) -> dict[str, dict[str, np.ndarray]]:
        return {"bm": {"real_time": self.stats}}

"""
        "It doesn't support filtering, different benchmarks and the metric is hardcoded - but you "
        "get the idea. It's that simple.\n"
        "Now to compare two datasets in csvs, just run:\n"
        "python -m benchstats ./csv1.csv ./csv2.csv --files_parser ./myCSV.py\n\n"
        "If you'll make a parser that could be useful to other people, please consider adding it "
        "to the project's built-in parsers list by opening a thread with a suggestion in "
        "https://github.com/Arech/benchstats/issues or by making a PR into the repo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "file1",
        help="Path to the first data file with benchmark results. See also --file1_parser and "
        "--filter1 arguments",
        metavar="<path/to/file1>",
    )

    parser.add_argument(
        "--files_parser",
        help="Sets files parser class identifier, if a built-in parser used (options are: "
        f"{', '.join(getBuiltinParsers())}). Or a path to .py file defining a custom parser, "
        "inherited from 'benchstats.common.ParserBase' class. The parser class name must be the "
        "same as the file name. See below for example",
        default="GbenchJson",
        metavar="<files parser class or path>",
    )

    parser.add_argument(
        "--file1_parser",
        help="Same as --files_parser, but applies only to file1.",
        default=None,
        metavar="<file1 parser class or path>",
    )

    parser.add_argument(
        "--filter1",
        help="If specified, sets a Python regular expression to select benchmarks by name from <file1>",
        metavar="<reg expr>",
        default=None,
    )

    parser.add_argument(
        "file2",
        help="Path to the second data file with benchmark results. See also --file2_parser and "
        "--filter2 arguments",
        metavar="<path/to/file1>",
    )

    parser.add_argument(
        "--file2_parser",
        help="Same as --files_parser, but applies only to file2.",
        default=None,
        metavar="<file2 parser class or path>",
    )

    parser.add_argument(
        "--filter2",
        help="If specified, sets a regular expression (in Python RE dialect) to select benchmarks by name from <file2>",
        metavar="<reg expr>",
        default=None,
    )

    parser.add_argument(
        "metrics",
        help="List of metric identifiers to use in tests for each benchmark. Default: %(default)s. "
        "For deterministic algorithms highly recommend measure and use minimum latency per "
        "repetition.",
        nargs="*",
        default=["real_time"],
    )

    parser.add_argument(
        "--main_metrics",
        help="Indexes in 'metrics' list specifying the main metrics. These are displayed differently and differences "
        "detected cause script to exit(1). Default: %(default)s.",
        nargs="+",
        metavar="<metric idx>",
        type=int,
        default=[0],
    )

    parser.add_argument(
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

    parser.add_argument(
        "--alpha",
        help="Set statistical significance level (a desired mistake probability level)\n"
        "Note! This is an actual probability of a mistake only when all preconditions of the "
        "chosen statistical test are met. One of the most important preconditions like indepence of "
        "individual measurements, or constancy of underlying distribution parameters are almost "
        "never met in a real-life benchmarking, even on a properly quiesced hardware. Real "
        "mistake rates are higher. Default %(default)s",
        metavar="<positive float less than 0.5>",
        type=float,
        default=kDefaultAlpha,
    )

    parser.add_argument(
        "--bonferroni",
        help="If set, applies a Bonferroni multiple comparisons correction "
        "(https://en.wikipedia.org/wiki/Bonferroni_correction).",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--expect_same",
        help="If set, assumes that distributions are the same (i.e. H0 hypothesis is true) and shows some additional "
        "statistics useful for ensuring that a benchmark code is stable enough, or the machine is quiesced enough. "
        "One good example is when <file1> and <file2> are made with exactly the same binary running on exactly the "
        "same machine in the same state - on a machine with a proper setup tests shouldn't find difference.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--no_debug_log",
        help="If set, disables debug logging to stdout",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--no_colors",
        help="If set, disables colored output",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--export_to",
        help="Path to file to store comparison results to.",
        metavar="<path/to/export_file>",
    )

    parser.add_argument(
        "--export_fmt",
        help=f"Format of export file. Options are: {', '.join(kAvailableFormats)}. If not set, "
        "inferred from --export_to file extension",
        choices=kAvailableFormats,
        default=None,
        metavar="<format id>",
    )

    parser.add_argument(
        "--export_light",
        help="If set, uses light theme instead of dark",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--hide_sample_sizes",
        help="If set, hides sizes of datasets used in a test",
        action="store_true",
        default=False,
    )

    return parser

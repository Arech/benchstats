# using abs path only here to alleviate debugging
from benchstats.compare import compareStats
from benchstats.common import LoggingConsole, detectExportFormat, bmNamesTransform
from benchstats.cli_parser import makeParser
from benchstats.render import renderComparisonResults
from benchstats.parsers import getParserFor

import argparse
import os
from rich.terminal_theme import DIMMED_MONOKAI as DarkTheme, DEFAULT_TERMINAL_THEME as LightTheme


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        parser = makeParser()
        args = parser.parse_args()

    export_fmt = detectExportFormat(args.export_to, args.export_fmt)
    if export_fmt is not None:
        if os.path.isfile(args.export_to):
            os.remove(args.export_to)
        else:
            assert not os.path.exists(args.export_to), "Invalid --export_to value"

    console = LoggingConsole(
        no_color=not args.colors,
        record=export_fmt is not None,
        log_level=(
            LoggingConsole.LogLevel.Debug if args.show_debug else LoggingConsole.LogLevel.Warning
        ),
    )

    if args.alpha > 0.5 or args.alpha <= 0:
        console.critical(
            "--alpha must be a positive number less than 0.5 (%.2f is given)" % args.alpha
        )
        return 2

    Parser1 = getParserFor(args.files_parser if args.file1_parser is None else args.file1_parser)
    source1 = Parser1(args.file1, args.filter1, args.metrics, debug_log=console)

    alt_delimiter = source1.getAltDelimiter()
    two_sources_mode = alt_delimiter is None

    s1 = source1.getStats()
    del source1

    s1 = bmNamesTransform(
        s1,
        getattr(args, "from") if args.from1 is None else args.from1,
        args.to if args.to1 is None else args.to1,
        1,
        console,
    )

    if two_sources_mode:
        if not args.file2:
            raise ValueError(
                "File1 parser set two sources comparison mode, the second source is needed!"
            )

        Parser2 = getParserFor(
            args.files_parser if args.file2_parser is None else args.file2_parser
        )
        source2 = Parser2(args.file2, args.filter2, args.metrics, debug_log=console)
        assert source2.getAltDelimiter() is None, (
            "Parser2 is incompatible with two sources comparison mode"
        )

        s2 = source2.getStats()
        del source2

        s2 = bmNamesTransform(
            s2,
            getattr(args, "from") if args.from2 is None else args.from2,
            args.to if args.to2 is None else args.to2,
            2,
            console,
        )
    else:
        if args.file2 or args.file2_parser or args.filter2 or args.from2 or args.to2:
            raise ValueError(
                "File1 parser set single source comparison mode, the second source and its arguments aren't supported"
            )

        if not isinstance(alt_delimiter, str) or not alt_delimiter:
            raise ValueError(
                "Invalid parser for file1, getAltDelimiter() must return None or a non empty string"
            )
        s2 = None

    alpha = args.alpha
    if args.bonferroni:
        alpha = alpha / len(s1)
        console.info(
            f"Bonferroni correction for {len(s1)} comparisons turns alpha={args.alpha:.3e} into {alpha:.3e}"
        )

    if args.main_metrics is None or len(args.main_metrics) < 1:
        main_metrics = [args.metrics[0]]
    else:
        main_metrics = [args.metrics[mi] for mi in args.main_metrics]

    cr = compareStats(
        s1,
        s2,
        method=args.method,
        main_metrics=main_metrics,
        alpha=alpha,
        alt_delimiter=alt_delimiter,
        debug_log=console,
        store_sets=args.sample_stats is not None,
    )

    renderComparisonResults(
        cr,
        console,
        expect_same=args.expect_same,
        main_metrics=main_metrics,
        sample_stats=args.sample_stats,
        dark_theme=not args.export_light,
        show_sample_sizes=args.sample_sizes,
        always_show_pvalues=args.always_show_pvalues,
        multiline=args.multiline,
        metric_precision=args.metric_precision,
        show_percent_diff=args.percent_diff,
    )

    if cr.at_least_one_differs:
        console.warning(
            "At least one significant difference in main metrics was detected. Returning with exit(1)."
        )

    if export_fmt is not None:
        if "txt" == export_fmt:
            console.save_text(args.export_to)
        elif "svg" == export_fmt:
            console.save_svg(
                args.export_to, title="", theme=LightTheme if args.export_light else DarkTheme
            )
        elif "html" == export_fmt:
            console.save_html(args.export_to, theme=LightTheme if args.export_light else DarkTheme)
        else:
            assert False, "NOT IMPLEMENTED?!"

    if cr.at_least_one_differs:
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

# using abs path only here to alleviate debugging
from benchstats.compare import compareStats
from benchstats.parser_GbenchJson import getLogger
from benchstats.cli_parser import makeParser
from benchstats.render import renderComparisonResults
from benchstats.parsers import getParserFor

import rich

"""
TODO:
- how to compare different bms from the same file (i.e. bm name translation method)?
- add credits to CLI parser
- make structlog respect --no_colors, or use Rich's logger https://rich.readthedocs.io/en/stable/console.html#logging
"""


def main():
    parser = makeParser()
    args = parser.parse_args()

    debug_log = not args.no_debug_log
    logger = getLogger() if debug_log else False

    if args.alpha > 0.5 or args.alpha <= 0:
        if debug_log:
            logger.critical(
                "--alpha must be a positive number less than 0.5 (%.2f is given)", args.alpha
            )
        exit(2)

    Parser1 = getParserFor(args.files_parser if args.file1_parser is None else args.file1_parser)
    Parser2 = getParserFor(args.files_parser if args.file2_parser is None else args.file2_parser)

    s1 = Parser1(args.file1, args.filter1, args.metrics, debug_log=logger).getStats()
    s2 = Parser2(args.file2, args.filter2, args.metrics, debug_log=logger).getStats()

    alpha = args.alpha
    if args.bonferroni:
        alpha = alpha / len(s1)
        if debug_log:
            logger.info(
                f"Bonferroni correction for {len(s1)} comparisons turns alpha={args.alpha:.3e} into {alpha:.3e}"
            )

    cr = compareStats(s1, s2, method=args.method, alpha=alpha, debug_log=logger)

    # not passing args.main_metrics and not making render do that to make sure we correctly refer
    # to indexes in args.metrics. compareStats() doesn't explicitly guarantee to keep the order.
    if args.main_metrics is None or len(args.main_metrics) < 1:
        main_metrics = [args.metrics[0]]
    else:
        main_metrics = [args.metrics[mi] for mi in args.main_metrics]

    renderComparisonResults(
        cr,
        expect_same=args.expect_same,
        main_metrics=main_metrics,
        export_to=args.export_to,
        export_fmt=args.export_fmt,
        export_dark=not args.export_light,
        disable_colors=args.no_colors,
        show_sample_sizes=not args.hide_sample_sizes,
        always_show_pvalues=args.always_show_pvalues
    )

    if cr.at_least_one_differs:
        if debug_log:
            logger.info(
                "At least one significant difference in main metrics was detected. Returning with exit(1)."
            )
        exit(1)


if __name__ == "__main__":
    main()

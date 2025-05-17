"""Rendering results of compare::compareStats()"""

from collections.abc import Iterable
import numpy as np
import rich
import rich.style
from rich.text import Text
import math

from .common import LoggingConsole
from .compare import CompareStatsResult, kMethods


# values are how many chars should be the same to trigger match
kPossibleStatNames = {"extremums": 2, "median": 3, "iqr": 2, "std": 3}
_kRobustStatValues = {"extremums": [0, 100], "median": [50], "iqr": [25, 75]}

kDefaultStyles = {
    "benchmark_name_same": None,
    "benchmark_name_diff_main": "#FF6060",
    "benchmark_name_diff_secondary": "#C0B040",
    "pval_format": ".5f",  # no point in e notation
    "pval_format_generic": ".1e",  # used iif pval_format isn't enough to print alpha & pvals.
    "default_metric_unit": "s",
    "diff_result_sign": "bold",
    "metric_main_diff": "#FF1010",
    "metric_main_same": None,
    "metric_scnd_diff": "#B0A000",
    "metric_scnd_same": None,
    "min_metric_name_len": 10,
    "percents_precision": 1,
    "row_styles_dark": ["", "on #202020"],
    "row_styles_light": ["", "on #F0F0F0"],
    "header_perc_fmt": ".2f",
    "show_stats_header": True,
}


def makeReadable(value: float, prec: int):
    """Prints float `value` in a readable form with a corresponding suffix (if it's known, otherwise
    uses scientific notation) and given precision"""

    sign = "" if value >= 0 else "-"
    value = abs(value)

    def _render(v: float):
        if v >= 100.0:
            return f"{sign}{v:{prec + 4}.{prec}f}"
        return (
            f"{sign}{v:{prec + 4}.{prec + 1}f}"
            if v >= 10.0
            else f"{sign}{v:{prec + 4}.{prec + 2}f}"
        )

    if 0 == value:
        return _render(0)

    if value >= 1:
        return _render(value)
    orig_value = value
    value *= 1000
    if value >= 1:
        return _render(value) + "m"
    value *= 1000
    if value >= 1:
        return _render(value) + "u"
    value *= 1000
    if value >= 1:
        return _render(value) + "n"
    value *= 1000
    if value >= 1:
        return _render(value) + "p"
    value *= 1000
    if value >= 1:
        return _render(value) + "f"
    return f"{sign}{orig_value:.{prec}e}"


def _sanitizeSampleStats(sample_stats, perc_fmt):
    if sample_stats is None:
        return None, None
    use_std = False
    perc = []
    for ss in sample_stats:
        assert isinstance(ss, (int, float, str))
        if isinstance(ss, str):
            lower_s = ss.lower()
            found = False
            for exp_name, chars_to_match in kPossibleStatNames.items():
                if chars_to_match > len(lower_s):
                    continue
                if exp_name[:chars_to_match] == lower_s[:chars_to_match]:
                    found = True
                    if exp_name in _kRobustStatValues:
                        perc.extend(_kRobustStatValues[exp_name])
                    else:
                        assert exp_name == "std", "kPossibleStatNames ->| _kRobustStatValues"
                        use_std = True
                    break
            if found:
                continue
        fv = float(ss)  # fails if can't parse
        assert 0 <= fv and fv <= 100, "Stat values must be in range [0,100]"
        perc.append(fv)

    if not use_std and len(perc) < 1:
        return None, None

    perc = sorted(list(frozenset(perc)))

    def _neatNum(v: float):
        if v == int(v):
            return f"{int(v):d}"
        return f"{v:{perc_fmt}}"

    column_descr = ""
    if len(perc):
        column_descr += "["
        column_descr += "%, ".join([_neatNum(p) for p in perc])
        column_descr += "%]"
        if use_std:
            column_descr += ", "

    if use_std:
        column_descr += "std"

    return {"percentiles": perc, "std": use_std}, column_descr


def renderComparisonResults(
    comp_res: CompareStatsResult,
    console: LoggingConsole | None,  # if none will construct own
    dark_theme: bool = True,
    title: None | bool | str = True,  # None, False - disables title, str - customizes it
    style_overrides: dict = None,  # overrides for kDefaultStyles
    main_metrics: Iterable[str] = None,
    show_sample_sizes: bool = False,
    sample_stats=None,  # or iterable with predefined values: float%, or from kPossibleStatNames.keys()
    expect_same: bool = False,  # if true, show stats from assumption h0 is true
    always_show_pvalues: bool = False,
    multiline: bool = True,  # per metric report uses several lines
    metric_precision: int = 4,  # total digits in metric reported value. Min 3
    show_percent_diff: bool = True,  # if true, also show percent difference between values.
    # Value 0 is always the base of comparison, so the diff is always (value1 - value0) / value0.
    # Curly braces are used to mark the percent difference in the output.
) -> None:
    if console is None:
        console = LoggingConsole(emoji=False, highlight=False)
    else:
        assert isinstance(console, LoggingConsole)

    assert isinstance(comp_res, CompareStatsResult)
    if len(comp_res.results) < 1:
        console.failure(
            "Object with comparison results is empty. Perhaps there was no "
            "intersection in benchmark names between set1 and set2?"
        )
        return

    if style_overrides is None:
        style_overrides = kDefaultStyles
    assert isinstance(style_overrides, dict)

    def _getFmt(field: str):
        return style_overrides.get(field, kDefaultStyles[field])

    sample_stats, _column_descr = _sanitizeSampleStats(sample_stats, _getFmt("header_perc_fmt"))

    pval_fmt = _getFmt("pval_format")
    # a failsafe against too small alpha and consequently pvals for differences
    pval_fmt = (
        _getFmt("pval_format_generic") if 0.0 == float(f"{comp_res.alpha:{pval_fmt}}") else pval_fmt
    )
    pval_total_len = len(f" p={1 / 3:{pval_fmt}}")

    if title is not None:
        if isinstance(title, bool):
            title = (
                f"Benchmark comparison results ([link={kMethods[comp_res.method]['url']}]"
                f"{kMethods[comp_res.method]['name']}[/link], alpha={comp_res.alpha:{pval_fmt}})"
                if title
                else None
            )
    assert title is None or isinstance(title, str)

    delim_space = "\n" if multiline else " "

    _column_descr = (
        f",{delim_space}{_column_descr}"
        if _getFmt("show_stats_header") and sample_stats is not None
        else ""
    )

    metrics = comp_res.getMetrics()
    if main_metrics is None or len(main_metrics) < 1:
        main_metrics = [metrics[0]]
    else:
        assert all([isinstance(m, str) and m in metrics for m in main_metrics])

    scnd_metrics = [m for m in metrics if m not in main_metrics]
    iter_metrics = [*main_metrics, *scnd_metrics]

    metric_unit_keys = [f"metric_{m}_unit" for m in iter_metrics]

    assert isinstance(metric_precision, int)
    metric_precision = max(3, metric_precision) - 3

    # vars for h0==true assumption
    fp_less_metrics = {}  # number of false positives in less comparison per metric
    fp_gr_metrics = {}  # number of false positives in greater comparison per metric

    def _makeColumns(metr: Iterable[str]) -> list[str]:
        return [f"{m} (means){_column_descr}" for m in metr]

    theme_style = "dark" if dark_theme else "light"
    row_styles_fld = f"row_styles_{theme_style}"
    _def_justify = "left"  # unfortunately, applies to all rows, instead of only captions
    # Also very unfortunately, there seems to be no way of controlling text overflows in rich tables
    # (what our --multiline option were meant to do). If there's more text in a row that the width
    # of the terminal, columns without no_wrap=True will be unconditionally wrapped. But if all
    # columns have no_wrap=True, text will be unconditionally cropped. So we have to have just one
    # most important column (benchmark name) with no_wrap=True and all the rest will be wrapped if
    # a row length is too large. What we need is a table expansion if rows are too lengthy, but
    # that doesn't seem to work (I remember vaguely it worked with some combination of settings a
    # while ago, but I'm not even sure it was due to settings, or due to a rich version change, or
    # due to potentially different terminal behavior).
    table = rich.table.Table(
        rich.table.Column("Benchmark", justify=_def_justify, no_wrap=True),
        *[rich.table.Column(s, justify=_def_justify) for s in _makeColumns(main_metrics)],
        *[rich.table.Column(s, justify=_def_justify) for s in _makeColumns(scnd_metrics)],
        title=title,
        row_styles=_getFmt(row_styles_fld),
    )

    perc0 = perc1 = std0 = std1 = std_perc_diff = ""  # not changing if flags aren't right
    show_sample_stats_perc = sample_stats is not None and len(sample_stats["percentiles"]) > 0
    show_sample_stats_std = sample_stats is not None and sample_stats["std"]
    std_sep = "," if show_sample_stats_perc else ""
    stats_set_delim = delim_space if show_sample_stats_perc else " "
    percent_delim = delim_space if show_percent_diff else " "

    for bm_name, results in comp_res.results.items():
        diff_main = any([r.result != "~" for m, r in results.items() if m in main_metrics])
        diff_scnd = any([r.result != "~" for m, r in results.items() if m in scnd_metrics])
        bm_fld = f"benchmark_name_{'diff_main' if diff_main else ('diff_secondary' if diff_scnd else 'same')}"

        cols = [None] * len(metrics)
        for idx, metric_name in enumerate(iter_metrics):
            res = results[metric_name]
            is_main = idx < len(main_metrics)
            is_diff = res.result != "~"

            m_fld = "main" if is_main else "scnd"
            diff_fld = "diff" if is_diff else "same"
            comp_res_fld = f"metric_{m_fld}_{diff_fld}"

            unit = style_overrides.get(metric_unit_keys[idx], kDefaultStyles["default_metric_unit"])
            comp_res_style = _getFmt(comp_res_fld)

            # set representative values & comparison result
            repr_value0 = (
                np.mean(res.val_set0) if isinstance(res.val_set0, np.ndarray) else res.val_set0
            )
            repr_value1 = (
                np.mean(res.val_set1) if isinstance(res.val_set1, np.ndarray) else res.val_set1
            )
            txt = Text(
                f"{makeReadable(repr_value0, metric_precision)}{unit} {res.result}",
                style=comp_res_style,
            )
            if is_diff:
                txt.stylize(_getFmt("diff_result_sign"), -1)
            txt.append(
                f" {makeReadable(repr_value1, metric_precision)}{unit}", style=comp_res_style
            )

            if show_percent_diff:
                percent_diff = (repr_value1 - repr_value0) * 100.0 / repr_value0
                txt.append(f" {{{percent_diff:+.1f}%}}")

            if sample_stats is not None:
                assert isinstance(res.val_set0, np.ndarray), "Need raw dataset to show sample_stats"
                percent_diff = ""
                if show_sample_stats_perc:
                    perc_set0 = np.percentile(res.val_set0, sample_stats["percentiles"])
                    perc_set1 = np.percentile(res.val_set1, sample_stats["percentiles"])
                    perc0 = (
                        f"[{','.join([makeReadable(pv, metric_precision) for pv in perc_set0])}]"
                    )
                    perc1 = (
                        f"[{','.join([makeReadable(pv, metric_precision) for pv in perc_set1])}]"
                    )
                    if show_percent_diff:
                        percent_diff = [
                            f"{(perc_set1[idx] - p0) * 100.0 / p0:+.1f}%"
                            for idx, p0 in enumerate(perc_set0)
                        ]
                        percent_diff = f"{{{','.join(percent_diff)}}}"

                if show_sample_stats_std:
                    s0 = np.std(res.val_set0, ddof=1)
                    s1 = np.std(res.val_set1, ddof=1)
                    std0 = f"{std_sep}{makeReadable(s0, metric_precision)}"
                    std1 = f"{std_sep}{makeReadable(s1, metric_precision)}"
                    if show_percent_diff:
                        std_perc_diff = f"{{{(s1 - s0) * 100.0 / s0:+.1f}%}}"

                txt.append(
                    f"{delim_space}{perc0}{std0} {res.result}{stats_set_delim}{perc1}{std1}"
                    f"{percent_delim}{percent_diff}{std_perc_diff}",
                    style=comp_res_style,
                )

            # pvalue
            if always_show_pvalues or is_diff:
                str_pval = f"{res.pvalue:{pval_fmt}}"
                # showing trailing plus to highlight that it's not a true zero, reasonably assuming pvalue is never zero
                next_char = "+" if 0 == float(str_pval) else " "
                txt.append(f"{delim_space}p={str_pval}{next_char}", style=comp_res_style)
            else:
                txt.append(
                    delim_space + " " * (pval_total_len * int(show_sample_sizes == True)),
                    style=comp_res_style,
                )

            # sample sizes
            if show_sample_sizes:
                txt.append(f"({res.size1} vs {res.size2})", style=comp_res_style)

            cols[idx] = txt

            if is_diff:
                if "<" == res.result:
                    fp_less_metrics[metric_name] = 1 + fp_less_metrics.get(metric_name, 0)
                else:
                    assert ">" == res.result
                    fp_gr_metrics[metric_name] = 1 + fp_gr_metrics.get(metric_name, 0)

        table.add_row(Text(bm_name, style=_getFmt(bm_fld)), *cols)

    console.print(table)

    if expect_same:
        n_total = len(comp_res.results)
        console.print(
            "Assuming [red bold](!!!)[/red bold] that underlying data generator is the same for both "
            "benchmark sets, here's the number of [bold]false positives[/bold] per metric for "
            f"{n_total} tests:"
        )
        metr_len = _getFmt("min_metric_name_len")
        prec = _getFmt("percents_precision")
        tot_width = math.ceil(math.log10(n_total))
        for m in iter_metrics:
            fp_l, fp_g = fp_less_metrics.get(m, 0), fp_gr_metrics.get(m, 0)
            console.print(
                f"[bold white]{m:{metr_len}s}:[/bold white]"
                f" for [bold]<[/bold] {fp_l:{tot_width}d} ({fp_l * 100 / n_total:{3 + prec}.{prec}f}%)"
                f" for [bold]>[/bold] {fp_g:{tot_width}d} ({fp_g * 100 / n_total:{3 + prec}.{prec}f}%)"
            )

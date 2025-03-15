"""Rendering results of compare::compareStats()"""

from collections.abc import Iterable
import numpy as np
import os
import rich
import rich.style
from rich.text import Text
from rich.terminal_theme import DIMMED_MONOKAI as DarkTheme, DEFAULT_TERMINAL_THEME as LightTheme
import math

from .compare import CompareStatsResult, kMethods


kAvailableFormats = ("txt", "svg", "html")

# values are how many chars should be the same to trigger match
kPossibleStatNames = {"extremums": 2, "median": 3, "iqr": 2, "std": 3}
_kRobustStatValues = {"extremums": [0, 100], "median": [50], "iqr": [25, 75]}

kDefaultStyles = {
    "benchmark_name_same": None,
    "benchmark_name_diff_main": "#FF6060",
    "benchmark_name_diff_secondary": "#C0B040",
    "metric_precision": 1,
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
    "row_styles_dark": ["", "on #181818"],
    "row_styles_light": ["", "on #F0F0F0"],
    "header_perc_fmt": ".2f",
    "show_stats_header": True,
}


def _detectExportFormat(export_to, export_fmt):
    assert (export_to is None and export_fmt is None) or (
        isinstance(export_to, str) and len(export_to) > 0
    )
    assert export_fmt is None or export_fmt in kAvailableFormats

    if export_to is not None and export_fmt is None:
        root, ext = os.path.splitext(export_to)
        assert ext in [
            "." + e for e in kAvailableFormats
        ], f"Unrecognized export file extension '{ext}' of a file in --export_to parameter"
        export_fmt = ext[1:]

    return export_fmt


def makeReadable(value: float, prec: int):
    """Prints float `value` in a readable form with a corresponding suffix (if it's known, otherwise
    uses scientific notation) and given precision"""

    def _render(v: float):
        if v>=100.0:
            return f"{v:{prec+4}.{prec}f}"
        return f"{v:{prec+4}.{prec+1}f}" if v >= 10.0 else f"{v:{prec+4}.{prec+2}f}"

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
    return f"{orig_value:.{prec}e}"


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
    export_to: None | str = None,
    export_fmt: None | str = None,
    export_dark: bool = True,  # use dark theme
    disable_colors: bool = False,
    title: None | bool | str = True,  # None, False - disables title, str - customizes it
    style_overrides: dict = None,  # overrides for kDefaultStyles
    main_metrics: Iterable[str] = None,
    show_sample_sizes: bool = False,
    sample_stats=None,  # or iterable with predefined values: float%, or from kPossibleStatNames.keys()
    expect_same: bool = False,  # if true, show stats from assumption h0 is true
    always_show_pvalues: bool = False,
    multiline: bool = True,  # per metric report uses several lines
):
    assert isinstance(comp_res, CompareStatsResult)
    if style_overrides is None:
        style_overrides = kDefaultStyles
    assert isinstance(style_overrides, dict)

    def _getFmt(field: str):
        return style_overrides.get(field, kDefaultStyles[field])

    sample_stats, _column_descr = _sanitizeSampleStats(sample_stats, _getFmt("header_perc_fmt"))
    export_fmt = _detectExportFormat(export_to, export_fmt)

    pval_fmt = _getFmt("pval_format")
    # a failsafe against too small alpha and consequently pvals for differences
    pval_fmt = (
        _getFmt("pval_format_generic") if 0.0 == float(f"{comp_res.alpha:{pval_fmt}}") else pval_fmt
    )
    pval_total_len = len(f" p={1/3:{pval_fmt}}")

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
    delim_nospace = "\n" if multiline else ""
    delim_space_multiline = " " if multiline else ""

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
    metric_prec = _getFmt("metric_precision")

    # vars for h0==true assumption
    fp_less_metrics = {}  # number of false positives in less comparison per metric
    fp_gr_metrics = {}  # number of false positives in greater comparison per metric

    def _makeColumns(metr: Iterable[str]) -> list[str]:
        return [f"{m} (means){_column_descr}" for m in metr]

    theme_style = "dark" if export_dark else "light"
    row_styles_fld = f"row_styles_{theme_style}"
    _def_justify = "left"  # unfortunately, applies to all rows, instead of only captions
    table = rich.table.Table(
        rich.table.Column("Benchmark", justify=_def_justify),
        *[rich.table.Column(s, justify=_def_justify) for s in _makeColumns(main_metrics)],
        *[rich.table.Column(s, justify=_def_justify) for s in _makeColumns(scnd_metrics)],
        title=title,
        row_styles=_getFmt(row_styles_fld),
    )

    perc1 = perc2 = std1 = std2 = ""
    show_sample_stats_perc = sample_stats is not None and len(sample_stats["percentiles"]) > 0
    show_sample_stats_std = sample_stats is not None and sample_stats["std"]
    std_sep = "," if show_sample_stats_perc else ""
    stats_set_delim = delim_space if show_sample_stats_perc else " "


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
            repr_value1 = (
                np.mean(res.val_set1) if isinstance(res.val_set1, np.ndarray) else res.val_set1
            )
            repr_value2 = (
                np.mean(res.val_set2) if isinstance(res.val_set2, np.ndarray) else res.val_set2
            )
            txt = Text(
                f"{makeReadable(repr_value1, metric_prec)}{unit} {res.result}",
                style=comp_res_style,
            )
            if is_diff:
                txt.stylize(_getFmt("diff_result_sign"), -1)
            txt.append(f" {makeReadable(repr_value2, metric_prec)}{unit}", style=comp_res_style)

            if sample_stats is not None:
                assert isinstance(res.val_set1, np.ndarray), "Need raw dataset to show sample_stats"
                if show_sample_stats_perc:
                    perc1 = f"[{','.join([makeReadable(pv,metric_prec) for pv in np.percentile(res.val_set1, sample_stats['percentiles'])])}]"
                    perc2 = f"[{','.join([makeReadable(pv,metric_prec) for pv in np.percentile(res.val_set2, sample_stats['percentiles'])])}]"

                if show_sample_stats_std:
                    std1 = f"{std_sep}{makeReadable(np.std(res.val_set1,ddof=1),metric_prec)}"
                    std2 = f"{std_sep}{makeReadable(np.std(res.val_set2,ddof=1),metric_prec)}"

                txt.append(
                    f"{delim_space}{perc1}{std1} {res.result}{stats_set_delim}{perc2}{std2}",
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

    console = rich.console.Console(
        record=(export_fmt is not None),
        color_system=(None if disable_colors else "auto"),
        emoji=False,
        highlight=False,
    )

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
                f" for [bold]<[/bold] {fp_l:{tot_width}d} ({fp_l*100/n_total:{3+prec}.{prec}f}%)"
                f" for [bold]>[/bold] {fp_g:{tot_width}d} ({fp_g*100/n_total:{3+prec}.{prec}f}%)"
            )

    if export_fmt is not None:
        if "txt" == export_fmt:
            console.save_text(export_to)
        elif "svg" == export_fmt:
            console.save_svg(export_to, title="", theme=DarkTheme if export_dark else LightTheme)
        elif "html" == export_fmt:
            console.save_html(export_to, theme=DarkTheme if export_dark else LightTheme)
        else:
            assert False, "NOT IMPLEMENTED?!"

"""Rendering results of compare::compareStats()"""

from collections.abc import Iterable
import os
import rich
from rich.text import Text
from rich.terminal_theme import DIMMED_MONOKAI as DarkTheme, DEFAULT_TERMINAL_THEME as LightTheme
import math

from .compare import CompareStatsResult, kMethods


kAvailableFormats = ("txt", "svg", "html")

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
    "sample_sizes": None,
    "min_metric_name_len": 10,
    "percents_precision": 1,
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
    if value >= 1:
        return f"{value:{prec+4}.{prec}f}"
    orig_value = value
    value *= 1000
    if value >= 1:
        return f"{value:{prec+4}.{prec}f}m"
    value *= 1000
    if value >= 1:
        return f"{value:{prec+4}.{prec}f}u"
    value *= 1000
    if value >= 1:
        return f"{value:{prec+4}.{prec}f}n"
    value *= 1000
    if value >= 1:
        return f"{value:{prec+4}.{prec}f}p"
    value *= 1000
    if value >= 1:
        return f"{value:{prec+4}.{prec}f}f"
    return f"{orig_value:.{prec}e}"


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
    expect_same: bool = False,  # if true, show stats from assumption h0 is true
):
    assert isinstance(comp_res, CompareStatsResult)
    if style_overrides is None:
        style_overrides = kDefaultStyles
    assert isinstance(style_overrides, dict)

    export_fmt = _detectExportFormat(export_to, export_fmt)

    pval_fmt = style_overrides.get("pval_format", kDefaultStyles["pval_format"])
    # a failsafe against too small alpha and consequently pvals for differences
    pval_fmt = (
        style_overrides.get("pval_format_generic", kDefaultStyles["pval_format_generic"])
        if 0.0 == float(f"{comp_res.alpha:{pval_fmt}}")
        else pval_fmt
    )
    pval_total_len = len(f" p={1/3:{pval_fmt}}")

    if title is not None:
        if isinstance(title, bool):
            title = (
                f"Benchmark comparison results ([link={kMethods[comp_res.method]['url']}]"
                f"{kMethods[comp_res.method]['name']}[/link], alpha={comp_res.alpha:{pval_fmt}}. "
                "Metric values are averages)"
                if title
                else None
            )
    assert title is None or isinstance(title, str)

    metrics = comp_res.getMetrics()
    if main_metrics is None or len(main_metrics) < 1:
        main_metrics = [metrics[0]]
    else:
        assert all([isinstance(m, str) and m in metrics for m in main_metrics])

    scnd_metrics = [m for m in metrics if m not in main_metrics]
    iter_metrics = [*main_metrics, *scnd_metrics]

    metric_unit_keys = [f"metric_{m}_unit" for m in iter_metrics]
    metric_prec = style_overrides.get("metric_precision", kDefaultStyles["metric_precision"])

    # vars for h0==true assumption
    fp_less_metrics = {}  # number of false positives in less comparison per metric
    fp_gr_metrics = {}  # number of false positives in greater comparison per metric

    table = rich.table.Table("Benchmark", *main_metrics, *scnd_metrics, title=title)

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
            fld = f"metric_{m_fld}_{diff_fld}"

            unit = style_overrides.get(metric_unit_keys[idx], kDefaultStyles["default_metric_unit"])
            def_style = style_overrides.get(fld, kDefaultStyles[fld])
            txt = Text(
                f"{makeReadable(res.repr_value1, metric_prec)}{unit} {res.result}",
                style=def_style,
            )
            if is_diff:
                txt.stylize(
                    style_overrides.get("diff_result_sign", kDefaultStyles["diff_result_sign"]), -1
                )
            txt.append(f" {makeReadable(res.repr_value2, metric_prec)}{unit}", style=def_style)
            if is_diff:
                str_pval = f"{res.pvalue:{pval_fmt}}"
                # showing trailing plus to highlight that it's not a true zero
                next_char = "+" if res.pvalue > 0 and 0 == float(str_pval) else " "
                txt.append(" p=" + str_pval, style=def_style)
            elif show_sample_sizes:
                txt.append(" " * pval_total_len, style=def_style)
                next_char = " "

            if show_sample_sizes:
                txt.append(
                    f"{next_char}({res.size1} vs {res.size2})",
                    style=style_overrides.get("sample_sizes", kDefaultStyles["sample_sizes"]),
                )

            cols[idx] = txt

            if is_diff:
                if "<" == res.result:
                    fp_less_metrics[metric_name] = 1 + fp_less_metrics.get(metric_name, 0)
                else:
                    assert ">" == res.result
                    fp_gr_metrics[metric_name] = 1 + fp_gr_metrics.get(metric_name, 0)

        table.add_row(
            Text(bm_name, style=style_overrides.get(bm_fld, kDefaultStyles[bm_fld])), *cols
        )

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
        metr_len = style_overrides.get("min_metric_name_len", kDefaultStyles["min_metric_name_len"])
        prec = style_overrides.get("percents_precision", kDefaultStyles["percents_precision"])
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

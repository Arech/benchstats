"""
qbench is a separate benchstats module designed to alleviate benchmarking and benchmark comparison
of Python callables.
"""

import time
import numpy as np
import itertools
import inspect
from rich.progress import Progress

from benchstats.compare import compareStats, CompareStatsResult
from benchstats.render import renderComparisonResults
from benchstats.common import LoggingConsole


# support for combined arguments of showBench()
_g_compareStats_args = None
_g_renderComparisonResults_args = None
_g_joint_args = None


def _getOptionalArgs(func):
    arg_spec = inspect.getfullargspec(func)
    return frozenset(arg_spec[0][len(arg_spec[0]) - len(arg_spec[3]) :])


if _g_compareStats_args is None:
    _g_compareStats_args = _getOptionalArgs(compareStats)
    _g_renderComparisonResults_args = _getOptionalArgs(renderComparisonResults)
    _g_joint_args = _g_compareStats_args.intersection(_g_renderComparisonResults_args)


def bench(
    funcs: tuple | list,
    iters: int = 100,
    reps: int = 10,
    warmup: int = 0,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Benchmarks the provided callables by measuring their runtimes over multiple iterations and repetitions.

    Args:
        funcs (tuple|list|callable): A tuple of callables, each callable must be invocable without
            arguments; Or a single callable to benchmark.
        iters (int, optional): Number of iterations per repetition. Defaults to 100.
        reps (int, optional): Number of repetitions. Defaults to 10.
        warmup (int, optional): Number of warmup iterations. Warmups are useful for letting the
            system to converge to a stable state. Also are mandatory for a certain code, such as
            jit-compiled functions from JAX. Defaults to 0.

    Returns:
        np.ndarray: A 3D numpy array of runtimes in seconds with shape (n_funcs, reps, iters). Or,
            if only one function is provided, a 2D array with shape (reps, iters).
    """
    if not isinstance(funcs, (list, tuple)):
        funcs = (funcs,)
    if any(not callable(func) for func in funcs):
        raise ValueError("All elements in funcs must be callable.")

    n_funcs = len(funcs)
    if n_funcs == 0:
        raise ValueError("funcs must be a tuple with at least one callable.")
    assert iters > 0, "iters must be a positive integer."
    assert reps > 0, "reps must be a positive integer."
    assert warmup >= 0, "reps must be a non-negative integer."

    if show_progress:
        progress = Progress(transient=True)
        warmup_task = progress.add_task("Warmup", total=warmup)
        iter_task = progress.add_task("Iterations", total=iters)
        reps_task = progress.add_task("Repetitions", total=reps)
        rwarmup = progress.track(range(warmup), task_id=warmup_task)
        rreps = progress.track(range(reps), task_id=reps_task)
        progress.start()
    else:
        rreps = range(reps)
        rwarmup = range(warmup)

    for w in rwarmup:
        for f in funcs:
            f()

    if show_progress:
        progress.update(warmup_task, visible=False)

    results = np.empty((n_funcs, reps, iters), dtype=np.float64)

    for r in rreps:
        if show_progress:
            progress.update(iter_task, completed=0)
        for i in range(iters):
            for f_idx, func in enumerate(funcs):
                start = time.perf_counter_ns()
                func()
                end = time.perf_counter_ns()
                results[f_idx, r, i] = (end - start) * 1e-9  # convert ns to seconds
            if show_progress:
                progress.update(iter_task, completed=i)

    if show_progress:
        progress.stop()

    if 1 == n_funcs:
        results = results[0]
    return results


def bench2(func1, func2, **kwargs):
    """
    Benchmarks two callables by measuring their runtimes over multiple iterations and repetitions.

    Args:
        func1 (callable): The first callable to benchmark.
        func2 (callable): The second callable to benchmark.
        **kwargs: Additional keyword arguments for the bench function.

    Returns:
        np.ndarray: A 3D numpy array of runtimes in seconds with shape (2, reps, iters).
    """
    return bench((func1, func2), **kwargs)


# TODO: would be nice if reporting supports some results tagging, so it's not just "set 1 vs set 2"
# but a more descriptive user provided strings, clarifying each set.
# This change require requires changes in benchmarks matching algo of compareStats() and reporting


def showBench(
    results: np.ndarray,
    bm_names: tuple | list | str = "code",
    alt_delimiter: str | None = None,
    metrics: dict = {"min": np.min, "mean": np.mean},
    console: LoggingConsole = LoggingConsole(),
    **kwCompareStats_and_renderArgs,
) -> CompareStatsResult:
    """
    Displays the benchmark results in a human-readable format.

    Args:
        results (np.ndarray): The benchmark results as a 3D or 2D numpy array. Essentially, the
            output of the bench() function.
            If a 2D array is provided (only a single function was benchmarked), then a fake
            comparison against itself is performed (essentially, it's just a performance report of
            the function). In that case `alt_delimiter` parameter must be None.
        bm_names (tuple|list|str, optional) and alt_delimiter: (str|None, optional): defines the
            names of the benchmarks and how they are compared one against the other. Options are:
            - if alt_delimiter is None:
                - if bm_names is a single string, then all results are assumed to be from
                    alternatives of the same code, and are all compared one against the other, using
                    numeric indices as alternative names.
                - if bm_names is a tuple or list of strings, of length that is an exact divisor of
                    the number of functions benchmarked. There must be more results than names.
                    The first set of functions are assumed to be different alternatives of a single
                    code with the first name, the second set of functions are assumed to be
                    different alternatives of the second set of functions with the second name, and
                    so on. Numeric indices are used as alternative names.
            - if alt_delimiter is a string:
                - bm_names must be a tuple or list of strings with the same length as the number
                    of functions benchmarked (results.shape[0]), where each string is parsed
                    according to the expected format:
                    "<common_name>{alt_delimiter}<alternative_name>".
        metrics (dict[str, callable], optional): A description of metrics functions used to
            aggregate data from individual benchmark iterations.
        kwCompareStats_and_renderArgs: Any optional arguments of the compareStats()
            or renderComparisonResults() functions. By default compareStats() also gets
            store_sets=True argument, and renderComparisonResults() gets
            show_sample_sizes=True and sample_stats=("extremums", "median") arguments.
    """
    if results.ndim == 2:
        results = np.expand_dims(results, axis=0)
    assert results.ndim == 3, "results must be a 3D numpy array."
    n_funcs = results.shape[0]

    if isinstance(bm_names, str):
        bm_names = (bm_names,)
    if not isinstance(bm_names, (list, tuple)):
        raise ValueError("names must be a tuple or list of strings.")
    n_names = len(bm_names)
    if n_names == 0:
        raise ValueError("names must be a non-empty tuple or list of strings.")

    compareStats_args = {"store_sets": True}
    renderComparisonResults_args = {
        "show_sample_sizes": True,
        "sample_stats": ("extremums", "median"),
    }
    unknown_args = {}
    for k, v in kwCompareStats_and_renderArgs.items():
        if k in _g_joint_args:
            compareStats_args[k] = v
            renderComparisonResults_args[k] = v
        elif k in _g_compareStats_args:
            compareStats_args[k] = v
        elif k in _g_renderComparisonResults_args:
            renderComparisonResults_args[k] = v
        else:
            unknown_args[k] = v
    if len(unknown_args) > 0:
        raise ValueError(
            f"Unknown arguments: {', '.join(unknown_args.keys())}. "
            "Please check the signature of compare.compareStats() and render.renderComparisonResults() for valid arguments."
        )

    def addBenchmark(sg: dict, bm_name: str, b_idx: int):
        assert bm_name not in sg, (
            f"Duplicate benchmark name '{bm_name}' found. Please provide unique names."
        )
        sg[bm_name] = {
            metric_name: metric_func(results[b_idx], axis=1)
            for metric_name, metric_func in metrics.items()
        }

    sg = {}
    if alt_delimiter is None:
        if n_funcs % n_names != 0:
            raise ValueError(
                f"Number of functions ({n_funcs}) must be divisible by the number of names ({n_names})."
            )
        n_funcs_per_bm = n_funcs // n_names

        alt_delimiter = "|"

        if n_funcs_per_bm <= 1:
            addBenchmark(sg, f"{bm_names[0]}{alt_delimiter}0", 0)
            addBenchmark(sg, f"{bm_names[0]}{alt_delimiter}same", 0)
        else:
            combination_set = list(itertools.combinations(range(n_funcs_per_bm), 2))
            func_group_idx = 0
            for bm_name in bm_names:
                for cs in combination_set:
                    n1 = f"{bm_name}{alt_delimiter}{cs[0]}"
                    if n1 not in sg:
                        addBenchmark(sg, n1, func_group_idx + cs[0])
                    n2 = f"{bm_name}{alt_delimiter}{cs[1]}"
                    if n2 not in sg:
                        addBenchmark(sg, n2, func_group_idx + cs[1])
                func_group_idx += n_funcs_per_bm

    else:
        assert isinstance(alt_delimiter, str)
        if n_names != n_funcs:
            raise ValueError(
                "If alt_delimiter is provided, names must be a tuple or list of strings with the same length as the number of functions benchmarked."
            )
        for b_idx, bm_name in enumerate(bm_names):
            if alt_delimiter not in bm_name:
                raise ValueError(
                    f"alt_delimiter '{alt_delimiter}' not found in benchmark name '{bm_name}'."
                )
            if bm_name in sg:
                raise ValueError(
                    f"Duplicate benchmark name '{bm_name}' found. Please provide unique names."
                )
            addBenchmark(sg, bm_name, b_idx)

    sr = compareStats(sg, None, alt_delimiter=alt_delimiter, debug_log=console, **compareStats_args)
    renderComparisonResults(sr, console=console, **renderComparisonResults_args)
    return sr


_g_bench_args = None
_g_showBench_args = None
if _g_bench_args is None:
    _g_bench_args = _getOptionalArgs(bench)
    _g_showBench_args = _getOptionalArgs(showBench)


def benchmark(funcs: tuple | list, **kwargs):
    """
    Benchmarks the provided callables and displays the results.
    Args:
        funcs (tuple|list|callable): A tuple of callables, each callable must be invocable without
            arguments; Or a single callable to benchmark.
        **kwargs: Additional keyword arguments for the bench() and showBench() functions, as well as
            optional arguments for compareStats() and renderComparisonResults() functions.
    """
    bench_args = {}
    show_args = {}
    compare_and_render_args = {}
    for k, v in kwargs.items():
        if k in _g_bench_args:
            bench_args[k] = v
        elif k in _g_showBench_args:
            show_args[k] = v
        else:
            compare_and_render_args[k] = v

    results = bench(funcs, **bench_args)
    return showBench(results, **show_args, **compare_and_render_args)

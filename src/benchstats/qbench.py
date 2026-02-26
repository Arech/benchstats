"""
qbench is a separate benchstats module designed to alleviate benchmarking and benchmark comparison
of Python callables.
"""

import argparse
from collections import namedtuple
from collections.abc import Callable, Iterable
import itertools
import inspect
import numpy as np
from rich.progress import Progress
from time import perf_counter_ns
from typing import Any

from benchstats.compare import compareStats, CompareStatsResult
from benchstats.render import renderComparisonResults
from benchstats.common import LoggingConsole


def _getOptionalArgs(func) -> dict[str, Any]:
    """Returns a dict of argument names of the function that have default values"""
    s = inspect.getfullargspec(func)
    kdefs = {}
    if s.defaults:
        def_ofs = len(s.args) - len(s.defaults)
        for i in range(len(s.defaults)):
            kdefs[s.args[def_ofs + i]] = s.defaults[i]
    if s.kwonlydefaults:
        kdefs.update(s.kwonlydefaults)

    return kdefs


# support for combined arguments of showBench()
_g_compareStats_args = frozenset(_getOptionalArgs(compareStats).keys())
_g_renderComparisonResults_args = frozenset(_getOptionalArgs(renderComparisonResults).keys())
_g_joint_args = _g_compareStats_args.intersection(_g_renderComparisonResults_args)


################################################################################
# core benchmark runners
################################################################################


class BenchmarkDescription(
    namedtuple(
        "BenchmarkDescription",
        ["bench_func", "args_func", "clear_cache_func", "wait_arg_complete", "wait_func_complete"],
    )
):
    """Describes a benchmark, i.e. which function to measure, how to get its arguments, how to
    respect asynchronicity, how to clear caches & etc.
    Note that
    """

    __slots__ = ()
    _max_len = 5
    _n_posargs = 3

    def __new__(
        cls,
        bench_func: Callable,
        args_func: Callable[[], Iterable] | None = None,
        clear_cache_func: Callable[[Iterable], None] | None = None,
        *,
        wait_complete: Callable[[Any], Any] | None = None,  # sets the default value for the 2 below
        wait_arg_complete: Callable[[Any], Any] | None = None,
        wait_func_complete: Callable[[Any], Any] | None = None,
    ):
        """Constructor to verify initialization correctness"""
        assert callable(bench_func)
        assert args_func is None or callable(args_func)
        assert clear_cache_func is None or callable(clear_cache_func)
        assert wait_complete is None or callable(wait_complete)
        assert wait_arg_complete is None or callable(wait_arg_complete)
        assert wait_func_complete is None or callable(wait_func_complete)

        if wait_complete:
            if wait_arg_complete is None:
                wait_arg_complete = wait_complete
            if wait_func_complete is None:
                wait_func_complete = wait_complete

        assert len(cls._fields) == cls._max_len
        assert cls._max_len - cls._n_posargs == 2
        return super().__new__(
            cls, bench_func, args_func, clear_cache_func, wait_arg_complete, wait_func_complete
        )

    @classmethod
    def _toArgsKwargs(
        cls,
        it,
        def_args_func: Callable[[], Iterable],
        def_wait_arg_complete: Callable[[Any], Any],
        def_wait_func_complete: Callable[[Any], Any],
        def_clear_cache: Callable[[Iterable], None],
    ):
        it_len = len(it)
        assert it_len <= cls._max_len

        args = [a for a in it[: cls._n_posargs]]
        if len(args) >= 2:
            if args[1] is None:
                args[1] = def_args_func
        else:
            args.append(def_args_func)

        if len(args) >= 3:
            if args[2] is None:
                args[2] = def_clear_cache
        else:
            args.append(def_clear_cache)

        assert len(args) == cls._n_posargs and all(callable(f) for f in args)

        def _extractDefault(idx, def_val):
            v = it[idx] if it_len >= cls._n_posargs else None
            return def_val if v is None else v

        kwargs = {
            "wait_arg_complete": _extractDefault(cls._n_posargs, def_wait_arg_complete),
            "wait_func_complete": _extractDefault(cls._n_posargs + 1, def_wait_func_complete),
        }
        assert len(kwargs) == cls._max_len - cls._n_posargs
        assert all(callable(f) for f in kwargs.values())
        return args, kwargs

    @classmethod
    def from_iterable(
        cls,
        it: Iterable,
        def_args_func: Callable[[], Iterable] | None = None,
        def_wait_arg_complete: Callable[[Any], Any] | None = None,
        def_wait_func_complete: Callable[[Any], Any] | None = None,
        def_clear_cache: Callable[[Iterable], None] | None = None,
    ):
        if def_args_func is None:
            def_args_func = lambda: tuple()  # noqa: E731
        def_wait_complete = lambda x: x  # noqa: E731
        if def_wait_arg_complete is None:
            def_wait_arg_complete = def_wait_complete
        if def_wait_func_complete is None:
            def_wait_func_complete = def_wait_complete
        if def_clear_cache is None:
            def_clear_cache = lambda x: None  # noqa: E731

        args, kwargs = cls._toArgsKwargs(
            it, def_args_func, def_wait_arg_complete, def_wait_func_complete, def_clear_cache
        )
        return cls(*args, **kwargs)


def _toBenchmarkDescription(
    funcs: Iterable | Callable,
    def_wait_arg_complete: Callable[[Any], Any] | None,
    def_wait_func_complete: Callable[[Any], Any] | None,
    def_clear_cache: Callable[[Iterable], None] | None,
) -> list[BenchmarkDescription]:
    """Transforms `funcs` argument of `bench()` function to a canonical list of BenchmarkDescription
    objects"""
    if not isinstance(funcs, Iterable):
        assert callable(funcs)
        funcs = (funcs,)

    assert len(funcs) > 0
    def_args_func = lambda: tuple()  # noqa: E731
    ret = []

    for e in funcs:
        if callable(e):
            # important to use from_iterable() to inject defaults
            ret.append(BenchmarkDescription.from_iterable((e, def_args_func)))
        else:
            ret.append(
                BenchmarkDescription.from_iterable(
                    e,
                    def_args_func,
                    def_wait_arg_complete,
                    def_wait_func_complete,
                    def_clear_cache,
                )
            )

    return ret


def bench(
    funcs: Iterable | Callable,
    *,
    iters: int = 100,
    reps: int = 10,
    warmup: int = 5,
    batch_functions: bool = False,
    randomize_iterations: bool = True,
    wait_complete: None | Callable[[Any], Any] = None,  # lambda x: x
    wait_arg_complete: None | Callable[[Any], Any] = None,
    wait_func_complete: None | Callable[[Any], Any] = None,
    clear_cache: None | Callable[[Iterable], None] = None,  # lambda x: None,
    show_progress_each: int = 1,
) -> np.ndarray:
    """
    Benchmarks the provided callables by measuring their runtimes over multiple iterations and repetitions.

    Arguments:
    - funcs (Iterable): A single callable, or an iterable describing callables to benchmark.
        In the latter case, each element could contain either:
        - a callable that could be used without arguments, or
        - an iterable containing `BenchmarkDescription` tuple or 1-3 callables:
            (bench_func, args_func, clear_cache_func), where args_func is function of 0 arguments
            returning a tuple or a list of arguments to pass to the bench_func, and clear_cache_func
            is a function taking an iterable returned from args_func() and doing all the necessary
            cache clearing.
    - iters (int, optional): Number of iterations per repetition. Defaults to 100.
    - reps (int, optional): Number of repetitions. Defaults to 10.
    - warmup (int, optional): Number of warmup iterations. Warmups are useful for letting the
        system to converge to a stable state. Also are mandatory for a certain code, such as
        jit-compiled functions from JAX. Defaults to 1.
    - batch_functions (bool, optional): controls how two inner-most benchmarking loops are
        organized. When False (the default), we first launch iteration loop, then loop over the
        functions: `for i in range(iters): for f in range(len(funcs)): do_benchmark(f)`. When
        True, it's reversed: `for f in range(len(funcs)): for i in range(iters): do_benchmark(f)`.
        The latter variant due to a strong hardware caches effect is only useful, when the function
        is going to be called in a tight loop, otherwise results will be more skewed.
    - randomize_iterations (bool, optional) Especially when batch_functions==False measuring timings
        of the same code in a predefined order might skewed results due to hardware caching effects.
        Setting this to True will ensure that each function will be called in a random order within
        an iteration. Defaults to True. Don't set to False unless len(funcs) >> 10.
    - wait_complete (Callable, optional): benchmarking asynchronous functions require awaiting for
        completion of their execution. wait_complete() is a function accepts a return value of
        bench_func() function to benchmark and returns when it's ready. If `wait_arg_complete`
        is None and `funcs` are used with args_func, - wait_arg_complete <- wait_complete
    - wait_arg_complete and wait_func_complete allow to set separate awaiting functions for
        args_func and bench_func. Note that wait_arg_complete is applied to each element of
        iterable that args_func() returns.
    - clear_cache (Callable, optional) - a function taking an iterable that an args_func could
        return, and performing all necessary HW & other cache clearing necessary. Return value is
        ignored. This parameter sets a default `clear_cache_func` for each of `funcs` if
        `clear_cache_func` isn't provided
    - show_progress_each - if an positive integer, will show progress bars and update each that
        number of iterations

    Returns:
        np.ndarray: A 3D numpy array of runtimes in seconds with shape (n_funcs, reps, iters). Or,
            if only one function is provided, a 2D array with shape (reps, iters).
    """
    show_progress = isinstance(show_progress_each, int) and show_progress_each > 0

    @staticmethod
    def _time_execution(bd: BenchmarkDescription) -> float:
        args = bd.args_func()
        assert isinstance(args, Iterable)
        # it's super important to materialize genexpr here, or it'll be timed!
        inputs = tuple(bd.wait_arg_complete(a) for a in args)

        start = perf_counter_ns()
        o = bd.wait_func_complete(bd.bench_func(*inputs))
        end = perf_counter_ns()
        return (end - start) * 1e-9

    if wait_arg_complete is None:
        wait_arg_complete = wait_complete
    if wait_func_complete is None:
        wait_func_complete = wait_complete

    funcs = _toBenchmarkDescription(funcs, wait_arg_complete, wait_func_complete, clear_cache)

    n_funcs = len(funcs)
    assert iters > 0, "iters must be a positive integer."
    assert reps > 0, "reps must be a positive integer."
    assert warmup >= 0, "reps must be a non-negative integer."

    if show_progress:
        progress = Progress(transient=True)
        warmup_task = progress.add_task("Warmup", total=warmup)
        if batch_functions:
            func_task = progress.add_task("Functions", total=n_funcs)
        else:
            iter_task = progress.add_task("Iterations", total=iters)
        reps_task = progress.add_task("Repetitions", total=reps)
        rwarmup = progress.track(range(warmup), task_id=warmup_task)
        rreps = progress.track(range(reps), task_id=reps_task)
        progress.start()
    else:
        rreps = range(reps)
        rwarmup = range(warmup)

    for f in funcs:
        for _ in rwarmup:
            _time_execution(f)

    if show_progress:
        progress.update(warmup_task, visible=False)

    results = np.empty((n_funcs, reps, iters), dtype=np.float64)
    bm_idxs = np.arange(n_funcs, dtype=np.int32)
    rng = np.random.default_rng()

    for r in rreps:
        if show_progress:
            progress.update(func_task if batch_functions else iter_task, completed=0)

        if batch_functions:
            if randomize_iterations:
                rng.shuffle(bm_idxs)
            for fi, f_idx in enumerate(bm_idxs):
                for i in range(iters):
                    results[f_idx, r, i] = _time_execution(funcs[f_idx])
                if show_progress and fi % show_progress_each == 0:
                    progress.update(func_task, completed=fi)
        else:
            for i in range(iters):
                if randomize_iterations:
                    rng.shuffle(bm_idxs)
                for f_idx in bm_idxs:
                    results[f_idx, r, i] = _time_execution(funcs[f_idx])
                if show_progress and i % show_progress_each == 0:
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
    *,
    bm_names: tuple | list | str = "code",
    alt_delimiter: str | None = None,
    metrics: dict = {"min": np.min, "mean": np.mean},
    console: LoggingConsole = LoggingConsole(),
    **kwCompareStats_and_renderArgs,
) -> CompareStatsResult:
    """
    Displays the benchmark results in a human-readable format.

    Args:
    - results (np.ndarray): The benchmark results as a 3D or 2D numpy array. Essentially, the
        output of the bench() function.
        If a 2D array is provided (only a single function was benchmarked), then a fake
        comparison against itself is performed (essentially, it's just a performance report of
        the function). In that case `alt_delimiter` parameter must be None.
    - bm_names (tuple|list|str, optional) and alt_delimiter: (str|None, optional): defines the
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
                according to the expected format: "<common_name>{alt_delimiter}<alternative_name>".
                Benchmarks with the same "<common_name>" are compared pairwise.
    - metrics (dict[str, callable], optional): A description of metrics functions used to
        aggregate data from individual benchmark iterations.
    - kwCompareStats_and_renderArgs: Any optional arguments of the compareStats()
        or renderComparisonResults() functions. By default compareStats() also gets
        store_sets=True argument, and renderComparisonResults() gets
        show_sample_sizes=True and sample_stats=("extremums", "median") arguments.

    Returns: CompareStatsResult object with comparison results
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


_g_bench_defaults = _getOptionalArgs(bench)
_g_bench_args = frozenset(_g_bench_defaults.keys())
_g_showBench_args = frozenset(_getOptionalArgs(showBench).keys())
assert not (_g_bench_args & _g_showBench_args)


def benchmark(funcs: tuple | list, **kwargs) -> tuple[CompareStatsResult, np.ndarray]:
    """
    Benchmarks the provided callables and displays the results.
    Args:
        funcs (tuple|list|callable): A tuple of callables, each callable must be invocable without
            arguments; Or a single callable to benchmark.
        **kwargs: Additional keyword arguments for the bench() and showBench() functions, as well as
            optional arguments for compareStats() and renderComparisonResults() functions.
    Returns: CompareStatsResult of the comparison and raw measured latencies as np.ndarray
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
    return showBench(results, **show_args, **compare_and_render_args), results


################################################################################
# benchmark registration & run support
################################################################################

# a global object to store registered benchmark generating functions.
# For more info see registerBenchmark()
benchmark_sets: dict[str, Callable] = {}


def registerBenchmark(f: Callable) -> Callable:
    """A function decorator to register a function as a benchmark generating function.

    A benchmark generating function `f` is a function of 0 parameters returning a dictionary mapping
    a benchmark name to an argument-less callable, or an iterable of 2 callables -
    (args_func, bench_func), where args_func is a function of 0 arguments returning a tuple or a
    list of arguments to pass to the bench_func. Name of `f` must adhere to
    `make_<bmset_name>_benchmark` convention, i.e. start with `make_` and on `_benchmark` strings,
    where the middle part defines a name of the benchmarks set.

    Each individual benchmark name in the returned by `f` dictionary must be unique across all
    benchmark sets going to be registered.
    """
    global benchmark_sets

    name_parts = f.__name__.split("_", maxsplit=-1)
    assert len(name_parts) >= 3
    assert name_parts[0] == "make" and name_parts[-1] == "benchmark"
    name = name_parts[1:-1]
    benchmark_sets["_".join(name)] = f
    return f


def getRegisteredBenchmarkSetNames() -> tuple[str, ...]:
    """Returns all benchmark set names registered so far"""
    return tuple(benchmark_sets.keys())


def getRegisteredBenchmarks(enabled: None | str | Iterable[str] = None) -> dict:
    """Finds benchmark sets matching the enabled, executes their generating functions and returns
    all obtained individual benchmarks from that, ready to call .bench()"""
    if not enabled:
        enabled = getRegisteredBenchmarkSetNames()
    if isinstance(enabled, str):
        enabled = (enabled,)

    bms = {}
    for bm_id in enabled:
        if bm_id not in benchmark_sets:
            raise ValueError(
                f"Benchmark set {bm_id} not found, available benchmark sets are: {', '.join(benchmark_sets.keys())}"
            )

        b = benchmark_sets[bm_id]()
        assert frozenset(b.keys()) not in bms, "Some benchmark ids are colliding!"
        bms.update(b)
    return bms


################################################################################
# CLI support
################################################################################


def makeArgumentParser(parser=None, *, allow_exports=True):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Benchmarks runner",
        )

    parser.add_argument(
        "benchmark_sets",
        nargs="*",
        default=None,
        help=f"Benchmark sets to run. Available: {', '.join(benchmark_sets.keys())}. Default: all.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=_g_bench_defaults["iters"],
        help=f"Iterations per repetition (default: {_g_bench_defaults['iters']})",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=_g_bench_defaults["reps"],
        help=f"Number of repetitions (default: {_g_bench_defaults['reps']})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=_g_bench_defaults["warmup"],
        help=f"Number of warmup iterations (default: {_g_bench_defaults['warmup']})",
    )
    parser.add_argument(
        "--batch_functions",
        action=argparse.BooleanOptionalAction,
        default=_g_bench_defaults["batch_functions"],
        help="If true: outer loop - benchmarks, inner loop - iterations. Otherwise reversed: "
        f"outer loop - iterations, inner loop - benchmarks. (default: {_g_bench_defaults['batch_functions']})",
    )
    parser.add_argument(
        "--randomize_iterations",
        action=argparse.BooleanOptionalAction,
        default=_g_bench_defaults["randomize_iterations"],
        help=f"Randomly shuffle benchmarks order (default: {_g_bench_defaults['randomize_iterations']})",
    )

    if allow_exports:
        parser.add_argument(
            "--export_path_pfx",
            type=str,
            default=None,
            help="If set, specifies a filename prefix for exporting console output. See `--export_*` flags below ",
        )
        parser.add_argument(
            "--export_svg",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="When --export_path_pfx is set, export console output to .svg file",
        )
        parser.add_argument(
            "--export_txt",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="When --export_path_pfx is set, export console output to .txt file",
        )
        parser.add_argument(
            "--export_results",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="When --export_path_pfx is set, export benchmark results tensor to .npy file",
        )
    return parser

from collections.abc import Iterable
from collections import namedtuple
import numpy as np
import scipy.stats
import itertools

from .common import LoggingConsole


# supported statistical methods.
kMethods = {
    "brunnermunzel": {
        "name": "Brunner Munzel test",
        "url": "https://en.wikipedia.org/wiki/Brunner_Munzel_Test",
    },
    "mannwhitneyu": {
        "name": "Mann-Whitney U test",
        "url": "https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test",
    },
}

kDefaultAlpha = 0.001

kMinStatsSize = 2
kMinReliableStatsSize = 9

kAllowedFpTypes = (float, np.floating)


class BmCompResult(
    namedtuple("BmCompResult", ["result", "pvalue", "val_set0", "val_set1", "size1", "size2"])
):
    __slots__ = ()

    def __new__(cls, res: str, pval: float, v0: float, v1: float, siz0: int, siz1: int):
        """Constructor to verify initialization correctness

        - res/result is a comparison result:
            < when set1 is stochastically less than set2,
            > when set1 is stochastically greater than set2,
            ~ when set1 is not stochastically less or greater than set2

        - pval/pvalue is a pvalue associated with less or greater comparison result, or a minimum of
            pvalues for less/greater comparison

        - v0/val_set0 and v1/val_set1 are either representative values (mean) of a corresponding
            set, or the whole set iself, depending on `store_sets` flag value of compareStats()

        siz1/size1 and siz2/size2 are sizes of respective metric value sets
        """

        # assert isinstance(rel, bool) and isinstance(pval, (float, np.floating))
        assert isinstance(pval, kAllowedFpTypes)
        assert (isinstance(v0, kAllowedFpTypes) and isinstance(v1, kAllowedFpTypes)) or (
            isinstance(v0, np.ndarray) and isinstance(v1, np.ndarray)
        )
        assert isinstance(res, str) and res in ("<", ">", "~")
        assert isinstance(siz0, int) and isinstance(siz1, int)
        if isinstance(v0, np.ndarray):
            assert siz0 == len(v0) and siz1 == len(v1)
        return super().__new__(
            cls,
            res,
            float(pval),
            v0 if isinstance(v0, np.ndarray) else float(v0),
            v1 if isinstance(v1, np.ndarray) else float(v1),
            siz0,
            siz1,
        )


class CompareStatsResult(
    namedtuple("CompareStatsResult", ["results", "method", "alpha", "at_least_one_differs"])
):
    __slots__ = ()

    def __new__(cls, res: dict[str, dict[str, BmCompResult]], met: str, al: float, one_dif: bool):
        """Constructor to verify initialization correctness.
        - res/results field is a mapping {benchmark_name -> {metric_name -> CompResult}}. Keys
            (benchmark_name's ) are common between the data sources sg0 and sg1
        """
        assert isinstance(res, dict) and isinstance(met, str) and len(met) > 0
        assert isinstance(al, kAllowedFpTypes) and isinstance(one_dif, (bool, np.bool_))
        return super().__new__(cls, res, met, float(al), bool(one_dif))

    def getMetrics(self) -> tuple[str]:
        return tuple(next(iter(self.results.values())).keys())

    def getBenchmarkNames(self) -> tuple[str]:
        return tuple(self.results.keys())

    def areAllSame(self) -> bool:
        """Tests if all benchmarks over all metrics compare same"""
        return all(["~" == cr.result for bm_res in self.results.values() for cr in bm_res.values()])

    def areMetricsSame(self, metrics: Iterable[str]) -> bool:
        """Tests if all benchmarks over specified metrics compare same"""
        return all(["~" == bm_res[m].result for bm_res in self.results.values() for m in metrics])


def poolBenchmarks(
    alt_delimiter: str,
    sg0: dict[str, dict[str, Iterable[float]]],
    sg1: dict[str, dict[str, Iterable[float]]] | None,
    logger: LoggingConsole | None = None,
) -> dict[str, dict[str, dict[str, Iterable[float]]]]:
    """Using the specified delimiter dividing the benchmark name into a common part and an
    alternative name part, merges the benchmarks from one or two sets of benchmarks into a single
    pool represented by a dictionary where a key specifies common part of benchmark name and a value
    is a dict mapping from alternative name to a dictionary of metric names and their values.

    If a benchmark name doesn't contain the delimiter, then its common name is assumed to be a whole
    benchmark name, and its alternative name is assumed to either "0" or "1" depending on the
    benchmark set (sg0 or sg1) it belongs to.
    """
    assert isinstance(alt_delimiter, str) and len(alt_delimiter) > 0
    assert isinstance(sg0, dict) and (isinstance(sg1, dict) or sg1 is None)

    def warn(*args, **kwargs):
        if logger is not None:
            logger.warning(args[0] % args[1:], **kwargs)

    def splitName(bm_name: str, pfx: str) -> tuple[str, str]:
        """Splits the benchmark name into a common part and an alternative part"""
        assert isinstance(bm_name, str)
        s = bm_name.split(sep=alt_delimiter, maxsplit=1)
        if len(s) > 1:
            alt_name = s[1].strip()
            return s[0].strip(), pfx + (
                ((alt_delimiter if pfx != "" else "") + alt_name) if len(alt_name) > 0 else ""
            )
        else:
            return bm_name, pfx

    def _do_pool(
        pool: dict[str, dict[str, dict[str, Iterable[float]]]],
        sg: dict[str, dict[str, Iterable[float]]],
        pfx: str,
    ):
        """Merges the benchmarks from one set of benchmarks into a pool"""
        for bm_name, metrics in sg.items():
            common_name, alt_name = splitName(bm_name, pfx)
            cmn_bm = pool.setdefault(common_name, {})
            if alt_name in cmn_bm:
                warn(
                    "Benchmark '%s%s' already exists in the pool. Skipping it.",
                    common_name,
                    alt_name,
                )
                continue
            cmn_bm[alt_name] = metrics

    pool = {}
    if sg1 is None:
        _do_pool(pool, sg0, "")
    else:
        _do_pool(pool, sg0, "0")
        _do_pool(pool, sg1, "1")

    return pool


def compareStats(
    sg0: dict[str, dict[str, Iterable[float]]],
    sg1: dict[str, dict[str, Iterable[float]]] | None,
    alt_delimiter: str | None = None,
    method: str = next(iter(kMethods.keys())),
    alpha: float = kDefaultAlpha,
    main_metrics: None | list[str] | tuple[str] = None,
    debug_log: None | bool | LoggingConsole = True,
    store_sets: bool = False,
    brunnermunzel_workaround: None | bool = None,
) -> CompareStatsResult:
    """Perform comparison for statistical significance of benchmark results using a chosen
    statistical method.

    Individual benchmark results (variables sg0 and sg1) are represented by a dictionary where a key
    specifies benchmark name and a value is another dictionary containing mapping from a metric name
    to an iterable of values of the metric.

    There are two ways to specify benchmark alternatives to compare against each other:

    1. Specify two different sets of benchmark results (sg0 and sg1). In that case benchmark names
    from group sg0 are directly matched to benchmark names from group sg1.

    2. Specify one or two sets of benchmark results with a benchmark alternative delimiter
    (alt_delimiter). The delimiter is used to split benchmark names into two parts: the first
    part is the common name of the benchmark, and the second part is the alternative name. All
    benchmark results with the same common name are compared against each other. For example, if
    alt_delimiter is "|", then benchmark names "foo|bar" and "foo|baz" found in sg0 will be
    compared against each other as "foo|bar vs baz", even if common name "foo" is not present in
    sg1, or sg1 is None. If sg1 also has, for example, "foo|qux" benchmark, then there'll be 3
    comparisons: "foo|0bar vs 0baz", "foo|0bar vs 1qux" and "foo|0baz vs 1qux". Numbers 0 and 1 in
    benchmark alternative names references source set of benchmark results (sg0 or sg1).

    `main_metrics` is either a list/tuple of strings describing containing main metrics for the
    purpose of computing of at_least_one_differs flag, or None (then all metrics are main)

    `debug_log` is a bool, but could also be None (==False) or a logger object like LoggingConsole.

    `store_sets` is a bool, True will make store whole corresponding dataset into a BmCompResult
    instead of a mean of the set.

    `brunnermunzel_workaround`: Formally speaking, Brunner Munzel test method is "underspecified"
    in a sense that by design it doesn't handle a case of non-intersecting sample sets, or sets
    consisting of the same single element (https://aakinshin.net/posts/brunner-munzel-corner-case/).
    scipy.stats.brunnermunzel() implementation at least in versions 1.10-1.15.2 directly follows the
    words of the original paper and doesn't introduce handling for that cases, which causes warnings
    and nans to appear in results. IMO from an engineering point of view this is more like just a
    bug (https://github.com/scipy/scipy/issues/22664). `brunnermunzel_workaround` is a flag to
    handle it: value of None handles the bug only for method == 'brunnermunzel', a bool otherwise
    controls whether to handle the bug irrespective of the method used (this might give different
    pvalues for other tests, but this shouldn't affect inference results unless alpha threshold
    value is too wild). It should be noted that for small sample sizes (under 10 elements in any of
    sample sets) this most certainly leads to a wrong p-value result that underestimate a
    probability of error. However, with 10, or preferably more, sample set sizes, this should be
    negligible.
    While brunnermunzel_workaround allows to disable the workaround, I doubt it has any practical
    value in the context of benchmark results comparison.

    Returns an instance of CompareStatsResult class
    """
    assert alt_delimiter is None or (isinstance(alt_delimiter, str) and len(alt_delimiter) > 0)
    with_alternatives = alt_delimiter is not None

    assert isinstance(sg0, dict) and (isinstance(sg1, dict) or (with_alternatives and sg1 is None))
    assert isinstance(method, str) and method in kMethods, "unsupported method"
    assert isinstance(alpha, kAllowedFpTypes) and 0 < alpha and alpha < 0.5
    assert brunnermunzel_workaround is None or isinstance(brunnermunzel_workaround, bool)
    assert main_metrics is None or isinstance(main_metrics, (list, tuple))

    if debug_log is None or (isinstance(debug_log, bool) and not debug_log):
        debug_log = False
    elif isinstance(debug_log, bool) and debug_log:
        logger = LoggingConsole(log_level=LoggingConsole.LogLevel.Debug)
    else:
        logger = debug_log
        debug_log = True

    if brunnermunzel_workaround is None:
        brunnermunzel_workaround = method == "brunnermunzel"

    def warn(*args, **kwargs):
        if debug_log:
            logger.warning(args[0] % args[1:], **kwargs)

    valid_metric_set_type = (list, tuple, np.ndarray)
    at_least_one_differs = False
    stat_func = getattr(scipy.stats, method)

    def computePValues(s0, s1):
        """Computes and return pvalues of s0 being stochastically less or greater than s1"""
        if brunnermunzel_workaround:
            # test for edge cases that brunnermunzel can't handle properly
            mn0, mx0, mn1, mx1 = np.min(s0), np.max(s0), np.min(s1), np.max(s1)
            assert np.all(np.isfinite([mn0, mx0, mn1, mx1]))  # sanity check, more asserts below
            all_eq = np.all([mn0 == mx0, mn0 == mn1, mn0 == mx1])
            all_less, all_greater = mx0 < mn1, mn0 > mx1

            if all_eq:
                return 1.0, 1.0
            elif all_less:
                return 0.0, 1.0
            elif all_greater:
                return 1.0, 0.0

        res_less = stat_func(s0, s1, alternative="less")
        res_greater = stat_func(s0, s1, alternative="greater")

        less_pvalue, greater_pvalue = res_less.pvalue, res_greater.pvalue
        if brunnermunzel_workaround:
            assert np.all(np.isfinite([less_pvalue, greater_pvalue]))

        return less_pvalue, greater_pvalue

    def compareBenchmark(bm_name: str, metrics0: dict, metrics1: dict) -> dict[str, BmCompResult]:
        """Compare two sets of metrics for a single benchmark"""
        nonlocal at_least_one_differs

        assert isinstance(bm_name, str)
        assert isinstance(metrics0, dict) and isinstance(metrics1, dict)
        assert len(metrics0) > 0 and len(metrics1) > 0

        common_metrics = metrics0.keys() & metrics1.keys()
        if len(common_metrics) != len(metrics0) or len(common_metrics) != len(metrics1):
            warn(
                "Benchmark '%s' has different metrics in set0 and set1. Metrics from set1 "
                "not found in set0 will be ignored.",
                bm_name,
            )
            if debug_log:
                logger.debug("Metrics in set0:", ", ".join(metrics0.keys()))
                logger.debug("Metrics in set1:", ", ".join(metrics1.keys()))

        bm_results = {}
        for metric_name, stats0 in metrics0.items():
            assert isinstance(metric_name, str)
            if metric_name not in metrics1:
                warn(
                    "benchmark '%s': metric '%s' not found in metrics for set1",
                    bm_name,
                    metric_name,
                )
                continue
            stats1 = metrics1[metric_name]

            assert isinstance(stats0, valid_metric_set_type)
            assert isinstance(stats1, valid_metric_set_type)
            if not isinstance(stats0, np.ndarray):
                stats0 = np.array(stats0)
            if not isinstance(stats1, np.ndarray):
                stats1 = np.array(stats1)
            assert np.ndim(stats0) == 1 and np.ndim(stats1) == 1

            size0, size1 = np.size(stats0), np.size(stats1)
            if size0 < kMinStatsSize or size1 < kMinStatsSize:
                warn(
                    "benchmark '%s': metric '%s', one of sizes (%d, %d) is less than min possible size %d",
                    bm_name,
                    metric_name,
                    size0,
                    size1,
                    kMinStatsSize,
                )
                continue

            is_reliable = size0 >= kMinReliableStatsSize and size1 >= kMinReliableStatsSize
            if not is_reliable:
                warn(
                    "benchmark '%s': metric '%s', one of sizes (%d, %d) is less than min recommended size %d. "
                    "Results might be not reliable",
                    bm_name,
                    metric_name,
                    size0,
                    size1,
                    kMinReliableStatsSize,
                )

            less_pvalue, greater_pvalue = computePValues(stats0, stats1)
            less_positive, greater_positive = less_pvalue < alpha, greater_pvalue < alpha
            if less_positive and greater_positive:
                # not sure this is a correct interpretation, but dunno what's better
                warn(
                    "benchmark '%s', metric '%s': both sides of the test indicate positive results. "
                    "Interpreting as 'no difference'",
                    bm_name,
                    metric_name,
                )
                less_positive, greater_positive = False, False

            if main_metrics is None or metric_name in main_metrics:
                at_least_one_differs = at_least_one_differs or less_positive or greater_positive

            bm_results[metric_name] = BmCompResult(
                "<" if less_positive else (">" if greater_positive else "~"),
                (
                    less_pvalue
                    if less_positive
                    else (greater_pvalue if greater_positive else min(less_pvalue, greater_pvalue))
                ),
                stats0 if store_sets else np.mean(stats0),
                stats1 if store_sets else np.mean(stats1),
                # is_reliable,
                size0,
                size1,
            )
        return bm_results

    results = {}

    if with_alternatives:
        pool = poolBenchmarks(alt_delimiter, sg0, sg1, logger if debug_log else None)

        for common_name, alternatives in pool.items():
            if len(alternatives) <= 1:
                warn(
                    "Benchmark '%s%s' has no alternatives. Skipping it.",
                    common_name,
                    next(iter(alternatives.keys())) if len(alternatives) == 1 else "???",
                )
                continue

            for altA, altB in itertools.combinations(alternatives.keys(), 2):
                assert isinstance(altA, str) and isinstance(altB, str)
                assert altA != altB
                bm_name = f"{common_name} {alt_delimiter} {altA} vs {altB}"
                assert bm_name not in results
                results[bm_name] = compareBenchmark(bm_name, alternatives[altA], alternatives[altB])

    else:
        common_bms = sg0.keys() & sg1.keys()
        if len(common_bms) != len(sg0) or len(common_bms) != len(sg1):
            warn(
                "Datasets contain different keys/benchmarks. Keys not found in both datasets will be ignored."
            )
            if debug_log:
                logger.debug("Benchmarks in set0:", ", ".join(sg0.keys()))
                logger.debug("Benchmarks in set1:", ", ".join(sg1.keys()))

        for bm_name, metrics1 in sg0.items():
            assert isinstance(bm_name, str)
            if bm_name not in sg1:
                warn("Key/benchmark name '%s' not found in set2", bm_name)
                continue
            results[bm_name] = compareBenchmark(bm_name, metrics1, sg1[bm_name])

    return CompareStatsResult(results, method, alpha, bool(at_least_one_differs))

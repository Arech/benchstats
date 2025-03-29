from collections.abc import Iterable
from collections import namedtuple
import numpy as np
import scipy.stats
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
    namedtuple("BmCompResult", ["result", "pvalue", "val_set1", "val_set2", "size1", "size2"])
):
    __slots__ = ()

    def __new__(cls, res: str, pval: float, v1: float, v2: float, siz1: int, siz2: int):
        """Constructor to verify initialization correctness

        - res/result is a comparison result:
            < when set1 is stochastically less than set2,
            > when set1 is stochastically greater than set2,
            ~ when set1 is not stochastically less or greater than set2
        
        - pval/pvalue is a pvalue associated with less or greater comparison result, or a minimum of
            pvalues for less/greater comparison
        
        - v1/val_set1 and v2/val_set2 are either representative values (mean) of a corresponding
            set, or the whole set iself, depending on `store_sets` flag value of compareStats()

        siz1/size1 and siz2/size2 are sizes of respective metric value sets
        """

        # assert isinstance(rel, bool) and isinstance(pval, (float, np.floating))
        assert isinstance(pval, kAllowedFpTypes)
        assert (isinstance(v1, kAllowedFpTypes) and isinstance(v2, kAllowedFpTypes)) or (
            isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray)
        )
        assert isinstance(res, str) and res in ("<", ">", "~")
        assert isinstance(siz1, int) and isinstance(siz2, int)
        if isinstance(v1, np.ndarray):
            assert siz1 == len(v1) and siz2 == len(v2)
        return super().__new__(
            cls,
            res,
            float(pval),
            v1 if isinstance(v1, np.ndarray) else float(v1),
            v2 if isinstance(v2, np.ndarray) else float(v2),
            siz1,
            siz2,
        )


class CompareStatsResult(namedtuple('CompareStatsResult',['results', 'method', 'alpha', 'at_least_one_differs'])):
    __slots__ = ()

    def __new__(cls, res: dict[str, dict[str, BmCompResult]], met: str, al: float, one_dif: bool):
        """Constructor to verify initialization correctness.
        - res/results field is a mapping {benchmark_name -> {metric_name -> CompResult}}. Keys
            (benchmark_name's ) are common between the data sources sg1 and sg2
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


def compareStats(
    sg1: dict[str, dict[str, Iterable[float]]],
    sg2: dict[str, dict[str, Iterable[float]]],
    method: str = next(iter(kMethods.keys())),
    alpha: float = kDefaultAlpha,
    main_metrics: None | list[str] | tuple[str] = None,
    debug_log: None | bool | LoggingConsole = True,
    store_sets: bool = False,
    scipy_bug_workaround: None | bool = None,
) -> CompareStatsResult:
    """Perform comparison for statistical significance between two groups of sets of statistics
    using specific statistical method.

    Each group is represented by a dictionary where key specifies a benchmark name and value
    is another dictionary metric_name->iterable_of_metric_values.

    `main_metrics` is either a list/tuple of strings describing containing main metrics for the
    purpose of computing of at_least_one_differs flag, or None (then all metrics are main)

    `debug_log` is a bool, but could also be None (==False) or a logger object like LoggingConsole.

    `store_sets` is a bool, True will make store whole corresponding dataset into a BmCompResult
    instead of a mean of the set.

    `scipy_bug_workaround`: scipy.stats.brunnermunzel() method at least in versions 1.10-1.15.2 has
    what I consider to be a bug (https://github.com/scipy/scipy/issues/22664) that causes warnings
    and nans in results to appear for nonintersecting data sets (i.e. when all elements of one set
    is strictly less or greater than all elements of the other set) or exactly equal sets.
    scipy.stats.mannwhitneyu() on the contrary behaves absolutely correctly.
    scipy_bug_workaround is a flag to handle the bug: value of None handles the bug only for
    method == 'brunnermunzel', a bool otherwise controls whether to handle the bug irrespective of
    the method used (this might give different pvalues for other tests, but this shouldn't affect
    inference results unless alpha threshold value is too wild)

    Returns an instance of CompareStatsResult class
    """
    assert isinstance(sg1, dict) and isinstance(sg2, dict)
    assert isinstance(method, str) and method in kMethods, "unsupported method"
    assert isinstance(alpha, kAllowedFpTypes) and 0 < alpha and alpha < 0.5
    assert scipy_bug_workaround is None or isinstance(scipy_bug_workaround, bool)
    assert main_metrics is None or isinstance(main_metrics, (list, tuple))

    if debug_log is None or (isinstance(debug_log, bool) and not debug_log):
        debug_log = False
    elif isinstance(debug_log, bool) and debug_log:
        logger = LoggingConsole(log_level=LoggingConsole.LogLevel.Debug)
    else:
        logger = debug_log
        debug_log = True

    if scipy_bug_workaround is None:
        scipy_bug_workaround = method == "brunnermunzel"

    def warn(*args, **kwargs):
        if debug_log:
            logger.warning(args[0] % args[1:], **kwargs)

    """if debug_log:
        logger.debug(
            "Comparing datasets with %s and alpha=%.6f, use_scipy_bug_workaround=%s"
            % (kMethods[method]["name"], alpha, scipy_bug_workaround)
        )"""

    common_bms = sg1.keys() & sg2.keys()
    if len(common_bms) != len(sg1) or len(common_bms) != len(sg2):
        warn(
            "Datasets contain different keys/benchmarks. Keys not found in both datasets will be ignored."
        )
        if debug_log:
            logger.debug("Benchmarks in set1:", ", ".join(sg1.keys()))
            logger.debug("Benchmarks in set2:", ", ".join(sg2.keys()))

    valid_metric_set_type = (list, tuple, np.ndarray)
    at_least_one_differs = False
    stat_func = getattr(scipy.stats, method)

    def computePValues(s1, s2):
        """Computes and return pvalues of s1 being stochastically less or greater than s2"""
        if scipy_bug_workaround:
            # testing for known (!!!) cases that brunnermunzel can't handle properly
            mn1, mx1, mn2, mx2 = np.min(s1), np.max(s1), np.min(s2), np.max(s2)
            assert np.all(np.isfinite([mn1, mx1, mn2, mx2]))  # sanity check, more asserts below
            all_eq = np.all([mn1 == mx1, mn1 == mn2, mn1 == mx2])
            all_less, all_greater = mx1 < mn2, mn1 > mx2

            if all_eq:
                return 1.0, 1.0
            elif all_less:
                return 0.0, 1.0
            elif all_greater:
                return 1.0, 0.0

        res_less = stat_func(s1, s2, alternative="less")
        res_greater = stat_func(s1, s2, alternative="greater")

        less_pvalue, greater_pvalue = res_less.pvalue, res_greater.pvalue
        if scipy_bug_workaround:
            assert np.all(np.isfinite([less_pvalue, greater_pvalue]))

        return less_pvalue, greater_pvalue

    results = {}

    for bm_name, metrics1 in sg1.items():
        assert isinstance(bm_name, str)
        if bm_name not in sg2:
            warn("Key/benchmark name '%s' not found in set2", bm_name)
            continue
        metrics2 = sg2[bm_name]
        assert isinstance(metrics1, dict) and isinstance(metrics2, dict)

        bm_results = {}
        for metric_name, stats1 in metrics1.items():
            assert isinstance(metric_name, str)
            if metric_name not in metrics2:
                warn(
                    "benchmark '%s', metric '%s' not found in metrics for set2",
                    bm_name,
                    metric_name,
                )
                continue
            stats2 = metrics2[metric_name]
            assert isinstance(stats1, valid_metric_set_type) and isinstance(
                stats2, valid_metric_set_type
            )
            if not isinstance(stats1, np.ndarray):
                stats1 = np.array(stats1)
            if not isinstance(stats2, np.ndarray):
                stats2 = np.array(stats2)
            assert np.ndim(stats1) == 1 and np.ndim(stats2) == 1

            size1, size2 = np.size(stats1), np.size(stats2)
            if size1 < kMinStatsSize or size2 < kMinStatsSize:
                warn(
                    "benchmark '%s', metric '%s', one of sizes (%d, %d) is less than min possible size %d",
                    bm_name,
                    metric_name,
                    size1,
                    size2,
                    kMinStatsSize,
                )
                continue

            is_reliable = size1 >= kMinReliableStatsSize and size2 >= kMinReliableStatsSize
            if not is_reliable:
                warn(
                    "benchmark '%s', metric '%s', one of sizes (%d, %d) is less than min recommended size %d. "
                    "Results might be not reliable",
                    bm_name,
                    metric_name,
                    size1,
                    size2,
                    kMinReliableStatsSize,
                )

            less_pvalue, greater_pvalue = computePValues(stats1, stats2)
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
                stats1 if store_sets else np.mean(stats1),
                stats2 if store_sets else np.mean(stats2),
                # is_reliable,
                size1,
                size2,
            )
        results[bm_name] = bm_results
    return CompareStatsResult(results, method, alpha, bool(at_least_one_differs))

import unittest
import benchstats.compare as bsc
import numpy as np

kSize = bsc.kMinReliableStatsSize
kMethods = tuple(bsc.kMethods.keys())
kAlpha = bsc.kDefaultAlpha


class TestCompareStatsResult(unittest.TestCase):
    def test_updatePvalStats(self):
        defargs = (kMethods[0], kAlpha, True)
        cs0 = bsc.CompareStatsResult(
            {
                "b1": {
                    "m1": bsc.BmCompResult("<", 0.01, 1, 2, kSize, kSize),
                    "m2": bsc.BmCompResult("<", 0.02, 1, 2, kSize, kSize),
                },
                "b2": {"m1": bsc.BmCompResult("~", 0.3, 2, 2, kSize, kSize)},
            },
            *defargs,
        )
        cs1 = bsc.CompareStatsResult(
            {
                "b1": {
                    "m1": bsc.BmCompResult("<", 0.02, 1, 2, kSize, kSize),
                    "m2": bsc.BmCompResult("~", 0.2, 1.9, 2, kSize, kSize),
                },
                "b2": {"m1": bsc.BmCompResult(">", 0.03, 2, 1, kSize, kSize)},
            },
            *defargs,
        )

        self.assertEqual(cs0.pval_stats_available, False)
        cs0.updatePvalStats(cs1)
        self.assertEqual(cs0.pval_stats_available, True)
        self.assertEqual(
            cs0.pval_stats,
            {
                "b1": {
                    "m1": {"<": [0.01, 0.02], ">": [], "~": []},
                    "m2": {"<": [0.02], ">": [], "~": [0.2]},
                },
                "b2": {"m1": {"<": [], "~": [0.3], ">": [0.03]}},
            },
        )


class TestComparisonMethods(unittest.TestCase):
    def _assertCommonExpectations(self, compareStats_result):
        res = compareStats_result.results
        self.assertEqual(1, len(res))
        metrics_res = res["b"]
        self.assertEqual(1, len(metrics_res))
        comp_res = metrics_res["m"]
        self.assertIsInstance(comp_res, bsc.BmCompResult)
        self.assertEqual(kSize, comp_res.size1)
        self.assertEqual(kSize, comp_res.size2)

    def _do_test_compare(self, s1, s2, exp_result):
        for method in kMethods:
            cs = bsc.compareStats(
                {"b": {"m": s1}}, {"b": {"m": s2}}, method=method, alpha=kAlpha, debug_log=False
            )
            self._assertCommonExpectations(cs)
            comp_res = cs.results["b"]["m"]
            self.assertEqual(exp_result, comp_res.result)
            if "~" == exp_result:
                self.assertGreaterEqual(comp_res.pvalue, kAlpha)
            else:
                self.assertLess(comp_res.pvalue, kAlpha)

    def test_compare_less(self):
        a, b = np.zeros((kSize,)), np.ones((kSize,))
        self._do_test_compare(a, b, "<")
        self._do_test_compare(b, a, ">")

    def test_compare_less2(self):
        a, b = np.zeros((kSize,)), np.ones((kSize,))
        a[0] = 1
        self._do_test_compare(a, b, "<")
        self._do_test_compare(b, a, ">")

    def test_compare_greater(self):
        a, b = np.arange(0, kSize), np.arange(-kSize - 1, -1)
        self._do_test_compare(a, b, ">")
        self._do_test_compare(b, a, "<")

    def test_compare_same(self):
        self._do_test_compare(np.zeros((kSize,)), np.zeros((kSize,)), "~")

    def test_compare_same_diff_order(self):
        a = np.zeros((kSize,))
        a[0] = 1
        b = np.zeros((kSize,))
        b[1] = 1
        self._do_test_compare(a, b, "~")
        self._do_test_compare(b, a, "~")

    def test_compare_same_diff_order2(self):
        a, b = np.arange(0, kSize), np.arange(kSize - 1, -1, -1)
        self._do_test_compare(a, b, "~")
        self._do_test_compare(b, a, "~")

    def test_compare_almost_same(self):
        a, b = np.arange(0, kSize), np.arange(1, kSize + 1)
        self._do_test_compare(a, b, "~")
        self._do_test_compare(b, a, "~")


if __name__ == "__main__":
    unittest.main()

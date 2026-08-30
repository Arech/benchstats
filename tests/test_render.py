"""These are likely will be mostly manual tests"""

import unittest
import numpy as np
import time
import traceback

from benchstats.compare import compareStats
from benchstats.render import renderComparisonResults


class TestRenderMANUAL(unittest.TestCase):
    def test_output(self):
        # we need to output grid of several main and secondary metric results
        n_samples = 10
        small = np.ones(n_samples)
        large = 2 * np.ones(n_samples)

        def _makeData(res):
            if "<" == res:
                return small, large
            elif ">" == res:
                return large, small
            assert "~" == res
            return small, small

        s1, s2 = {}, {}
        possible_results = ["<", ">", "~"]
        for main1 in possible_results:
            m11, m12 = _makeData(main1)
            for main2 in possible_results:
                m21, m22 = _makeData(main2)
                for scnd1 in possible_results:
                    r11, r12 = _makeData(scnd1)
                    for scnd2 in possible_results:
                        name = f"m1_{main1} m2_{main2} s1_{scnd1} s2_{scnd2}"
                        r21, r22 = _makeData(scnd2)
                        s1[name] = {"main1": m11, "scnd1": r11, "main2": m21, "scnd2": r21}
                        s2[name] = {"main1": m12, "scnd1": r12, "main2": m22, "scnd2": r22}

        main_metrics = ["main1", "main2"]
        cr = compareStats(s1, s2, main_metrics=main_metrics)
        renderComparisonResults(cr, main_metrics=main_metrics, drop_pvalues=True)


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv + ["-vs"]))

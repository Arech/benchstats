"""These are likely will be mostly manual tests"""

import unittest
import numpy as np
import time
import traceback

from benchstats.compare import compareStats
from benchstats.render import renderComparisonResults


class TestRenderMANUAL(unittest.TestCase):
    def test_output(self):
        # we need to output grid:
        # [main: <, >, ~] x [scnd: <, >, ~]
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
        for main in ["<", ">", "~"]:
            m1, m2 = _makeData(main)
            for scnd in ["<", ">", "~"]:
                name = f"main_{main} scnd_{scnd}"
                r1, r2 = _makeData(scnd)
                s1[name] = {"main": m1, "scnd": r1}
                s2[name] = {"main": m2, "scnd": r2}

        cr = compareStats(s1, s2, main_metrics="main")
        renderComparisonResults(cr, main_metrics="main")


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv + ["-vs"]))

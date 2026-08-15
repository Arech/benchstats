"""
The simplest parser ever for single column CSV files.
"""

import numpy as np
from benchstats.common import ParserBase

class parser_SingleColumnCSV(ParserBase):
    def __init__(self, fpath, filter, metrics, debug_log=None) -> None:
        # note that `filter` value is passed from `--filter1` as it is, so one
        # can treat it wider, as an arbitrary user-controllable parametrization
        # of the parser
        self.stats = np.loadtxt(fpath, dtype=np.float64)

    def getStats(self) -> dict[str, dict[str, np.ndarray]]:
        return {"bm": {"real_time": self.stats}}
        # the outer dictionary define benchmark_name -> data mapping, while
        # the inner data dictionary maps metric_name -> 1d_array of numbers to
        # compare. There can be as many benchmarks and metrics as needed.
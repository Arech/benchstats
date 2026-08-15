"""
This is a test parser for the benchstats CLI. It "reads" a file containing / generates
benchmark results which are tested against the other "file"
"""

import numpy as np
from benchstats.common import ParserBase


class parser_two(ParserBase):
    def __init__(self, fpath, filter, metrics, debug_log=None) -> None:
        assert filter in ("1", "2")
        assert len(fpath) == len("path/to/filex")
        assert fpath.startswith("path/to/file") and fpath.endswith(filter)
        assert metrics == ["real_time"]
        assert debug_log is not None

        is_1 = filter == "1"
        ofs = 0 if is_1 else 0.1
        bm1 = np.random.default_rng().uniform(ofs, 1.0 + ofs, size=1000)
        bm2 = np.random.default_rng().uniform(0.0, 1.0, size=1000)

        self.stats = {"bm1": {"real_time": bm1}, "bm2": {"real_time": bm2}}

    def getStats(self) -> dict[str, dict[str, np.ndarray]]:
        return self.stats

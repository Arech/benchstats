"""
This is a test parser for the benchstats CLI. It "reads"/generates all contending benchmarks
from a single data source
"""

import numpy as np
from benchstats.common import ParserBase


class parser_single(ParserBase):
    def __init__(self, fpath, filter, metrics, debug_log=None) -> None:
        assert filter is None
        assert fpath == "path/to/file"
        assert metrics == ["real_time"]
        assert debug_log is not None

        def _gen(ofs=0.0):
            return np.random.default_rng().uniform(ofs, 1.0 + ofs, size=1000)

        self.stats = {
            "bm1|var1": {"real_time": _gen()},
            "bm1|var2": {"real_time": _gen(0.1)},
            "bm2|opt1": {"real_time": _gen()},
            "bm2|opt2": {"real_time": _gen()},
            "bm2|opt3": {"real_time": _gen()},
        }

    def getStats(self) -> dict[str, dict[str, np.ndarray]]:
        return self.stats

    def getAltDelimiter(self) -> str | None:
        return "|"

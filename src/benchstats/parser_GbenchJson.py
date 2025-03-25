import json
import numpy as np
import re

# also must use abs package path to be dynamically loadable
from benchstats.common import ParserBase, LoggingConsole


class parser_GbenchJson(ParserBase):
    _kTimeUnitToSecondsMul = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
    }

    # metrics that require application of _kTimeUnitToSecondsMul. User counters are written as is and don't require
    # translation.
    _kApplyTimeTranslationTo = ("real_time", "cpu_time")

    def __init__(self, json_file_path, filter, metrics, debug_log=True) -> None:
        """
        - json_file_path is a file path to json file generated with `--benchmark_out=` argument
            of Google Benchmark binary.
        - filter is None or a string with a regular expression to match against a benchmark name.
        - metrics is a list of string representing metrics to extract for each benchmark repetition.
        - debug_log is a flag to enable/disable logging, or a logger object like LoggingConsole to
            send logs to
        """
        assert isinstance(json_file_path, str)
        assert filter is None or isinstance(filter, (str, re.Pattern))
        assert isinstance(metrics, (list, tuple)) and len(metrics) > 0
        assert all([isinstance(m, str) for m in metrics])

        if debug_log is None or (isinstance(debug_log, bool) and not debug_log):
            self.debug_log = False
        elif isinstance(debug_log, bool) and debug_log:
            self.debug_log = True
            self.logger = LoggingConsole(log_level=LoggingConsole.LogLevel.Debug)
        else:
            self.debug_log = True
            self.logger = debug_log

        self.file = json_file_path
        self.filter = filter

        # if self.debug_log:
        #    self.logger.debug("%s: Loading '%s'" % (self.__class__.__name__, json_file_path))

        raw_bms = self._getRawBenchmarksGrouped(self._load())
        # self._iterationData = self._getRawBmsIterations(raw_bms)
        self._stats = self._getRawBmsStats(raw_bms, metrics)

    def getStats(self) -> dict[str, dict[str, np.ndarray]]:
        """Return a dict that describes benchmarks and its measured statistics so it and can be
        directly passed as any of the first two arguments to compare::compareStats(). See that
        method for the details."""
        return self._stats

    def _getRawBmsStats(self, raw_bms, metrics) -> dict[str, dict[str, np.ndarray]]:
        """Turns raw benchmarks dictionary into a dict with statistics to pass to
        compare::compareStats()."""
        return {
            bm_name: {
                m: np.array(
                    [
                        b[m]
                        * (
                            self._kTimeUnitToSecondsMul.get(b["time_unit"], 1.0)
                            if m in self._kApplyTimeTranslationTo
                            else 1.0
                        )
                        for b in bms
                    ],
                    dtype=np.float64,  # no point in bigger dtype, as internally it's doubles
                )
                for m in metrics
            }
            for bm_name, bms in raw_bms.items()
        }

    def _getRawBmsIterations(self, raw_bms) -> dict[str, int]:
        """Given a dict of raw benchmarks, returns a dict bm_name->number of iterations while
        controlling that each benchmark has the same amount of iterations."""
        ret = {}
        for bm_name, ind_bms in raw_bms.items():
            iters = ind_bms[0]["iterations"]
            if self.debug_log and not all([iters == b["iterations"] for b in ind_bms]):
                self.logger.error(
                    "%s: Not all repetitions of the same benchmark '%s' have the same number of iterations %d! "
                    "Report of iterations number will be incorrect for the benchmark!"
                    % (self.__class__.__name__, bm_name, iters)
                )
            ret[bm_name] = iters
        return ret

    @staticmethod
    def _getRawBenchmarksGrouped(obj):
        """Returns a dict of benchmark_name -> list of raw (not aggregate) benchmarks from the
        obj json"""
        ret = {}
        for raw in filter(lambda b: b["run_type"] == "iteration", obj["benchmarks"]):
            n = raw["name"]
            if n in ret:
                ret[n].append(raw)
            else:
                ret[n] = [raw]
        return ret

    def _load(self):
        """Loads json file into returned object with filtering, with benchmarks sorted by name
        Inspired by https://github.com/google/benchmark/blob/c58e6d0710581e3a08d65c349664128a8d9a2461/tools/gbench/util.py#L117
        """
        no_filter = self.filter is None
        if not no_filter:
            re_filter = (
                self.filter if isinstance(self.filter, re.Pattern) else re.compile(self.filter)
            )

        def take_bench(benchmark):
            if no_filter:
                return True
            name = benchmark.get("run_name", None) or benchmark["name"]
            return re_filter.search(name) is not None

        def logError(obj_name):
            if self.debug_log:
                self.logger.error(
                    "%s: didn't find '%s' object when reading file '%s'. "
                    "Will try to proceed, but no guarantees..."
                    % (self.__class__.__name__, obj_name, self.file)
                )

        def sortBms(bms):
            return sorted(
                sorted(
                    sorted(
                        sorted(
                            bms,
                            key=lambda b: b["repetition_index"] if "repetition_index" in b else -1,
                        ),
                        key=lambda b: (
                            1 if "run_type" in b and b["run_type"] == "aggregate" else 0
                        ),
                    ),
                    key=lambda b: (
                        b["per_family_instance_index"] if "per_family_instance_index" in b else -1
                    ),
                ),
                key=lambda b: (b["family_index"] if "family_index" in b else -1),
            )

        with open(self.file, "r") as f:
            results = json.load(f)
            if "context" in results:
                if "json_schema_version" in results["context"]:
                    json_schema_version = results["context"]["json_schema_version"]
                    if json_schema_version != 1 and self.debug_log:
                        self.logger.error(
                            "%s: unexpected schema version %s found when reading file '%s'. "
                            "Only version 1 is supported by now. "
                            "Will try to proceed, but no guarantees..."
                            % (self.__class__.__name__, str(json_schema_version), self.file)
                        )
                else:
                    logError("context::json_schema_version")
            else:
                logError("context")

            if "benchmarks" not in results:
                if self.debug_log:
                    self.logger.failure(
                        "%s: didn't find 'benchmarks' object when reading file '%s'. "
                        "No data to read!" % (self.__class__.__name__, self.file)
                    )
                raise RuntimeError(f"No data to read from '{self.file}'")

            results["benchmarks"] = sortBms(filter(take_bench, results["benchmarks"]))
            return results

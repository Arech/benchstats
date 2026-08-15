"""Describes a base class for a source of statistics capable of producing (from a file or whatnot)
data in format suitable for calling compare::compareStats()
"""

import os
import re
import rich.console
from collections.abc import Iterable
import enum

kAvailableFormats = ("txt", "svg", "html")


class ParserBase:
    def __init__(self, source_id, filter, metrics, debug_log=None) -> None:
        """Expected source constructor.
        - source_id is an identifier of a data source that the class knows how to process and turn
            into inputs of compare::compareStats(). Data type is derived class implementation
            dependent.
            Typically it's a string referencing a file to read and parse.
        - filter is an object describing how to filter data source when not everything from it
            is needed.
            Typically, it's a string with a regular expression to match against a benchmark name.
        - metrics is a list of string representing metrics to extract for each benchmark repetition.
        - debug_log is a flag to enable/disable logging, or a logger object like LoggingConsole to
            send logs to. When the parser is called by benchstats CLI handler, it's always an
            instance of LoggingConsole, so support for other variants (None or bool) are needed only
            for built-in parsers to alleviate use of the package components in a source code.
        """
        pass  # derived class knows what to do

    def getStats(self) -> dict[str, dict[str, Iterable[float]]]:
        """Return a dict that describes benchmarks and its measured statistics so it and can be
        directly passed as any of the first two arguments to compare::compareStats(). See that
        method for the details.
        """
        raise RuntimeError("DERIVED CLASS MUST IMPLEMENT METHOD")

    def getAltDelimiter(self) -> str | None:
        """Sets comparison mode and returns benchmark <name>|<variant> delimiter.

        If a parser class returns None from this method, this sets two sources
        comparison mode, i.e. it makes benchstats CLI tool expect that user supplies file1
        and file2, each containing data_set1 or data_set2 data respectively. The tool
        must compare data from data_set1 to a matching data in data_set2.

        If a parser class returns a string from this method, this sets a single source
        comparison method, where the string defines a delimiter substring in a benchmark name
        that separates the benchmark identifier from the variant identifier. For details,
        see the docstring of benchstats.compare.compareStats(), especially a description of
        the alt_delimiter= argument.
        Note that such types of a parser is only supported for `--file_parser` of `--file1_parser`
        CLI arguments.
        """
        return None


def detectExportFormat(export_to, export_fmt):
    assert (export_to is None and export_fmt is None) or (
        isinstance(export_to, str) and len(export_to) > 0
    )
    assert export_fmt is None or export_fmt in kAvailableFormats

    if export_to is not None and export_fmt is None:
        root, ext = os.path.splitext(export_to)
        assert ext in ["." + e for e in kAvailableFormats], (
            f"Unrecognized export file extension '{ext}' of a file in --export_to parameter"
        )
        export_fmt = ext[1:]

    return export_fmt


class LoggingConsole(rich.console.Console):
    # @enum.verify(enum.CONTINUOUS)  # not supported by Py 3.10
    class LogLevel(enum.IntEnum):
        Trace = 0
        Debug = 1
        Info = 2
        Warning = 3
        Error = 4  # recoverable
        Failure = 5  # non-recoverable, can continue working
        Critical = 6  # non-recoverable, must abort

    def __init__(self, log_level: LogLevel = LogLevel.Debug, **kwargs):
        assert isinstance(log_level, LoggingConsole.LogLevel)
        self.log_level = log_level
        self._n_errors: int = 0
        if "emoji" not in kwargs:
            kwargs["emoji"] = False
        if "highlight" not in kwargs:
            kwargs["highlight"] = False
        super().__init__(**kwargs)

    def cleanNumErrors(self) -> int:
        r = self._n_errors
        self._n_errors = 0
        return r

    def getNumErrors(self) -> int:
        return self._n_errors

    def _do_log(self, color: str, lvl: str, *args, **kwargs):
        if "sep" in kwargs:
            sep = kwargs["sep"] if len(kwargs["sep"]) > 0 else " "
        else:
            sep = " "
            kwargs["sep"] = sep
        return super().print(f"[[{color}]{lvl:4s}[/{color}]]{sep}", *args, **kwargs)

    def will_log(self, level) -> bool:
        return self.log_level <= level

    def trace(self, *args, **kwargs):
        if self.log_level > LoggingConsole.LogLevel.Trace:
            return None
        return self._do_log("blue", "trce", *args, **kwargs)

    def debug(self, *args, **kwargs):
        if self.log_level > LoggingConsole.LogLevel.Debug:
            return None
        return self._do_log("bright_black", "dbg", *args, **kwargs)

    def info(self, *args, **kwargs):
        if self.log_level > LoggingConsole.LogLevel.Info:
            return None
        return self._do_log("bright_white", "info", *args, **kwargs)

    def warning(self, *args, **kwargs):
        if self.log_level > LoggingConsole.LogLevel.Warning:
            return None
        return self._do_log("yellow", "warn", *args, **kwargs)

    def error(self, *args, **kwargs):
        self._n_errors += 1
        if self.log_level > LoggingConsole.LogLevel.Error:
            return None
        return self._do_log("red", "Err", *args, **kwargs)

    def failure(self, *args, **kwargs):
        self._n_errors += 1
        if self.log_level > LoggingConsole.LogLevel.Failure:
            return None
        return self._do_log("bright_red", "FAIL", *args, **kwargs)

    def critical(self, *args, **kwargs):
        self._n_errors += 1
        if self.log_level > LoggingConsole.LogLevel.Critical:
            return None
        return self._do_log("bright_magenta", "CRIT", *args, **kwargs)


def bmNamesTransform(
    stats: dict[str, dict[str, Iterable[float]]],
    re_from: str | None,
    re_to: str | None,
    set_idx: int,
    console: LoggingConsole,
) -> dict[str, dict[str, Iterable[float]]]:
    assert isinstance(stats, dict) and isinstance(set_idx, int)
    assert isinstance(console, LoggingConsole)
    if re_from is None:
        # not nice that it glues func args with CLI args, but ok to simplify err handling
        assert re_to is None, (
            f"--to{set_idx} can only be used when there's a corresponding --from{set_idx}"
        )
        return stats
    assert isinstance(re_from, str) and len(re_from) > 0
    if re_to is None:
        re_to = ""
    else:
        assert isinstance(re_to, str)

    r = re.compile(re_from)

    old_bms = stats.keys()
    new_bms = [r.subn(re_to, bm_name)[0] for bm_name in old_bms]
    unique_len = len(frozenset(new_bms))
    if unique_len != len(new_bms):
        console.warning(
            f"--from{set_idx} -> --to{set_idx} regexp '{re_from}'->'{re_to}' is a narrowing conversion "
            f"generates only {unique_len} unique name from {len(new_bms)}. This most likely"
            " isn't what you want, since the order of narrowing is not specified and you might end"
            " up having a wrong data for some new benchmark name."
        )
        console.debug("Old names are:", ", ".join(old_bms))
        console.debug("New names are:", ", ".join(new_bms))
    else:
        assert unique_len == len(stats)

    return {r.subn(re_to, bm_name)[0]: bm_metrics for bm_name, bm_metrics in stats.items()}

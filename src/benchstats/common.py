"""Describes a base class for a source of statistics capable of producing (from a file or whatnot)
data in format suitable for calling compare::compareStats()
"""

import logging
import structlog
from collections.abc import Iterable


class ParserBase:
    def __init__(self, source_id, filter, metrics, debug_log=True) -> None:
        """Expected source constructor.
        - source_id is an identifier of a data source that the class knows how to process and turn
            into inputs of compare::compareStats(). Data type is derived class implementation
            dependent.
            Typically it's a string referencing a file to read and parse.
        - filter is an object describing how to filter data source when not everything from it
            is needed.
            Typically, it's a string with a regular expression to match against a benchmark name.
        - metrics is a list of string representing metrics to extract for each benchmark repetition.
        - debug_log is a flag to enable/disable logging, or a standard logger object to send logs to
        """
        pass  # derived class knows what to do

    def getStats(self) -> dict[str, dict[str, Iterable[float]]]:
        """Return a dict that describes benchmarks and its measured statistics so it and can be
        directly passed as any of the first two arguments to compare::compareStats(). See that
        method for the details.
        """
        raise RuntimeError("DERIVED CLASS MUST IMPLEMENT METHOD")


def getLogger():
    if not structlog.is_configured():
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                # structlog.processors.StackInfoRenderer(),
                # structlog.dev.set_exc_info,
                # structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=False,
        )
    return structlog.get_logger()

"""Parsers management"""

import os
import importlib.util
import sys
from glob import glob
from benchstats.common import ParserBase

_kThisDir = os.path.dirname(__file__)
_kPfx = "parser_"


def _getBuiltinParserFiles() -> list[str]:
    return glob(os.path.join(_kThisDir, f"{_kPfx}*.py"))


def _filepath2ParserId(fpath: str) -> str:
    return os.path.basename(fpath)[len(_kPfx) : -3]


def getBuiltinParsers() -> list[str]:
    return [_filepath2ParserId(f) for f in _getBuiltinParserFiles()]


def _getBuiltinParserFileFor(parser_id: str) -> str | None:
    """returns a path to a builtin parser with given ID, or None if no such parser found"""
    parser_id = parser_id.lower()
    for pfile in _getBuiltinParserFiles():
        if _filepath2ParserId(pfile).lower() == parser_id:
            return pfile
    return None


def _loadParserFrom(fpath: str):
    module_name = os.path.splitext(os.path.basename(fpath))[0]
    spec = importlib.util.spec_from_file_location(module_name, fpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    parser = getattr(module, module_name)
    assert issubclass(parser, ParserBase), "Parsers must derive from benchstats.common.ParserBase"
    return parser


def getParserFor(id_or_filepath: str):
    """Returns a class object corresponding to a given parser identifier (if there's such built in
    parser) or to a parser class loaded from the given file path"""
    assert isinstance(id_or_filepath, str) and len(id_or_filepath) > 0

    # first always test built-in parsers. For them we could compare ignoring case
    builtin_path = _getBuiltinParserFileFor(id_or_filepath)
    return _loadParserFrom(id_or_filepath if builtin_path is None else builtin_path)

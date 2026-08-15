"""Parsers management"""

import os
import importlib
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


def _loadParserFromFile(fpath: str):
    module_name = os.path.splitext(os.path.basename(fpath))[0]
    spec = importlib.util.spec_from_file_location(module_name, fpath)
    if spec is None:
        raise ValueError(f"Can't read parser from file '{fpath}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, module_name):
        raise ValueError(
            f"Parser file '{fpath}' must define `class {module_name}(ParserBase)` with parser implementation."
        )

    parser = getattr(module, module_name)
    if not isinstance(parser, type) or not issubclass(parser, ParserBase):
        raise ValueError("Parsers must derive from benchstats.common.ParserBase")

    return parser


def _loadParserFromImportPath(import_path: str):
    """Load a parser class from a dotted import path, e.g. 'benchstats.parser_SingleColumnCSV'."""
    assert isinstance(import_path, str) and len(import_path) > 0
    if "." not in import_path:
        raise ValueError(
            f"Import path '{import_path}' must be a dotted module path (e.g. 'benchstats.parser_SingleColumnCSV')"
        )

    try:
        module = importlib.import_module(import_path)
    except ImportError as e:
        raise ValueError(f"Can't import parser module '{import_path}'") from e

    class_name = import_path.rsplit(".", 1)[-1]
    if not hasattr(module, class_name):
        raise ValueError(
            f"Parser module '{import_path}' must define `class {class_name}(ParserBase)` with parser implementation."
        )

    parser = getattr(module, class_name)
    if not isinstance(parser, type) or not issubclass(parser, ParserBase):
        raise ValueError("Parsers must derive from benchstats.common.ParserBase")

    return parser


def getParserFor(id_or_filepath: str):
    """Returns a class object corresponding to a given parser identifier (if there's such built in
    parser) or to a parser class loaded from the given file path"""
    assert isinstance(id_or_filepath, str) and len(id_or_filepath) > 0

    # first always test built-in parsers. For them we could compare ignoring case
    builtin_path = _getBuiltinParserFileFor(id_or_filepath)

    if builtin_path is None and not os.path.isfile(id_or_filepath):
        # not a built in and not a file. Try to import as module
        if "." in id_or_filepath and not id_or_filepath.endswith(".py"):
            return _loadParserFromImportPath(id_or_filepath)

        raise ValueError(f"Can't load parser from a non-existing file '{id_or_filepath}'")

    return _loadParserFromFile(id_or_filepath if builtin_path is None else builtin_path)

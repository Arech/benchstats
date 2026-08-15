import os
import subprocess
import unittest
import numpy as np
import time
import traceback


_this_file_dir = os.path.dirname(os.path.realpath(__file__))


def run(args: list[str]) -> str:
    return subprocess.run(["python3", "-m", "benchstats"] + args, capture_output=True, text=True)


class TestMain(unittest.TestCase):
    def test_two_sources_mode(self):
        args = [
            "path/to/file1",
            "path/to/file2",
            "--filter1",
            "1",
            "--filter2",
            "2",
            "--files_parser",
            os.path.join(_this_file_dir, "parser_two.py"),
        ]
        res = run(args)
        # print(res.stdout)
        # print(res.stderr)
        self.assertEqual(res.returncode, 1)
        assert "Benchmark comparison results (Brunner Munzel test, alpha=0.00100)" in res.stdout
        assert "At least one significant difference in main metrics was detected." in res.stdout
        ntests = 0
        for l in res.stdout.splitlines():
            if "(1000 vs 1000)" in l:
                ntests += 1
                if "bm1" in l:
                    assert " < " in l
                elif "bm2" in l:
                    assert " ~ " in l
        assert ntests == 2

    def test_single_source_mode(self):
        args = [
            "path/to/file",
            "--files_parser",
            os.path.join(_this_file_dir, "parser_single.py"),
        ]
        res = run(args)
        #print(res.stdout)
        # print(res.stderr)
        self.assertEqual(res.returncode, 1)
        assert "Benchmark comparison results (Brunner Munzel test, alpha=0.00100)" in res.stdout
        assert "At least one significant difference in main metrics was detected." in res.stdout
        ntests = 0
        for l in res.stdout.splitlines():
            if "(1000 vs 1000)" in l:
                ntests += 1
                if "bm1 | var1 vs var2" in l:
                    assert " < " in l
                elif (
                    "bm2 | opt1 vs opt2" in l or "bm2 | opt1 vs opt3" in l or "bm2 | opt2 vs opt3" in l
                ):
                    assert " ~ " in l
        assert ntests == 4

    def test_load_parser_by_package_name(self):
        args = [
            os.path.join(_this_file_dir, "data/f1.csv"),
            os.path.join(_this_file_dir, "data/f2.csv"),
            "--files_parser",
            "benchstats.parser_SingleColumnCSV",
        ]
        res = run(args)
        #print(res.stdout)
        #print(res.stderr)
        self.assertEqual(res.returncode, 1)
        assert "Benchmark comparison results (Brunner Munzel test, alpha=0.00100)" in res.stdout
        assert "At least one significant difference in main metrics was detected." in res.stdout
        ntests = 0
        for l in res.stdout.splitlines():
            if "(1000 vs 1000)" in l:
                ntests += 1
                if "bm" in l:
                    assert " < " in l
        assert ntests == 1


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))

import sys
import unittest
import numpy as np
import pytest
import time
import traceback

import benchstats.qbench as qb


class TestBench(unittest.TestCase):
    def test_canonical_form(self):
        f1 = lambda: time.sleep(0.001)  # noqa : E731
        f2 = lambda x: time.sleep(x)  # noqa : E731
        r = qb.bench((f1, (f2, lambda: (0.01,))), iters=2, reps=1)
        np.testing.assert_allclose(
            r, np.array([[[0.001, 0.001]], [[0.01, 0.01]]]), rtol=0, atol=0.005
        )

    def test_wait_arg_complete(self):
        def wait_arg_complete(x):
            s = traceback.extract_stack()
            self.assertGreater(len(s), 4)
            # ensuring that wait_arg_complete() isn't called under benchmarks timing section
            # (might happen if genexpr isn't instantiated previously)
            self.assertNotIn("bench_func", s[-3].line)
            return x

        qb.bench(
            ((lambda x: x, lambda: (1,)),),
            iters=1,
            reps=1,
            warmup=0,
            wait_arg_complete=wait_arg_complete,
        )

    def test_BenchmarkDescription(self):
        # test initialization
        BD = qb.BenchmarkDescription

        def bench_func():
            return 42

        def wait_complete():
            return 24

        def _assertRaw(obj):
            self.assertIs(obj.bench_func, bench_func)
            self.assertIsNone(obj.args_func)
            self.assertIsNone(obj.clear_cache_func)
            self.assertIs(obj.wait_arg_complete, wait_complete)
            self.assertIs(obj.wait_func_complete, wait_complete)

        a = BD(bench_func, wait_complete=wait_complete)
        _assertRaw(a)

        # test from_iterable
        def _assertFromIterable(obj, has_completes=True):
            self.assertIs(obj.bench_func, bench_func)
            self.assertIsNotNone(obj.args_func)
            self.assertIsNotNone(obj.clear_cache_func)
            if has_completes:
                self.assertIs(obj.wait_arg_complete, wait_complete)
                self.assertIs(obj.wait_func_complete, wait_complete)
            else:
                self.assertIsNotNone(obj.wait_arg_complete)
                self.assertIsNotNone(obj.wait_func_complete)
                
        b = BD.from_iterable(a)
        _assertFromIterable(b)
        c = BD.from_iterable((bench_func,))
        _assertFromIterable(c, False)
        d = BD.from_iterable(
            (bench_func,),
            def_wait_arg_complete=wait_complete,
            def_wait_func_complete=wait_complete,
        )
        _assertFromIterable(d)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))

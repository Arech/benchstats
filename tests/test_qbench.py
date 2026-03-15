import sys
from typing import Iterable
import unittest
import numpy as np
import pytest
import time
import traceback

import benchstats.qbench as qb


class TestQBench(unittest.TestCase):
    def test_canonical_form(self):
        f1 = lambda: time.sleep(0.001)  # noqa : E731
        f2 = lambda x: time.sleep(x)  # noqa : E731
        r = qb.bench((f1, (f2, lambda: (0.01,))), iters=2, reps=1, show_progress_each=0)
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
            show_progress_each=0,
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

    def test_function_propagation(self):
        bench_cnt, args_cnt, cache_cnt, wait_arg_cnt, wait_func_cnt = 0, 0, 0, 0, 0

        def bench_func():
            nonlocal bench_cnt
            bench_cnt += 1
            return 1

        def bench_func2(x, y):
            self.assertEqual(x, 42)
            self.assertEqual(y, 24)
            nonlocal bench_cnt
            bench_cnt += 1
            return 1

        def args_func():
            nonlocal args_cnt
            args_cnt += 1
            return tuple()

        def args_func2():
            nonlocal args_cnt
            args_cnt += 1
            return 42, 24

        def cache_func(x):
            self.assertEqual(x, tuple())
            nonlocal cache_cnt
            cache_cnt += 1

        def cache_func2(x):
            self.assertEqual(x, (42, 24))
            nonlocal cache_cnt
            cache_cnt += 1

        def wait_arg(x):
            self.assertIn(x, (42, 24))
            nonlocal wait_arg_cnt
            wait_arg_cnt += 1
            return x

        def wait_func(x):
            self.assertEqual(x, 1)
            nonlocal wait_func_cnt
            wait_func_cnt += 1
            return x

        def wait_arg_func(x):
            self.assertIn(x, (42, 24, 1))
            nonlocal wait_func_cnt
            wait_func_cnt += 1
            return x

        def check_reset_counters(exp_bench, exp_args, exp_cache, exp_wait_arg, exp_wait_func):
            nonlocal bench_cnt, args_cnt, cache_cnt, wait_arg_cnt, wait_func_cnt
            self.assertEqual(exp_bench, bench_cnt)
            self.assertEqual(exp_args, args_cnt)
            self.assertEqual(exp_cache, cache_cnt)
            self.assertEqual(exp_wait_arg, wait_arg_cnt)
            self.assertEqual(exp_wait_func, wait_func_cnt)
            bench_cnt, args_cnt, cache_cnt, wait_arg_cnt, wait_func_cnt = 0, 0, 0, 0, 0

        common_opts = {"iters": 3, "reps": 2, "warmup": 1, "show_progress_each": 0}

        r = qb.bench(bench_func, **common_opts)
        np.testing.assert_array_equal(r.shape, (2, 3))
        check_reset_counters(7, 0, 0, 0, 0)

        qb.bench(bench_func, **common_opts, clear_cache=cache_func, wait_complete=wait_func)
        check_reset_counters(7, 0, 7, 0, 7)

        qb.bench(
            bench_func,
            **common_opts,
            clear_cache=cache_func,
            wait_arg_complete=wait_arg,
            wait_func_complete=wait_func,
        )
        check_reset_counters(7, 0, 7, 0, 7)

        qb.bench(
            ((bench_func2, args_func2),),
            **common_opts,
            clear_cache=cache_func2,
            wait_arg_complete=wait_arg,
            wait_func_complete=wait_func,
        )
        check_reset_counters(7, 7, 7, 14, 7)

        qb.bench(
            ((bench_func, args_func, cache_func),),
            **common_opts,
            wait_arg_complete=wait_arg,
            wait_func_complete=wait_func,
        )
        check_reset_counters(7, 7, 7, 0, 7)

        qb.bench(
            ((bench_func2, args_func2, cache_func2),),
            **common_opts,
            wait_arg_complete=wait_arg,
            wait_func_complete=wait_func,
        )
        check_reset_counters(7, 7, 7, 14, 7)

        qb.bench(
            ((bench_func2, args_func2, cache_func2, wait_arg, wait_func),),
            **common_opts,
        )
        check_reset_counters(7, 7, 7, 14, 7)

        qb.bench(
            (
                qb.BenchmarkDescription(
                    bench_func2,
                    args_func2,
                    cache_func2,
                    wait_arg_complete=wait_arg,
                    wait_func_complete=wait_func,
                ),
            ),
            **common_opts,
        )
        check_reset_counters(7, 7, 7, 14, 7)

        qb.bench(
            (
                qb.BenchmarkDescription(
                    bench_func2,
                    args_func2,
                    cache_func2,
                    wait_complete=wait_arg_func,
                ),
            ),
            **common_opts,
        )
        check_reset_counters(7, 7, 7, 0, 7 + 14)

        qb.bench(
            (
                qb.BenchmarkDescription(
                    bench_func2,
                    args_func2,
                    cache_func2,
                    wait_complete=wait_arg_func,
                ),
                (bench_func, args_func, cache_func),
            ),
            **common_opts,
        )
        check_reset_counters(7 * 2, 7 * 2, 7 * 2, 0, 7 + 14)

    def test_showBench_with_delimiter(self):
        def_prms = {"alt_delimiter": "|", "render_report": False}
        nreps = 10
        results = np.stack(
            (
                np.ones((nreps, 1)),
                1.1 * np.ones((nreps, 1)),
                2 * np.ones((nreps, 1)),
                1.9 * np.ones((nreps, 1)),
                3 * np.ones((nreps, 1)),
                3 * np.ones((nreps, 1)),
            ),
            axis=0,
        )
        bm_names = ("alg0|A", "alg0|B", "alg1|A", "alg1|B", "alg2|A", "alg2|B")
        sr = qb.showBench(results, bm_names=bm_names, **def_prms)
        assert "<" == sr.results["alg0 | A vs B"]["mean"].result
        assert ">" == sr.results["alg1 | A vs B"]["mean"].result
        assert "~" == sr.results["alg2 | A vs B"]["mean"].result

    def test_showBench_one_name(self):
        def_prms = {"render_report": False}
        nreps = 10
        results = np.stack(
            (
                np.ones((nreps, 1)),
                1.1 * np.ones((nreps, 1)),
                0.9 * np.ones((nreps, 1)),
            ),
            axis=0,
        )
        sr = qb.showBench(results, bm_names="code", **def_prms)
        assert "<" == sr.results["code | 0 vs 1"]["mean"].result
        assert ">" == sr.results["code | 1 vs 2"]["mean"].result
        assert ">" == sr.results["code | 0 vs 2"]["mean"].result

    def test_showBench_tuple(self):
        def_prms = {"render_report": False}
        nreps = 10
        results = np.stack(
            (
                np.ones((nreps, 1)),
                1.1 * np.ones((nreps, 1)),
                2 * np.ones((nreps, 1)),
                1.9 * np.ones((nreps, 1)),
                3 * np.ones((nreps, 1)),
                3 * np.ones((nreps, 1)),
            ),
            axis=0,
        )
        bm_names = ("alg0", "alg1", "alg2")
        sr = qb.showBench(results, bm_names=bm_names, **def_prms)
        assert "<" == sr.results["alg0 | 0 vs 1"]["mean"].result
        assert ">" == sr.results["alg1 | 0 vs 1"]["mean"].result
        assert "~" == sr.results["alg2 | 0 vs 1"]["mean"].result

    def test_showBench_pvalue_stats_bootstrap(self):
        r = np.array([[[1, 3, 3], [1, 3, 3]], [[1, 2, 2], [1, 2, 2]]])
        cmnargs = dict(
            metrics={"min": np.min},
            pvalue_stats_bootstrap_seed=3,
            alpha=0.499,
            render_report=False,
            console=None,
        )
        # seed is chosen to modify the first bootstrap result and produce the same second result

        def test_pvs(pvs, same_len):
            assert len(pvs) == 3
            assert len(pvs[">"]) == 1
            self.assertAlmostEqual(pvs[">"][0], 0.3986798203636032)
            assert len(pvs["<"]) == 0
            assert len(pvs["~"]) == same_len
            assert all(pvs["~"][i] == 1.0 for i in range(same_len))

        cr = qb.showBench(r, pvalue_stats_bootstrap=1, **cmnargs)
        test_pvs(next(iter(next(iter(cr.pval_stats.values())).values())), 1)

        cr = qb.showBench(r, pvalue_stats_bootstrap=2, **cmnargs)
        test_pvs(next(iter(next(iter(cr.pval_stats.values())).values())), 2)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))

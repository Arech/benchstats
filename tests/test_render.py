import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rich.color import Color, ColorSystem
from rich.text import Text

from benchstats.common import LoggingConsole
from benchstats.compare import compareStats
from benchstats.render import kDefaultStyles, renderComparisonResults


class TestRender(unittest.TestCase):
    def test_metric_colors_distinct_in_4bits(self):
        # can't property test None, so checking these as assumptions
        assert kDefaultStyles["metric_main_~"] is None
        assert kDefaultStyles["metric_scnd_~"] is None
        clr_flds = ["metric_main_<", "metric_main_>", "metric_scnd_<", "metric_scnd_>"]
        clrs = set()
        for fld in clr_flds:
            clr = Color.parse(kDefaultStyles[fld]).downgrade(ColorSystem.STANDARD)
            n = clr.number
            assert n not in clrs
            clrs.add(n)
        assert len(clrs) == 4

    def test_table_has_correct_entries_and_colors(self):
        # we need to output grid of several main and secondary metric results
        n_samples = 10
        small = np.ones(n_samples)
        large = 2 * np.ones(n_samples)

        def _makeData(res) -> tuple:
            if "<" == res:
                return small, large, "1.000s < 2.000s {+100.0%}"
            elif ">" == res:
                return large, small, "2.000s > 1.000s {-50.0%}"
            assert "~" == res
            return small, small, "1.000s ~ 1.000s {+0.0%}"

        def _bmNameColorFieldName(m1, m2, s1, s2) -> str:
            if m1 != "~":
                return f"metric_main_{m1}"
            if m2 != "~":
                return f"metric_main_{m2}"
            if s1 != "~":
                return f"metric_scnd_{s1}"
            if s2 != "~":
                return f"metric_scnd_{s2}"
            return "metric_main_~"

        def _colorForField(fld_name) -> Color | None:
            clr = kDefaultStyles[fld_name]
            if clr is None:
                return None
            return Color.parse(clr).downgrade(ColorSystem.STANDARD)

        s1, s2 = {}, {}
        possible_results = ["<", ">", "~"]
        CellExpectation = tuple[str, Color | None]  # text & color
        expected: list[tuple[CellExpectation, ...]] = []
        for main1 in possible_results:
            m11, m12, m1e = _makeData(main1)
            for main2 in possible_results:
                m21, m22, m2e = _makeData(main2)
                for scnd1 in possible_results:
                    r11, r12, s1e = _makeData(scnd1)
                    for scnd2 in possible_results:
                        name = f"m1_{main1} m2_{main2} s1_{scnd1} s2_{scnd2}"
                        r21, r22, s2e = _makeData(scnd2)
                        s1[name] = {"main1": m11, "scnd1": r11, "main2": m21, "scnd2": r21}
                        s2[name] = {"main1": m12, "scnd1": r12, "main2": m22, "scnd2": r22}

                        expected.append((
                            (
                                name,
                                _colorForField(_bmNameColorFieldName(main1, main2, scnd1, scnd2)),
                            ),
                            (m1e, _colorForField(f"metric_main_{main1}")),
                            (m2e, _colorForField(f"metric_main_{main2}")),
                            (s1e, _colorForField(f"metric_scnd_{scnd1}")),
                            (s2e, _colorForField(f"metric_scnd_{scnd2}")),
                        ))

        console = LoggingConsole(
            force_terminal=True,
            width=150,
            color_system="standard",  # stable 16-color ANSI; avoid truecolor if you only care about names
            legacy_windows=False,
        )

        main_metrics = ["main1", "main2"]
        cr = compareStats(s1, s2, main_metrics=main_metrics, debug_log=console)

        with console.capture() as cap:
            renderComparisonResults(
                cr, main_metrics=main_metrics, drop_pvalues=True, console=console
            )
        output = cap.get()
        # print(output)

        def assert_in_color(
            output: str, string: str, color: Color | None, line_idx, cell_idx
        ) -> None:
            text = Text.from_ansi(output)
            assert text.plain.strip() == string, (
                f"{text.plain!r} doesn't match expected {string!r} (line_idx={line_idx}, cell_idx={cell_idx})"
            )
            i = text.plain.find(string)
            assert i >= 0, f"{string!r} not in {text.plain!r}"
            for offset in range(i, i + len(string)):
                styles = [span.style for span in text.spans if span.start <= offset < span.end]
                assert (not styles and color is None) or any(
                    (s.color is None and color is None)
                    or (
                        s.color is not None
                        and color is not None
                        and s.color.type == color.type
                        and s.color.number == color.number
                    )
                    for s in styles
                ), (
                    f"{string!r} at {offset} (line_idx={line_idx}, cell_idx={cell_idx}) not {color}; styles={styles}"
                )

        horz_occurrences = 0
        l_idx = 0
        for line in output.splitlines():
            if "──────────────────" in line or "━━━━━━━━━━━━━━━━━━" in line:
                horz_occurrences += 1
                continue
            if horz_occurrences != 2:
                continue
            if horz_occurrences > 2:
                break
            # print(line)
            line_exp = expected[l_idx]
            for idx, cell in enumerate(u for s in line.split("│") if (u := s.strip())):
                # print (cell)
                exp = line_exp[idx]
                assert_in_color(cell, exp[0], exp[1], l_idx, idx)

            l_idx += 1
        assert l_idx == len(expected)

    """
    def test_html_font(self):
        console = LoggingConsole(record=True)
        console.info("Test message")
        with TemporaryDirectory() as tmpdir:
            fpath = str(Path(tmpdir) / "myhtml.html")
            console.save_html(fpath)
            with open(fpath, 'r') as file:
                file_content = file.read()
        print("\nHTML:\n")
        print(file_content)
    """


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv + ["-vs"]))

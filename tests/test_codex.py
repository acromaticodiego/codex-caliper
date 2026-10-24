"""Tests del veredicto CODEX STAN 205-1997 y del grado comercial (Task 10).

Fija los dos umbrales del estandar (longitud y calibre) como inclusivos, y el
truncamiento (no redondeo) del grado comercial una vez alcanzado el minimo
Cavendish de 34 mm.
"""

from __future__ import annotations

import pytest

from codex_caliper.codex import evaluate


@pytest.mark.parametrize(
    "length,caliper,expected",
    [
        (140.0, 27.0, True),  # exactamente en los minimos: cumple
        (139.9, 27.0, False),
        (140.0, 26.9, False),
        (190.0, 38.0, True),
    ],
)
def test_codex_thresholds(length, caliper, expected):
    assert evaluate(length, caliper).passes is expected


def test_commercial_grade_is_the_truncated_caliper_in_mm():
    assert evaluate(190.0, 38.7).commercial_grade == 38


def test_no_commercial_grade_below_the_minimum():
    assert evaluate(190.0, 33.2).commercial_grade is None


@pytest.mark.parametrize(
    "caliper,expected_grade",
    [
        (33.9, None),  # justo debajo del minimo comercial: sin grado
        (34.0, 34),    # exactamente en el minimo: inclusivo, SI tiene grado
        (34.1, 34),    # justo encima: grado 34
    ],
)
def test_commercial_grade_minimum_is_inclusive(caliper, expected_grade):
    assert evaluate(190.0, caliper).commercial_grade == expected_grade

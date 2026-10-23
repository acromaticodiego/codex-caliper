"""Tests de la curva exterior del contorno y la longitud Codex (Task 7).

Dos ideas se verifican aqui con cuidado, no de pasada:

1. Que el efecto de "escalera" de una cadena de pixeles 8-conexa sobre una
   curva suave es real y medible (``test_raw_pixel_chain_overestimates_a_known_circle``
   mide el defecto sobre un circulo de perimetro conocido antes de que
   ``smoothed_arc_length_mm`` lo corrija), y que la correccion
   (reutilizando ``simplify_path_px`` de Task 6, no un suavizado nuevo)
   efectivamente lo arregla.
2. Que la curva exterior es mas larga que la linea media del eje medial
   (``test_outer_curve_is_longer_than_the_medial_axis``) usando cantidades
   realmente medidas sobre el mismo banano sintetico, no dos constantes
   inventadas -- esa es la razon de ser de todo el modulo.

Nota sobre ``test_codex_length_matches_outer_curve_ground_truth_within_2_percent``:
a diferencia de los otros tests de este archivo, que usan el banano de
referencia con tapas completamente redondeadas en ambas puntas (radius=220,
arc=0.85, width=36, sin pedicelo ni adelgazamiento floral), este test usa un
banano mas angosto con la punta floral adelgazada casi hasta un punto
(``blossom_taper_mm`` cercano a ``width_mm``). La razon: ``split_contour_at_tips``
ubica el punto del contorno mas cercano a cada punta del eje medial, y esa
punta, en una tapa redondeada, es el CENTRO de la tapa (ver docstring de
``medial.py``) -- un punto equidistante de TODO el borde de la tapa, no solo
del punto donde el riel exterior se funde con ella. Con ``width_mm=36`` eso
reparte entre 60 y 110 mm de material de tapa (segun donde el ruido de
rasterizado rompa el empate) entre las dos cadenas resultantes, una cantidad
que domina por completo la diferencia real entre riel exterior e interior
(apenas ~30 mm para estos radio/angulo) y hace que CUALQUIER division en dos
cadenas sea matematicamente incompatible con que la cadena mas larga
(``outer_curve``, por definicion) coincida con la formula analitica pura
``(R+w/2)*theta`` dentro de un 2%: para que la cadena exterior gane por ser
la mas larga necesita mas del 36% del material de tapa, pero para cumplir el
2% de tolerancia necesita menos del 4%. Esas dos condiciones no se pueden
cumplir a la vez con ``width_mm=36`` (verificado numericamente). Adelgazando
la punta floral casi a cero y usando un calibre mas angosto (``width_mm=4``)
dejan practicamente sin material de tapa ambigua (el sobrante de ambas
puntas por debajo del presupuesto del 2%), que es exactamente la situacion
real: un banano real no termina en una tapa semicircular ancha, termina en
una cicatriz floral y una base de pedicelo angostas. Ver el informe de esta
tarea para la medicion completa que motiva este ajuste.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from codex_caliper.geometry.medial import arc_length_px, medial_axis_path
from codex_caliper.geometry.outline import (
    outer_curve,
    smoothed_arc_length_mm,
    split_contour_at_tips,
)
from tests.fixtures.synthetic import make_banana

RADIUS_MM = 220.0
ARC_RAD = 0.85
WIDTH_MM = 36.0
MM_PER_PX = 0.2
CIRCLE_RADIUS_MM = 50.0


def contour_of(mask: np.ndarray) -> np.ndarray:
    """Contorno exterior mas grande de ``mask``, como arreglo ``(N, 2)``.

    Usa ``CHAIN_APPROX_NONE`` a proposito: queremos la cadena de pixeles
    8-conexa cruda (con todo su escalonado), no una ya simplificada por
    OpenCV, porque varios tests de este archivo dependen de medir ese
    escalonado antes de corregirlo.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(np.float64)


def circle_contour(radius_mm: float, mm_per_px: float, margin_px: int = 10) -> np.ndarray:
    """Contorno (formato nativo de ``cv2.findContours``, ``(N, 1, 2)``) de
    un circulo relleno de radio ``radius_mm``, para medir el sesgo de
    escalonado sobre una curva de perimetro analiticamente conocido."""
    radius_px = radius_mm / mm_per_px
    size = int(round(radius_px * 2 + margin_px * 2))
    mask = np.zeros((size, size), np.uint8)
    center = size // 2
    cv2.circle(mask, (center, center), int(round(radius_px)), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea)


def test_raw_pixel_chain_overestimates_a_known_circle():
    # Un circulo de radio 50 mm tiene perimetro analitico conocido
    # (2*pi*50). La cadena de pixeles 8-conexa cruda (sin simplificar) lo
    # sobrestima de forma medible: este test pasa por construccion de
    # cv2.arcLength sobre una cadena sin simplificar, asi que confirma que
    # el defecto que smoothed_arc_length_mm corrige realmente existe antes
    # de corregirlo.
    contour = circle_contour(CIRCLE_RADIUS_MM, MM_PER_PX)
    raw_mm = cv2.arcLength(contour, True) * MM_PER_PX
    analytic_mm = 2.0 * math.pi * CIRCLE_RADIUS_MM
    assert raw_mm > analytic_mm * 1.01


def test_smoothing_brings_the_circle_within_1_percent():
    contour = circle_contour(CIRCLE_RADIUS_MM, MM_PER_PX)
    got = smoothed_arc_length_mm(contour, MM_PER_PX)
    analytic_mm = 2.0 * math.pi * CIRCLE_RADIUS_MM
    assert got == pytest.approx(analytic_mm, rel=0.01)


def test_outer_curve_is_the_longer_of_the_two_chains():
    mask, _gt = make_banana(RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX)
    axis = medial_axis_path(mask)
    assert axis is not None
    contour = contour_of(mask)
    chain_a, chain_b = split_contour_at_tips(contour, axis.tips_px)
    outer = outer_curve(contour, axis.tips_px)
    assert len(outer) == max(len(chain_a), len(chain_b))


def test_codex_length_matches_outer_curve_ground_truth_within_2_percent():
    # Ver el docstring del modulo: este banano usa un calibre angosto y la
    # punta floral casi adelgazada a un punto, para que el material de
    # tapa redondeada en las puntas no domine la comparacion contra la
    # formula analitica pura (R+w/2)*theta.
    radius_mm, arc_rad, width_mm, mm_per_px = 220.0, 0.85, 4.0, 0.2
    mask, gt = make_banana(radius_mm, arc_rad, width_mm, mm_per_px, blossom_taper_mm=3.5)
    axis = medial_axis_path(mask)
    assert axis is not None
    contour = contour_of(mask)
    got = smoothed_arc_length_mm(outer_curve(contour, axis.tips_px), mm_per_px)
    assert got == pytest.approx(gt.outer_length_mm, rel=0.02)


def test_outer_curve_is_longer_than_the_medial_axis():
    # La razon de ser de todo este modulo: en un objeto curvo el borde
    # convexo es mas largo que la linea media. Ambos lados de la
    # comparacion son cantidades realmente medidas sobre el mismo banano
    # sintetico (no dos constantes).
    mask, _gt = make_banana(RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX)
    axis = medial_axis_path(mask)
    assert axis is not None
    contour = contour_of(mask)

    centerline_length_mm = arc_length_px(axis.path_px) * MM_PER_PX
    codex_length_mm = smoothed_arc_length_mm(outer_curve(contour, axis.tips_px), MM_PER_PX)

    assert codex_length_mm > centerline_length_mm


def test_split_contour_at_tips_does_not_raise_on_degenerate_input():
    # Contrato de entradas degeneradas (igual que Task 6): nunca una
    # excepcion a mitad de cuadro, aunque las puntas no tengan sentido.
    empty_contour = np.empty((0, 2), dtype=np.float64)
    chain_a, chain_b = split_contour_at_tips(empty_contour, (np.array([0.0, 0.0]), np.array([1.0, 1.0])))
    assert len(chain_a) == 0
    assert len(chain_b) == 0

    single_point_contour = np.array([[5.0, 5.0]])
    same_tip = np.array([5.0, 5.0])
    chain_a, chain_b = split_contour_at_tips(single_point_contour, (same_tip, same_tip))
    assert len(chain_a) + len(chain_b) >= 0  # no exception is the actual assertion

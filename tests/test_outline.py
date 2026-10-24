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

Nota sobre ``test_codex_length_matches_outer_curve_ground_truth_within_2_percent``
(corregida en la ronda de revision 1 -- ver el informe de esta tarea para el
historial completo): a diferencia de los otros tests de este archivo, que
usan el banano de referencia con tapas completamente redondeadas en ambas
puntas (radius=220, arc=0.85, width=36, sin pedicelo ni adelgazamiento
floral), este test usa un banano con la punta floral adelgazada casi hasta
un punto (``blossom_taper_mm`` cercano a ``width_mm``) y angulo de arco mas
grande. La razon tiene DOS causas, y no pesan igual:

**Causa primaria, deterministica, no corregible con un mejor algoritmo de
particion**: ``gt.outer_length_mm`` es la formula analitica pura del riel,
``(R+w/2)*theta`` -- por construccion EXCLUYE todo el material de la tapa
redondeada de las puntas. Pero cualquier particion del contorno real en dos
cadenas tiene que asignar ESE material a alguna de las dos, porque esta
fisicamente alli en la mascara rasterizada. Una tapa completa expone un arco
de ``pi*w/2`` (medio circulo de radio ``w/2``): con ``width_mm=36`` eso es
``pi*18 = 56.55 mm`` por punta, un 28% de los 202.3 mm del riel exterior
puro -- y con las DOS puntas redondeadas (el fixture original, sin
adelgazar), la cadena mas larga midio 284.90 mm contra un ground truth de
202.30 mm: **+40.83%**, medido directamente con la implementacion real, no
estimado. Ningun algoritmo de particion en dos cadenas puede hacer que la
cadena exterior quede dentro del 2% de la formula pura mientras dos tapas
anchas sigan expuestas: es un desajuste estructural entre QUE mide la
formula (riel sin tapas) y QUE contiene el contorno (riel CON tapas), no un
defecto de implementacion.

**Causa secundaria, de ambiguedad por empate**: la punta del eje medial en
una tapa redondeada es el CENTRO de esa tapa (ver docstring de
``medial.py``), un punto casi equidistante de TODO el borde expuesto de la
tapa -- medido sobre el fixture original (width=36): la punta esta a 17.2 mm
del contorno (la mitad del calibre, 18 mm, como se espera de una tapa
redondeada) y hay 85 y 97 puntos del contorno (en las dos puntas
respectivamente) a menos de 1 px de esa distancia minima. Que punto exacto
de ese empate gana la busqueda de "mas cercano" depende del ruido de
rasterizado, no de la geometria -- pero esto solo decide COMO se reparte el
material de la tapa entre las dos cadenas, no SI hay material de tapa que
repartir. Es secundario frente a la causa primaria: incluso con una
particion perfectamente deterministica y libre de ruido, el material de la
tapa seguiria estando alli y seguiria rompiendo el 2% a ``width_mm=36``.

Adelgazar la punta floral casi a cero (``blossom_taper_mm`` ~ ``width_mm``)
elimina el material de tapa de ESE extremo (un cono no tiene "tapa": el
radio cae a 0 exactamente en la punta). Queda sin adelgazar el extremo del
pedicelo (sin pedicelo en este fixture, osea tambien una tapa redondeada
completa) -- su contribucion, acotada por ``(w/2)*pi``, es la que el test
verifica explicitamente que no domine, en vez de asumirlo. Ver el informe
de esta tarea para la medicion completa, incluida la prueba de que el test
efectivamente falla si ``outer_curve`` selecciona la cadena equivocada.
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
    # Ver el docstring del modulo: este banano adelgaza la punta floral casi
    # a un punto y usa un angulo de arco grande para que el riel exterior
    # puro (R+w/2)*theta sea mucho mas grande que el material de tapa
    # redondeada que queda sin adelgazar en el extremo del pedicelo.
    radius_mm, arc_rad, width_mm, mm_per_px = 220.0, 4.0, 12.0, 0.2
    blossom_taper_mm = 11.9
    mask, gt = make_banana(radius_mm, arc_rad, width_mm, mm_per_px, blossom_taper_mm=blossom_taper_mm)
    axis = medial_axis_path(mask)
    assert axis is not None
    contour = contour_of(mask)

    chain_a, chain_b = split_contour_at_tips(contour, axis.tips_px)
    length_a = smoothed_arc_length_mm(chain_a, mm_per_px)
    length_b = smoothed_arc_length_mm(chain_b, mm_per_px)
    outer_mm = smoothed_arc_length_mm(outer_curve(contour, axis.tips_px), mm_per_px)
    inner_mm = min(length_a, length_b)

    assert outer_mm == pytest.approx(gt.outer_length_mm, rel=0.02)

    # No alcanza con "outer > inner" por cualquier margen: eso pasaria por
    # casualidad aunque outer_curve devolviera la cadena equivocada, o
    # aunque no se partiera el contorno en absoluto (ver fix round 1, donde
    # el fixture original fallaba exactamente asi). El riel exterior e
    # interior puros difieren, en forma cerrada, en width_mm*arc_rad
    # ((R+w/2)*theta - (R-w/2)*theta = w*theta) -- la unica tapa que queda
    # sin adelgazar (la del pedicelo) puede desplazar el margen MEDIDO hasta
    # (w/2)*pi de distancia de ese valor cerrado (la mitad de esa tapa para
    # cada cadena, en el peor de los casos), pero nunca mas: el margen real
    # tiene que caer dentro de esa cota, no solo ser positivo.
    expected_gap_mm = width_mm * arc_rad
    cap_slack_mm = (width_mm / 2.0) * math.pi
    margin_mm = outer_mm - inner_mm
    assert margin_mm == pytest.approx(expected_gap_mm, abs=cap_slack_mm)


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

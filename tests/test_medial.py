"""Tests del eje medial podado (camino geodesico mas largo) y su perfil de
grosor.

La verdad de referencia para las puntas es siempre la geometria cerrada del
banano sintetico (Task 2), que nunca toca el rasterizado: el eje medial de
una capsula (rectangulo con tapas semicirculares) termina, por construccion,
en el CENTRO de cada tapa -- es decir, en el punto de la curva central con
``phi = +-arc_rad/2``. ``_expected_tips_px`` replica a mano el mapeo mm->px
que usa ``_rasterize`` (sin pedicelo ni adelgazamiento floral, el unico caso
que usa este archivo) para ubicar esos dos puntos en pixeles sin inspeccionar
la mascara.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from skimage.morphology import skeletonize

from codex_caliper.geometry.medial import (
    arc_length_px,
    medial_axis_path,
    width_profile_mm,
)
from tests.fixtures.synthetic import make_banana

RADIUS_MM = 220.0
ARC_RAD = 0.85
WIDTH_MM = 36.0
MM_PER_PX = 0.2
CANVAS_MARGIN_MM = 20.0


def _expected_tips_px(
    radius_mm: float,
    arc_rad: float,
    width_mm: float,
    mm_per_px: float,
    margin_mm: float = CANVAS_MARGIN_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Puntas esperadas del eje medial, en pixeles del lienzo.

    Replica el bounding box que ``_rasterize`` (Task 2) calcula a partir de
    los circulos estampados sobre el arco -- origen en ``(min_x, min_y)``,
    escala ``mm_per_px`` -- para un banano SIN pedicelo ni adelgazamiento
    floral, y ubica alli los dos puntos ``phi = +-arc_rad/2`` de la curva
    central.
    """
    half_w = width_mm / 2.0
    x_extent = radius_mm * math.sin(arc_rad / 2.0)
    y_tip = -radius_mm * math.cos(arc_rad / 2.0)
    min_x = -x_extent - half_w - margin_mm
    min_y = -radius_mm - half_w - margin_mm  # phi=0 es el punto mas extremo en y
    tip_a = np.array([(-x_extent - min_x) / mm_per_px, (y_tip - min_y) / mm_per_px])
    tip_b = np.array([(x_extent - min_x) / mm_per_px, (y_tip - min_y) / mm_per_px])
    return tip_a, tip_b


def _distance_between(tips_px: tuple[np.ndarray, np.ndarray], expected: tuple[np.ndarray, np.ndarray]) -> float:
    """Distancia maxima entre puntas detectadas y esperadas, probando ambos
    emparejamientos (el orden punta-a-punta del eje medial no esta
    garantizado de antemano)."""
    a, b = tips_px
    ea, eb = expected
    direct = max(float(np.linalg.norm(a - ea)), float(np.linalg.norm(b - eb)))
    swapped = max(float(np.linalg.norm(a - eb)), float(np.linalg.norm(b - ea)))
    return min(direct, swapped)


def tiny_blob() -> np.ndarray:
    """Un blob de 2x2 pixeles: menos que unos pocos pixeles de eje medial posible."""
    mask = np.zeros((50, 50), np.uint8)
    mask[24:26, 24:26] = 255
    return mask


def perfect_disc() -> np.ndarray:
    """Un disco perfecto: su eje medial colapsa cerca del centro, sin dos
    puntas significativas (elongacion ~1, ver ``_is_elongated`` en medial.py)."""
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(mask, (100, 100), 60, 255, -1)
    return mask


def _bump_mask(mask: np.ndarray, path_px: np.ndarray, radius_px: np.ndarray) -> np.ndarray:
    """Agrega una protuberancia lateral pequena, adosada al costado del
    banano a mitad de camino (lejos de ambas puntas), pensada para generar
    una rama en el esqueleto sin alargar el camino principal punta a punta.

    Se ubica con la normal local al eje medial (no al azar) para garantizar
    que quede efectivamente afuera del contorno existente del banano, en vez
    de perderse dentro de el.
    """
    i = max(1, min(int(len(path_px) * 0.35), len(path_px) - 2))
    direction = path_px[i + 1] - path_px[i - 1]
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    local_radius = float(radius_px[i])
    bump_radius = 9.0
    center = path_px[i] + normal * (local_radius + 6.0)

    bumped = mask.copy()
    cv2.circle(bumped, (int(round(center[0])), int(round(center[1]))), int(bump_radius), 255, -1)
    return bumped


def test_tips_land_near_the_ends_of_the_synthetic_banana():
    mask, _gt = make_banana(RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX)
    axis = medial_axis_path(mask)
    assert axis is not None
    expected = _expected_tips_px(RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX)
    assert _distance_between(axis.tips_px, expected) < 5.0


def test_centerline_length_matches_ground_truth_within_2_percent():
    mask, gt = make_banana(RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX)
    axis = medial_axis_path(mask)
    assert axis is not None
    length = arc_length_px(axis.path_px) * MM_PER_PX
    assert length == pytest.approx(gt.centerline_length_mm, rel=0.02)


@pytest.mark.parametrize("width_mm", [36.0, 24.0, 12.0])
def test_width_profile_abscissa_matches_the_analytic_centerline(width_mm):
    # Task 8, ronda de revision 4: ``s`` esta en MILIMETROS, asi que tiene
    # que medir milimetros reales. Antes de esta ronda era la longitud
    # acumulada del camino CRUDO del esqueleto, con el mismo sesgo de
    # escalera que ``arc_length_px`` existe para corregir: medido +5.94%
    # (w=36), +5.78% (w=24) y +5.77% (w=12) sobre la longitud central
    # analitica -- sistematico y casi constante entre calibres, es
    # discretizacion, no geometria del banano. Nada en la firma de
    # ``width_profile_mm`` advertia de ese 6%, y Task 8 (area y volumen
    # integrados sobre esta abscisa) mas Tasks 9 y 14 consumen el mismo
    # perfil, asi que la correccion se aplica aqui, una sola vez.
    #
    # Parametrizado sobre tres calibres porque el sesgo era practicamente
    # independiente del calibre: un test de un solo calibre no distinguiria
    # una correccion real de una constante ajustada a ese caso.
    mask, gt = make_banana(RADIUS_MM, ARC_RAD, width_mm, MM_PER_PX)
    axis = medial_axis_path(mask)
    assert axis is not None
    s_mm, w_mm = width_profile_mm(axis, MM_PER_PX)

    assert len(s_mm) == len(w_mm)  # correspondencia indice a indice intacta
    assert s_mm[0] == 0.0
    assert np.all(np.diff(s_mm) >= 0.0)  # monotona: sigue siendo una abscisa

    # el eje medial de una capsula va de CENTRO a CENTRO de tapa, asi que su
    # longitud es la de la curva central analitica (ver docstring del modulo
    # de tests). Medido tras la correccion: +0.42% / +0.33% / +0.16%.
    assert s_mm[-1] == pytest.approx(gt.centerline_length_mm, rel=0.02)

    # y es la misma longitud que mide ``arc_length_px`` sobre el mismo
    # camino: la correccion es exactamente ese reescalado, no un ajuste
    # aparte que pudiera desviarse de la primitiva.
    assert s_mm[-1] == pytest.approx(arc_length_px(axis.path_px) * MM_PER_PX, rel=1e-9)


def test_width_profile_is_flat_for_a_constant_width_capsule():
    mask, _gt = make_banana(RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX)
    axis = medial_axis_path(mask)
    assert axis is not None
    _s, w = width_profile_mm(axis, MM_PER_PX)
    middle = w[len(w) // 4 : 3 * len(w) // 4]
    assert middle.std() < 0.5
    assert middle.mean() == pytest.approx(WIDTH_MM, abs=0.6)


def test_spurs_do_not_lengthen_the_path():
    mask_plain, _gt = make_banana(RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX)
    axis_plain = medial_axis_path(mask_plain)
    assert axis_plain is not None

    mask_bump = _bump_mask(mask_plain, axis_plain.path_px, axis_plain.radius_px)

    # Confirmar, inspeccionando el esqueleto, que la protuberancia realmente
    # agrega una rama (no un engrosamiento que el esqueleto ignora): el
    # esqueleto del banano con bulto debe tener mas pixeles que el del
    # banano simple por un margen que no sea ruido de un pixel suelto.
    skeleton_plain = skeletonize(mask_plain.astype(bool))
    skeleton_bump = skeletonize(mask_bump.astype(bool))
    assert skeleton_bump.sum() > skeleton_plain.sum() + 3

    axis_bump = medial_axis_path(mask_bump)
    assert axis_bump is not None

    length_plain = arc_length_px(axis_plain.path_px) * MM_PER_PX
    length_bump = arc_length_px(axis_bump.path_px) * MM_PER_PX
    assert length_bump == pytest.approx(length_plain, rel=0.02)


@pytest.mark.parametrize(
    "mask",
    [np.zeros((50, 50), np.uint8), tiny_blob(), perfect_disc()],
    ids=["empty", "tiny_blob", "perfect_disc"],
)
def test_returns_none_for_degenerate_masks(mask):
    assert medial_axis_path(mask) is None

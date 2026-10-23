"""Pruebas del generador de bananos sinteticos (fixture de verdad analitica).

Toda la geometria de las tareas 6, 7 y 8 se valida contra estos bananos
sinteticos porque su longitud, calibre, area y volumen se conocen en forma
cerrada (cápsula de ancho constante sobre un arco de circunferencia), sin
depender de ninguna medicion.
"""

import cv2
import numpy as np
import pytest

from tests.fixtures.synthetic import GroundTruth, make_banana, make_banana_scene


def shadow_region_of(scene: np.ndarray) -> np.ndarray:
    """Mascara booleana de los pixeles de sombra en una escena sintetica.

    La sombra es el unico elemento gris neutro (B=G=R) de la escena que no
    es el fondo blanco: el banano es amarillo saturado (canales desiguales)
    y, si hay marcadores, estos son casi negros. Aislar "neutro pero ni
    blanco ni casi negro" aisla exactamente la sombra, sin asumir su
    posicion ni su desplazamiento.
    """
    b = scene[..., 0].astype(np.int16)
    g = scene[..., 1].astype(np.int16)
    r = scene[..., 2].astype(np.int16)
    is_neutral = (np.abs(b - g) <= 2) & (np.abs(g - r) <= 2) & (np.abs(b - r) <= 2)
    is_mid_tone = (r < 250) & (r > 30)
    return is_neutral & is_mid_tone


def test_ground_truth_matches_closed_form():
    _, gt = make_banana(radius_mm=220.0, arc_rad=0.85, width_mm=36.0, mm_per_px=0.2)
    assert gt.centerline_length_mm == pytest.approx(220.0 * 0.85)
    assert gt.outer_length_mm == pytest.approx((220.0 + 18.0) * 0.85)
    assert gt.caliber_mm == 36.0


def test_rendered_area_matches_analytic_area_within_1_percent():
    mask, gt = make_banana(radius_mm=220.0, arc_rad=0.85, width_mm=36.0, mm_per_px=0.2)
    rendered = mask.sum() / 255 * 0.2 ** 2
    assert rendered == pytest.approx(gt.area_mm2, rel=0.01)


@pytest.mark.parametrize(
    "radius_mm, arc_rad, width_mm, mm_per_px",
    [
        (220.0, 0.85, 36.0, 0.2),
        (220.0, 0.85, 36.0, 0.5),
        (220.0, 0.85, 36.0, 1.0),
        (100.0, 1.2, 20.0, 0.2),
        (220.0, 0.85, 10.0, 0.2),
    ],
)
def test_rendered_area_matches_analytic_area_across_parameter_range(radius_mm, arc_rad, width_mm, mm_per_px):
    """El 1% de tolerancia es una propiedad del rasterizador, no de un solo caso.

    Fix round 1 (Task 2): a resolucion nativa, ``cv2.circle`` sobreestima el
    area tanto mas cuanto mas grueso es ``mm_per_px`` frente al calibre del
    banano (hasta 3.36% en estos mismos parametros antes del fix). Estos
    cinco casos son justo los que superaban el 1% declarado; fijarlos aqui
    evita que Tasks 6/7/8 confundan ese sesgo de rasterizado con un bug en
    su propia geometria al elegir sus propias dimensiones de prueba.
    """
    mask, gt = make_banana(radius_mm=radius_mm, arc_rad=arc_rad, width_mm=width_mm, mm_per_px=mm_per_px)
    rendered = mask.sum() / 255 * mm_per_px ** 2
    assert rendered == pytest.approx(gt.area_mm2, rel=0.01)


def test_pedicel_adds_pixels_but_not_ground_truth():
    plain, gt_a = make_banana(220.0, 0.85, 36.0, 0.2)
    stemmed, gt_b = make_banana(220.0, 0.85, 36.0, 0.2, pedicel_mm=25.0, pedicel_width_mm=9.0)
    assert stemmed.sum() > plain.sum()
    assert gt_b.outer_length_mm == gt_a.outer_length_mm


def test_scene_shadow_is_desaturated():
    scene = make_banana_scene([{"radius_mm": 220.0, "arc_rad": 0.85, "width_mm": 36.0}], 0.2)
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    # la sombra debe tener saturacion baja pese a ser oscura en luminancia
    assert hsv[..., 1][shadow_region_of(scene)].max() < 40

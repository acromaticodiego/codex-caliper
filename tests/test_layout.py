"""Tests de la derivacion del layout de marcadores a partir de un frame.

Por que esta derivacion existe
------------------------------
El ``marker_layout`` de la config dice donde esta el centro de cada ArUco en
milimetros, y de el depende TODA la metrica: una homografia ajustada contra
un layout equivocado rectifica a un plano equivocado, sin lanzar nada. Hasta
ahora el unico camino para obtenerlo era medir los cuatro centros con metro.

La tentacion es derivarlo con regla de tres: ``mm_per_px`` a partir del lado
medio del marcador en pixeles, y los centros en milimetros como los centros
en pixeles por esa escala (que es justo lo que calcula
``tools/aruco_probe.py``). Eso IGNORA la perspectiva, y en este rig la
perspectiva no es despreciable: medido sobre un frame real, el lado del
marcador va de 112.9 a 122.2 px segun donde caiga en el cuadro, un 8% de
dispersion. Los tests de aqui estan construidos sobre una camara con
escorzo real por esa razon: una derivacion lineal los falla.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from codex_caliper.aruco_tools import MarkerDetection
from codex_caliper.config import AppConfig
from codex_caliper.layout import derive_marker_layout

#: Layout FISICO de la prueba, deliberadamente distinto del que llevara la
#: config: la derivacion tiene que recuperar ESTE, no el declarado.
TRUE_LAYOUT_MM = {1: (210.0, 140.0), 2: (10.0, 135.0), 3: (5.0, 5.0), 4: (205.0, 0.0)}

#: Como esta pegado cada marcador, en cuartos de vuelta. Son los giros
#: reales del rig: tres a media vuelta y uno a un cuarto.
MOUNTED_TURNS = {1: 2, 2: 2, 3: 2, 4: 1}


def _cfg(**overrides) -> AppConfig:
    base = dict(marker_thickness_mm=2.0, marker_layout=dict(TRUE_LAYOUT_MM))
    base.update(overrides)
    return AppConfig(**base)


def _detections(cfg: AppConfig, layout_mm: dict[int, tuple[float, float]],
                turns: dict[int, int], noise_px: float = 0.0) -> list[MarkerDetection]:
    """Detecciones de una camara con escorzo REAL sobre ``layout_mm``.

    El termino proyectivo no es decorativo: sin el, una derivacion lineal
    (escala unica desde el lado medio del marcador) acertaria y los tests no
    distinguirian una implementacion correcta de una equivocada.

    ``noise_px`` tampoco lo es, y se aprendio rompiendo el codigo a
    proposito: SIN ruido, las 4 esquinas de UN marcador determinan la
    homografia exacta (8 ecuaciones, 8 incognitas, datos perfectos y el
    modelo verdadero es una homografia), asi que una implementacion que se
    quede en un solo marcador de referencia -- la que este modulo documenta
    como mala, con lados de 11 a 22 mm sobre el rig real -- pasa todos los
    tests igual que la buena. Es el patron que este repo ya pago caro. El
    ruido de deteccion es lo que separa las dos: extrapolar desde un
    cuadrado de ~118 px lo amplifica, y ajustar los cuatro a la vez lo
    promedia.
    """
    theta = math.radians(7.0)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    linear = (1.0 / 0.21) * rotation
    translation = np.array([400.0, 250.0])
    perspective = np.array([0.0006, 0.0004])

    def project(point_mm: np.ndarray) -> np.ndarray:
        return (linear @ point_mm + translation) / (1.0 + perspective @ point_mm)

    half = cfg.marker_length_mm / 2.0
    square = [(-half, -half), (half, -half), (half, half), (-half, half)]
    rng = np.random.default_rng(20261103)

    detections: list[MarkerDetection] = []
    for marker_id, centre in layout_mm.items():
        turn = turns[marker_id] % 4
        corners = np.array(
            [project(np.array(centre) + np.array(square[(j + turn) % 4])) for j in range(4)]
        )
        if noise_px:
            corners = corners + rng.normal(0.0, noise_px, corners.shape)
        detections.append(MarkerDetection(id=marker_id, corners=corners))
    return detections


def _pairwise_distances(layout: dict[int, tuple[float, float]]) -> dict[tuple[int, int], float]:
    """Distancias entre centros. Son invariantes a traslacion y rotacion, asi
    que comparan la GEOMETRIA del layout sin fijar un sistema de referencia."""
    return {
        (a, b): float(np.linalg.norm(np.array(layout[a]) - np.array(layout[b])))
        for a, b in itertools.combinations(sorted(layout), 2)
    }


def test_derives_the_true_marker_geometry_under_real_perspective():
    """Las distancias entre centros derivadas deben coincidir con las reales
    bajo una camara con escorzo.

    Rechaza la derivacion LINEAL (``mm_per_px`` del lado medio del marcador,
    centros en pixeles por esa escala): bajo perspectiva, el marcador mas
    lejano sale mas pequeno en pixeles, asi que esa escala unica lo coloca
    mas cerca de lo que esta. Tambien rechaza una derivacion que ajuste la
    homografia contra el layout DECLARADO en vez de contra el tamano
    conocido del marcador -- devolveria el layout declarado, que aqui es el
    verdadero solo por casualidad del fixture, asi que el test lo desvia
    adrede abajo.
    """
    cfg = _cfg(marker_layout={1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.0, 1.0), 4: (1.0, 1.0)})
    detections = _detections(cfg, TRUE_LAYOUT_MM, MOUNTED_TURNS)

    derived = derive_marker_layout(detections, cfg)

    expected = _pairwise_distances(TRUE_LAYOUT_MM)
    got = _pairwise_distances(derived.centres_mm)
    for pair, distance in expected.items():
        assert got[pair] == pytest.approx(distance, abs=0.5), f"par {pair}"


def test_derived_marker_sides_come_back_at_the_known_physical_size():
    """El lado de cada marcador derivado debe volver a ``marker_length_mm``.

    Es la condicion que FIJA la escala metrica: la derivacion no conoce
    ninguna distancia del mundo salvo que cada marcador mide 24 mm de lado.
    Rechaza una implementacion que recupere un layout con la forma correcta
    pero a una escala arbitraria -- que es lo que sale si se ajusta la
    homografia sin imponer el tamano del marcador.
    """
    cfg = _cfg()
    detections = _detections(cfg, TRUE_LAYOUT_MM, MOUNTED_TURNS)

    derived = derive_marker_layout(detections, cfg)

    for marker_id, side in derived.marker_sides_mm.items():
        assert side == pytest.approx(cfg.marker_length_mm, abs=0.2), f"marcador {marker_id}"


def test_how_the_markers_are_glued_does_not_change_the_derived_layout():
    """Girar un marcador sobre su propia chincheta no mueve su centro, asi
    que el layout derivado tiene que ser el mismo con cualquier montaje.

    Rechaza una derivacion que compare las esquinas back-proyectadas contra
    un cuadrado SIEMPRE alineado a los ejes, ignorando como esta pegado cada
    marcador: con un marcador a un cuarto de vuelta esa comparacion fuerza
    un giro global de compromiso y deforma el layout entero.
    """
    cfg = _cfg()
    straight = derive_marker_layout(_detections(cfg, TRUE_LAYOUT_MM, dict.fromkeys(TRUE_LAYOUT_MM, 0)), cfg)
    mounted = derive_marker_layout(_detections(cfg, TRUE_LAYOUT_MM, MOUNTED_TURNS), cfg)

    straight_distances = _pairwise_distances(straight.centres_mm)
    for pair, distance in _pairwise_distances(mounted.centres_mm).items():
        assert distance == pytest.approx(straight_distances[pair], abs=0.2), f"par {pair}"


def test_reports_the_residual_of_its_own_square_condition():
    """La derivacion tiene que publicar lo bien que cuadra, no solo su
    resultado: con datos sinteticos perfectos el RMS es practicamente cero,
    y sobre un frame real sube -- es la unica senal que tiene el operador
    para saber si el layout derivado es de fiar.

    Rechaza una implementacion que devuelva un RMS fijo o no lo calcule.
    """
    cfg = _cfg()
    detections = _detections(cfg, TRUE_LAYOUT_MM, MOUNTED_TURNS)

    derived = derive_marker_layout(detections, cfg)

    assert derived.square_rms_mm < 0.05


def test_refuses_to_derive_a_layout_from_fewer_than_four_markers():
    """Con menos de 4 marcadores no hay con que fijar las 8 incognitas de la
    homografia a partir de cuadrados de 24 mm, y un layout derivado de menos
    seria una invencion presentada como medida.

    Rechaza una implementacion que acepte cualquier cantidad y devuelva un
    layout plausible ajustado sobre datos insuficientes.
    """
    cfg = _cfg()
    detections = _detections(cfg, TRUE_LAYOUT_MM, MOUNTED_TURNS)[:3]

    with pytest.raises(ValueError, match="4"):
        derive_marker_layout(detections, cfg)


def test_fits_all_four_markers_at_once_instead_of_extrapolating_from_one():
    """Con ruido de deteccion realista, el layout sigue saliendo bien.

    Rechaza la implementacion que se queda en la homografia exacta de UN
    marcador de referencia y proyecta los demas por ella. Sin ruido esa
    implementacion es indistinguible de la buena -- 4 esquinas determinan
    una homografia exacta sobre datos perfectos -- asi que es el ruido lo
    que la separa: extrapolar un cuadrado de ~118 px hasta 1000 px de
    distancia amplifica el error de esquina, mientras que ajustar los
    cuatro marcadores a la vez lo promedia.

    Verificado rompiendo el codigo a proposito: devolviendo la homografia
    de arranque en vez de la ajustada, este test falla y el resto de este
    archivo sigue en verde.
    """
    cfg = _cfg()
    detections = _detections(cfg, TRUE_LAYOUT_MM, MOUNTED_TURNS, noise_px=0.5)

    derived = derive_marker_layout(detections, cfg)

    expected = _pairwise_distances(TRUE_LAYOUT_MM)
    got = _pairwise_distances(derived.centres_mm)
    for pair, distance in expected.items():
        assert got[pair] == pytest.approx(distance, abs=2.0), f"par {pair}"

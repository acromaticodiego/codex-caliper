"""Tests de deteccion de marcadores ArUco y del estado de confianza de calibracion.

El hallazgo central (medido sobre el montaje real) es que el defecto de OpenCV
para `perspectiveRemovePixelPerCell` (4 px/celda) deja indecodificable el
marcador id 3 impreso en FDM: una sola linea de capa domina una muestra de
16 px. Subirlo a 8 (64 px/celda) hace que los cuatro IDs decodifiquen en las
cuatro capturas reales del montaje. La alternativa de subir
`errorCorrectionRate` tambien "arregla" el sintoma pero por el motivo
equivocado (acepta mas bits erroneos en vez de muestrear mejor), asi que no
se prueba aqui como solucion valida.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from codex_caliper.aruco_tools import (
    CalibrationState,
    MarkerDetection,
    build_detector_params,
    calibration_state,
    detect_markers,
)
from codex_caliper.config import AppConfig
from tests.fixtures.synthetic import make_banana_scene

CAPTURAS_DIR = Path(__file__).resolve().parent.parent / "capturas"
REAL_CAPTURE_NAMES = [
    "probe_1792776518.png",
    "probe_1792776524.png",
    "probe_1792776526.png",
    "probe_1792776528.png",
]


def _cfg(marker_ids: list[int] | None = None) -> AppConfig:
    return AppConfig(marker_thickness_mm=2.0, marker_ids=marker_ids or [1, 2, 3, 4])


def _four_marker_scene(mm_per_px: float = 0.3) -> np.ndarray:
    """Escena sintetica con los 4 marcadores reales del rig (ids 1-4)."""
    markers = {
        1: (0.0, 0.0),
        2: (300.0, 0.0),
        3: (0.0, 200.0),
        4: (300.0, 200.0),
    }
    return make_banana_scene([], mm_per_px, markers=markers, marker_dict="DICT_6X6_50")


def fake_detection(marker_id: int) -> MarkerDetection:
    """Una ``MarkerDetection`` minima, con esquinas arbitrarias, solo para
    ejercitar ``calibration_state`` (que no mira las esquinas en absoluto)."""
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    return MarkerDetection(id=marker_id, corners=corners)


def real_captures() -> list[np.ndarray]:
    """Carga las cuatro capturas reales del montaje usadas como regresion."""
    frames = []
    for name in REAL_CAPTURE_NAMES:
        path = CAPTURAS_DIR / name
        frame = cv2.imread(str(path))
        assert frame is not None, f"could not read real capture {path}"
        frames.append(frame)
    return frames


def test_detects_four_markers_in_a_rendered_board():
    cfg = _cfg()
    scene = _four_marker_scene()
    assert len(detect_markers(scene, cfg)) == 4


@pytest.mark.parametrize(
    "n,expected",
    [
        (4, CalibrationState.OK),
        (3, CalibrationState.DEGRADED),
        (2, CalibrationState.BLOCKED),
        (0, CalibrationState.BLOCKED),
    ],
)
def test_calibration_state_degrades_then_blocks(n, expected):
    detections = [fake_detection(i) for i in range(n)]
    assert calibration_state(detections) == expected


def test_ignores_markers_whose_id_is_not_in_the_configured_set():
    mm_per_px = 0.3
    markers = {
        0: (0.0, 0.0),
        1: (300.0, 0.0),
        2: (0.0, 200.0),
        3: (300.0, 200.0),
        9: (600.0, 100.0),
    }
    scene = make_banana_scene([], mm_per_px, markers=markers, marker_dict="DICT_6X6_50")
    cfg = _cfg(marker_ids=[0, 1, 2, 3])
    detections = detect_markers(scene, cfg)
    assert {d.id for d in detections} == {0, 1, 2, 3}


def test_cell_sampling_is_raised_above_the_opencv_default():
    params = build_detector_params()
    assert params.perspectiveRemovePixelPerCell >= 8


def test_all_four_real_markers_decode_in_every_capture():
    cfg = _cfg()
    for frame in real_captures():
        detections = detect_markers(frame, cfg)
        assert {d.id for d in detections} == {1, 2, 3, 4}


def test_detect_markers_collapses_a_repeated_id_to_one_detection():
    """Dos marcadores fisicos distintos decodificados al mismo id (o, en el
    montaje real, un candidato espurio de la veta de la impresion que
    decodifica por casualidad al mismo id que un marcador genuino) deben
    colapsar a una sola deteccion: el id nunca debe aparecer dos veces en la
    lista que ve el resto del pipeline (Task 5 usa 4 esquinas por id como
    correspondencias de una homografia; un id duplicado la corrompe
    silenciosamente).
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    marker_img = cv2.aruco.generateImageMarker(dictionary, 1, 120)
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    canvas = np.full((400, 400, 3), 255, np.uint8)
    canvas[20:140, 20:140] = marker_bgr
    canvas[220:340, 220:340] = marker_bgr

    cfg = _cfg(marker_ids=[1, 2, 3, 4])
    detections = detect_markers(canvas, cfg)

    assert [d.id for d in detections] == [1]


def test_calibration_state_counts_distinct_ids_not_list_length():
    """3 marcadores distintos mas un duplicado de uno de ellos son 4 entradas
    en la lista pero solo 3 ids distintos: debe clasificar DEGRADED, no OK.
    """
    detections = [fake_detection(1), fake_detection(2), fake_detection(3), fake_detection(1)]
    assert calibration_state(detections) == CalibrationState.DEGRADED

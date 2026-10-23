"""Tests de la homografia del plano metrico: ajuste, residual de confianza,
degradacion elegante y rectificacion a lienzo cenital.

El caso central (``test_measures_the_same_length_at_canvas_centre_and_corner``)
es la razon de ser de toda la tarea: sin homografia, un objeto mide distinto
segun donde caiga en el cuadro, porque el plano de camara nunca es
perfectamente paralelo al plano de trabajo. La escena sintetica de ese test
NO es ortografica -- se le aplica una transformacion proyectiva deliberada
(una "camara" con escorzo real) antes de calibrar, y el propio test verifica
que, sin corregir, el mismo segmento de 100 mm mide visiblemente distinto en
el centro que en la esquina; solo despues de pasar por
``compute_plane_calibration`` + ``rectify`` las dos medidas coinciden.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np
import pytest

from codex_caliper.aruco_tools import CalibrationState, MarkerDetection, detect_markers
from codex_caliper.config import AppConfig
from codex_caliper.homography import (
    canvas_size_for,
    compute_plane_calibration,
    rectify,
    world_points_for,
)
from tests.fixtures.synthetic import make_banana_scene

MARKERS_MM = {1: (0.0, 0.0), 2: (300.0, 0.0), 3: (0.0, 200.0), 4: (300.0, 200.0)}


def _cfg(**overrides) -> AppConfig:
    base = dict(marker_thickness_mm=2.0, marker_layout=dict(MARKERS_MM))
    base.update(overrides)
    return AppConfig(**base)


def _fake_camera_detections(cfg: AppConfig) -> list[MarkerDetection]:
    """Fabrica detecciones "perfectas": las esquinas de cada marcador del
    layout declarado, proyectadas con una homografia de camara conocida
    (rotacion + escala + un termino proyectivo suave), sin ruido de deteccion.

    Aisla el ajuste de homografia del resto del pipeline (deteccion real de
    ArUco, que ya tiene su propia bateria de tests en Task 4).
    """
    theta = math.radians(7.0)
    image_scale = 1.0 / 0.21  # px de imagen por mm -- distinto del lienzo (0.2)
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    linear = image_scale * rotation
    translation = np.array([400.0, 250.0])
    perspective = np.array([0.0003, 0.0002])

    def project(point_mm: np.ndarray) -> np.ndarray:
        xy = linear @ point_mm + translation
        denom = 1.0 + perspective @ point_mm
        return xy / denom

    placeholder = [MarkerDetection(id=marker_id, corners=np.zeros((4, 2))) for marker_id in cfg.marker_ids]
    world_mm = world_points_for(placeholder, cfg)
    image_px = np.array([project(point) for point in world_mm])
    return [
        MarkerDetection(id=marker_id, corners=image_px[i * 4 : (i + 1) * 4])
        for i, marker_id in enumerate(cfg.marker_ids)
    ]


def test_recovers_scale_from_a_known_synthetic_homography():
    """Con las 16 esquinas proyectadas por una homografia de camara conocida
    y sin ruido, el ajuste debe recuperar la escala del lienzo casi exacta y
    un residual de reproyeccion practicamente nulo."""
    cfg = _cfg()
    detections = _fake_camera_detections(cfg)

    plane = compute_plane_calibration(detections, cfg)

    assert plane is not None
    assert plane.state == CalibrationState.OK
    assert plane.mm_per_px == pytest.approx(0.2, rel=1e-6)
    assert plane.residual_mm < 0.05


def test_measures_the_same_length_at_canvas_centre_and_corner():
    """Un segmento de 100 mm colocado en el centro y en la esquina del plano
    debe medir lo mismo tras rectificar -- esta es la razon de ser de la
    homografia.

    La escena de partida NO es ortografica: se renderiza el plano en mm con
    ``make_banana_scene`` (que si es ortografico) y luego se la somete a una
    transformacion proyectiva (``H_cam``) que comprime el borde superior de
    la imagen respecto al inferior, simulando una camara con escorzo real.
    El test primero verifica, sobre la imagen SIN corregir, que el mismo
    segmento de 100 mm produce una longitud en pixeles claramente distinta
    segun donde caiga (de lo contrario la escena no estaria probando nada:
    seria una distorsion trivial que una sola escala ya mediria bien). Solo
    despues verifica que, tras calibrar y rectificar, ambas longitudes
    coinciden.
    """
    mm_per_px_world = 0.15
    cfg = _cfg()

    world_scene = make_banana_scene([], mm_per_px_world, markers=MARKERS_MM, marker_dict="DICT_6X6_50")

    # Mapeo mm->px de la escena ortografica, calibrado empiricamente contra
    # los propios marcadores (no se asume el origen interno del fixture).
    ortho_detections = detect_markers(world_scene, cfg)
    assert {d.id for d in ortho_detections} == set(cfg.marker_ids)
    origin_estimates = [
        np.array(MARKERS_MM[d.id]) - d.corners.mean(axis=0) * mm_per_px_world for d in ortho_detections
    ]
    origin_mm = np.mean(origin_estimates, axis=0)

    def world_px_for(point_mm: tuple[float, float]) -> np.ndarray:
        return (np.array(point_mm, dtype=np.float64) - origin_mm) / mm_per_px_world

    centre_a, centre_b = (100.0, 100.0), (200.0, 100.0)  # 100 mm, cerca del centro del cuadrilatero
    corner_a, corner_b = (20.0, 20.0), (120.0, 20.0)  # 100 mm, cerca de la esquina del marcador 1
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)

    world_scene_with_dots = world_scene.copy()
    for point_mm, color in ((centre_a, RED), (centre_b, RED), (corner_a, BLUE), (corner_b, BLUE)):
        px = world_px_for(point_mm)
        cv2.circle(world_scene_with_dots, (int(round(px[0])), int(round(px[1]))), 8, color, -1, cv2.LINE_8)

    height, width = world_scene_with_dots.shape[:2]
    src_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    top_margin_px = 0.15 * width
    dst_corners = np.array(
        [[top_margin_px, 0], [width - top_margin_px, 0], [width, height], [0, height]], dtype=np.float32
    )
    camera_homography = cv2.getPerspectiveTransform(src_corners, dst_corners)
    frame = cv2.warpPerspective(
        world_scene_with_dots, camera_homography, (width, height), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )

    def pair_distance_px(image: np.ndarray, color: tuple[int, int, int]) -> float:
        diff = np.abs(image.astype(np.int16) - np.array(color, dtype=np.int16))
        mask = np.all(diff <= 40, axis=-1).astype(np.uint8)
        n_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        centroids = []
        for label in range(1, n_labels):
            ys, xs = np.nonzero(labels == label)
            if len(xs) < 4:
                continue
            centroids.append(np.array([xs.mean(), ys.mean()]))
        assert len(centroids) == 2, f"expected 2 dots of this colour, found {len(centroids)}"
        return float(np.linalg.norm(centroids[0] - centroids[1]))

    # Sin corregir: el mismo segmento de 100 mm produce longitudes en pixeles
    # claramente distintas segun donde caiga -- la distorsion es real.
    raw_centre_mm = pair_distance_px(frame, RED) * mm_per_px_world
    raw_corner_mm = pair_distance_px(frame, BLUE) * mm_per_px_world
    assert abs(raw_centre_mm - raw_corner_mm) > 2.0

    detections = detect_markers(frame, cfg)
    plane = compute_plane_calibration(detections, cfg)
    assert plane is not None

    canvas = rectify(frame, plane, canvas_size_for(cfg))

    length_at_centre = pair_distance_px(canvas, RED) * plane.mm_per_px
    length_at_corner = pair_distance_px(canvas, BLUE) * plane.mm_per_px

    assert length_at_centre == pytest.approx(length_at_corner, abs=0.3)


def test_residual_grows_when_a_marker_is_misplaced():
    """Si un marcador esta fisicamente desplazado respecto al layout
    declarado en la config, el residual de reproyeccion debe crecer y
    delatarlo -- es la metrica de confianza que ve el operador."""
    cfg = _cfg()
    detections = _fake_camera_detections(cfg)

    shifted_layout = dict(cfg.marker_layout)
    shifted_id = cfg.marker_ids[0]
    cx, cy = shifted_layout[shifted_id]
    shifted_layout[shifted_id] = (cx + 15.0, cy + 15.0)
    shifted_cfg = dataclasses.replace(cfg, marker_layout=shifted_layout)

    plane = compute_plane_calibration(detections, shifted_cfg)

    assert plane is not None
    assert plane.residual_mm > 1.0


def test_returns_none_when_fewer_than_three_markers():
    """Con menos de 3 marcadores distintos no hay suficiente geometria: el
    caller nunca debe recibir una PlaneCalibration que no pueda sostener una
    medicion."""
    cfg = _cfg()
    two_detections = _fake_camera_detections(cfg)[:2]

    assert compute_plane_calibration(two_detections, cfg) is None


def test_degrades_to_twelve_points_with_three_markers():
    """Con 3 marcadores visibles el sistema sigue calibrando, con 12 puntos
    en vez de 16, y lo declara DEGRADED en vez de fallar."""
    cfg = _cfg()
    three_detections = _fake_camera_detections(cfg)[:3]

    plane = compute_plane_calibration(three_detections, cfg)

    assert plane is not None
    assert plane.state == CalibrationState.DEGRADED


def test_canvas_size_for_covers_the_marker_rectangle_plus_half_marker_margin():
    """El lienzo debe cubrir el rectangulo que abren los CENTROS de los
    marcadores declarados, mas medio lado de marcador por borde (para que el
    cuerpo entero de cada marcador quede dentro), a la escala del lienzo."""
    cfg = _cfg()  # MARKERS_MM: x en [0, 300], y en [0, 200]; marker_length_mm=24, canvas_mm_per_px=0.2

    width_px, height_px = canvas_size_for(cfg)

    margin_mm = cfg.marker_length_mm / 2.0
    expected_width_px = math.ceil((300.0 + margin_mm - (0.0 - margin_mm)) / cfg.canvas_mm_per_px)
    expected_height_px = math.ceil((200.0 + margin_mm - (0.0 - margin_mm)) / cfg.canvas_mm_per_px)
    assert (width_px, height_px) == (expected_width_px, expected_height_px)


def test_world_points_for_corner_order_matches_opencv_detection():
    """La convencion de orden de esquinas de OpenCV (horario desde la
    superior izquierda del propio marcador) no se asume: se verifica contra
    una deteccion real. Si ``world_points_for`` no respetara ese mismo orden,
    la homografia se ajustaria con esquinas de puntos distintos emparejadas
    entre si -- un error que produce una transformacion de apariencia
    plausible pero equivocada, sin que ningun test de "forma" lo note.
    """
    marker_centre_mm = (100.0, 60.0)
    cfg = _cfg(marker_layout={1: marker_centre_mm})
    scene = make_banana_scene([], 0.3, markers={1: marker_centre_mm}, marker_dict="DICT_6X6_50")

    detections = detect_markers(scene, cfg)
    assert len(detections) == 1
    corners_px = detections[0].corners

    # Esquina superior izquierda real en la imagen: entre las dos de menor y
    # (el "arriba" de la imagen), la de menor x.
    top_two_idx = np.argsort(corners_px[:, 1])[:2]
    top_left_idx = top_two_idx[np.argmin(corners_px[top_two_idx, 0])]
    assert top_left_idx == 0, "OpenCV no devuelve la esquina superior izquierda primero: revisar la suposicion"

    world = world_points_for(detections, cfg)
    cx, cy = marker_centre_mm
    half = cfg.marker_length_mm / 2.0
    assert world[0] == pytest.approx([cx - half, cy - half])

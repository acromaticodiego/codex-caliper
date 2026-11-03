"""Tests de calibracion intrinseca: recuperacion de K/distorsion, umbral de
aceptacion, persistencia y la guardia de resolucion contra zoom movido."""

import json

import numpy as np
import cv2
import pytest

from codex_caliper.intrinsics import (
    Intrinsics,
    IntrinsicsMismatch,
    calibrate_from_corners,
    find_chessboard,
    undistort,
)

PATTERN_SIZE = (9, 6)  # esquinas internas (columnas, filas)
SQUARE_MM = 25.0
IMAGE_SIZE = (1920, 1080)

KNOWN_FX = 1400.0
KNOWN_FY = 1400.0
KNOWN_CX, KNOWN_CY = IMAGE_SIZE[0] / 2.0, IMAGE_SIZE[1] / 2.0
KNOWN_K = np.array(
    [[KNOWN_FX, 0, KNOWN_CX], [0, KNOWN_FY, KNOWN_CY], [0, 0, 1]], dtype=np.float64
)
KNOWN_DIST = np.array([-0.22, 0.09, 0.0006, -0.0004, 0.015], dtype=np.float64)

# 15 poses (rx, ry, rz en grados; tx, ty, tz en mm) deliberadamente variadas en
# angulo y profundidad -- un set casi frontal y casi identico deja el sistema
# mal condicionado y la calibracion no converge de forma fiable.
SYNTHETIC_POSES = [
    (0, 0, 0, 0, 0, 500),
    (25, 0, 0, 0, 0, 450),
    (-25, 0, 0, 0, 0, 450),
    (0, 25, 0, 0, 0, 450),
    (0, -25, 0, 0, 0, 450),
    (20, 20, 10, 80, 40, 500),
    (-20, -20, -10, -80, -40, 500),
    (30, -15, 5, -100, 60, 550),
    (-30, 15, -5, 100, -60, 550),
    (15, 0, 20, 0, 0, 400),
    (0, 15, -20, 0, 0, 600),
    (35, 25, 0, 60, -60, 650),
    (-35, -25, 0, -60, 60, 650),
    (10, -30, 15, -40, 30, 480),
    (-10, 30, -15, 40, -30, 480),
]


def _board_object_points() -> np.ndarray:
    """Puntos 3D (Z=0) de un tablero de 9x6 esquinas, centrado en su propio origen."""
    cols, rows = PATTERN_SIZE
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_MM
    objp[:, 0] -= (cols - 1) * SQUARE_MM / 2.0
    objp[:, 1] -= (rows - 1) * SQUARE_MM / 2.0
    return objp


def _synthetic_views():
    """Proyecta el tablero conocido desde las 15 poses con K/dist conocidos."""
    objp = _board_object_points()
    obj_pts, img_pts = [], []
    for rx, ry, rz, tx, ty, tz in SYNTHETIC_POSES:
        rvec = np.deg2rad([rx, ry, rz]).reshape(3, 1)
        tvec = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1)
        projected, _ = cv2.projectPoints(objp, rvec, tvec, KNOWN_K, KNOWN_DIST)
        obj_pts.append(objp)
        img_pts.append(projected.reshape(-1, 2).astype(np.float32))
    return obj_pts, img_pts


def _dummy_intrinsics(**overrides) -> Intrinsics:
    defaults = dict(
        camera_matrix=KNOWN_K.copy(),
        dist_coeffs=KNOWN_DIST.copy(),
        rms_px=0.3,
        image_size=IMAGE_SIZE,
        zoom_note="zoom ~4mm, foco nitido en plano de trabajo",
    )
    defaults.update(overrides)
    return Intrinsics(**defaults)


def test_recovers_known_intrinsics_from_synthetic_views():
    obj_pts, img_pts = _synthetic_views()
    intr = calibrate_from_corners(obj_pts, img_pts, IMAGE_SIZE)
    assert intr.camera_matrix[0, 0] == pytest.approx(KNOWN_FX, rel=0.02)
    assert intr.dist_coeffs[0] == pytest.approx(KNOWN_DIST[0], abs=0.02)
    assert intr.is_acceptable


def test_is_acceptable_follows_the_half_pixel_threshold():
    assert _dummy_intrinsics(rms_px=0.21).is_acceptable is True
    assert _dummy_intrinsics(rms_px=0.80).is_acceptable is False


def test_is_placeholder_fires_only_on_an_exact_zero_rms_not_on_any_good_calibration():
    """Fix de la demo de LinkedIn. La implementacion equivocada que esta
    prueba RECHAZA: reusar ``is_acceptable`` (``rms_px < INTRINSICS_MAX_RMS_PX``)
    como señal de "placeholder" tambien -- esa condicion es VERDADERA para
    0.0 (el placeholder real) pero TAMBIEN para 0.21 (una calibracion
    genuinamente buena), asi que marcaria como placeholder una calibracion
    real que no lo es. ``is_placeholder`` tiene que distinguir ambos casos:
    dispara con 0.0 exacto y NO dispara con 0.21, aunque los dos sean
    ``is_acceptable``.
    """
    assert _dummy_intrinsics(rms_px=0.0).is_placeholder is True
    assert _dummy_intrinsics(rms_px=0.21).is_placeholder is False
    assert _dummy_intrinsics(rms_px=0.21).is_acceptable is True, (
        "0.21 tiene que seguir siendo is_acceptable -- la prueba de arriba solo "
        "vale si ambos casos comparten ese True y is_placeholder los distingue igual"
    )


def test_roundtrip_preserves_matrix_and_zoom_note(tmp_path):
    intr = _dummy_intrinsics(zoom_note="zoom marcado con pintura en 4mm")
    intr.save(tmp_path / "intrinsics.json")
    loaded = Intrinsics.load(tmp_path / "intrinsics.json")

    np.testing.assert_allclose(loaded.camera_matrix, intr.camera_matrix)
    np.testing.assert_allclose(loaded.dist_coeffs, intr.dist_coeffs)
    assert loaded.rms_px == intr.rms_px
    assert loaded.image_size == intr.image_size
    assert loaded.zoom_note == intr.zoom_note


def test_load_raises_when_image_size_differs_from_current_capture():
    intr = _dummy_intrinsics(image_size=(1280, 720))
    with pytest.raises(IntrinsicsMismatch):
        intr.require_compatible((1920, 1080))


def test_require_compatible_accepts_matching_resolution():
    intr = _dummy_intrinsics(image_size=(1920, 1080))
    intr.require_compatible((1920, 1080))  # no debe lanzar


def _render_chessboard(pattern_size=PATTERN_SIZE, square_px=60, margin_px=60):
    """Dibuja un tablero binario nitido con `pattern_size` esquinas internas."""
    cols, rows = pattern_size
    squares_x, squares_y = cols + 1, rows + 1
    w = squares_x * square_px + 2 * margin_px
    h = squares_y * square_px + 2 * margin_px
    img = np.full((h, w), 255, dtype=np.uint8)
    for r in range(squares_y):
        for c in range(squares_x):
            if (r + c) % 2 == 0:
                y0 = margin_px + r * square_px
                x0 = margin_px + c * square_px
                img[y0:y0 + square_px, x0:x0 + square_px] = 0
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), square_px, margin_px


def test_find_chessboard_detects_and_refines_corners():
    frame, square_px, margin_px = _render_chessboard()
    corners = find_chessboard(frame, PATTERN_SIZE)

    cols, rows = PATTERN_SIZE
    assert corners is not None
    assert corners.reshape(-1, 2).shape == (cols * rows, 2)
    # La esquina interna superior-izquierda cae en (margin+square, margin+square).
    first = corners.reshape(-1, 2)[0]
    expected = (margin_px + square_px, margin_px + square_px)
    assert first[0] == pytest.approx(expected[0], abs=0.5)
    assert first[1] == pytest.approx(expected[1], abs=0.5)


def test_find_chessboard_returns_none_when_pattern_absent():
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    assert find_chessboard(blank, PATTERN_SIZE) is None


def test_undistort_is_identity_for_zero_distortion():
    intr = Intrinsics(
        camera_matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]]),
        dist_coeffs=np.zeros(5),
        rms_px=0.1,
        image_size=(640, 480),
        zoom_note="",
    )
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    result = undistort(frame, intr)
    np.testing.assert_array_equal(result, frame)


def test_undistort_raises_when_frame_resolution_does_not_match_calibration():
    # Intrinsecos calibrados a 640x480; el frame entregado es 1920x1080 --
    # exactamente el caso de un anillo de zoom movido entre calibrar y medir.
    intr = _dummy_intrinsics(image_size=(640, 480))
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    with pytest.raises(IntrinsicsMismatch):
        undistort(frame, intr)


def test_undistort_does_not_raise_when_frame_resolution_matches_calibration():
    intr = _dummy_intrinsics(image_size=(1920, 1080))
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    undistort(frame, intr)  # no debe lanzar


def test_dist_coeffs_is_flattened_to_1d_on_direct_construction():
    # Esta es exactamente la forma (5,1) que devuelve cv2.calibrateCamera sin
    # ravelar; una tarea posterior podria construir Intrinsics asi a mano.
    column_dist = np.array([[-0.22], [0.09], [0.0006], [-0.0004], [0.015]])
    intr = _dummy_intrinsics(dist_coeffs=column_dist)
    assert intr.dist_coeffs.shape == (5,)
    assert intr.dist_coeffs[0] == pytest.approx(-0.22)


def test_dist_coeffs_stays_1d_after_save_and_load_roundtrip(tmp_path):
    column_dist = np.array([[-0.22], [0.09], [0.0006], [-0.0004], [0.015]])
    intr = _dummy_intrinsics(dist_coeffs=column_dist)
    intr.save(tmp_path / "intrinsics.json")
    loaded = Intrinsics.load(tmp_path / "intrinsics.json")

    assert loaded.dist_coeffs.shape == (5,)
    assert loaded.dist_coeffs[0] == pytest.approx(-0.22)


# ---------------------------------------------------------------------------
# Forma de ``image_size`` (ronda de correcciones finales, minor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param({"w": 1920, "h": 1080}, id="object-instead-of-pair"),
        pytest.param("19", id="two-character-string"),
        pytest.param([1920, 1080, 3], id="three-components"),
        pytest.param([1920], id="one-component"),
        pytest.param(1920, id="bare-number"),
        pytest.param([1920, None], id="null-component"),
        pytest.param([True, False], id="booleans"),
    ],
)
def test_an_image_size_of_the_wrong_shape_raises_value_error(tmp_path, bad_value):
    """``tuple()`` sobre un OBJETO de JSON devuelve sus CLAVES.

    La implementacion equivocada que esta prueba RECHAZA es
    ``image_size=tuple(data["image_size"])``: con ella,
    ``{"w": 1920, "h": 1080}`` -- una forma que alguien escribe a mano con
    toda naturalidad -- se carga como ``('w', 'h')`` sin ningun error.

    No es corrupcion silenciosa (``require_compatible`` bloquea la medicion
    igual), pero culpa a la cosa equivocada: el operador lee "Recalibrate --
    a zoom or focus ring may have moved" y repite el paso MAS CARO de todo
    el procedimiento por un campo mal escrito en un JSON. ``"19"`` y los
    booleanos estan aqui por lo mismo que en ``test_config.py``: longitud 2
    y ``isinstance(True, int)``.
    """
    path = tmp_path / "intrinsics.json"
    path.write_text(
        json.dumps(
            {
                "camera_matrix": [[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]],
                "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                "rms_px": 0.21,
                "image_size": bad_value,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        Intrinsics.load(path)
    assert "image_size" in str(excinfo.value)


def test_a_well_shaped_image_size_still_loads(tmp_path):
    """El otro lado del mecanismo: lo valido sigue cargando, y sigue siendo
    una pareja de ENTEROS (``require_compatible`` compara contra la tupla de
    enteros que entrega ``CameraThread.actual_size``, asi que un 1920.0
    flotante tendria que fallar esa comparacion)."""
    path = tmp_path / "intrinsics.json"
    Intrinsics(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros(5),
        rms_px=0.21,
        image_size=(1920, 1080),
    ).save(path)

    loaded = Intrinsics.load(path)
    assert loaded.image_size == (1920, 1080)
    loaded.require_compatible((1920, 1080))

"""Tests de las vistas de calibracion intrinseca y de plano (Task 19).

El brief de esta tarea traia seis pruebas de las que CUATRO no discriminaban
una implementacion correcta de una incorrecta (rulings del encargo); se
listan aqui, no solo se mencionan:

1. ``test_undistortion_toggle_switches_the_preview`` original hacia
   ``view.undistort_preview = True; assert view.displayed_frame_is_undistorted()
   is True`` -- si ``displayed_frame_is_undistorted`` simplemente devuelve la
   bandera que ``undistort_preview`` acaba de fijar, la prueba se lee a si
   misma y pasa con CUALQUIER implementacion, incluida una que no corrija
   nada. Aqui se compara el cuadro MOSTRADO antes y despues del toggle, con
   un cuadro con textura real (un degradado) donde la correccion de
   distorsion produce una diferencia de pixeles medible -- y se prueba el
   caso ``False`` tambien.
2. ``test_calibration_is_rejected_above_the_rms_threshold`` original solo
   probaba el rechazo, lo que no distingue "rechaza cuando corresponde" de
   "rechaza siempre"; y ``"0.5" in rejection_message()`` lo satisface
   cualquier mensaje con un 0.5 en cualquier parte. Aqui se cubre el caso
   ACEPTADO tambien, y el mensaje se verifica contra el RMS real obtenido y
   la constante real del umbral (``INTRINSICS_MAX_RMS_PX``), no una
   subcadena.
3. ``test_capture_counter_increases_with_each_accepted_pose`` original solo
   contaba exitos. Se agrega el caso en que ``find_chessboard`` no encuentra
   tablero: una pose sin tablero no es una pose "accepted" y no debe
   incrementar el contador.
4. ``test_plane_view_reports_residual_in_millimetres`` original solo pedia
   ``"mm" in view.residual_text()``, que satisface cualquier cadena con esas
   letras. Aqui se fija el VALOR real contra el ``PlaneCalibration``
   calculado de verdad por ``compute_plane_calibration``.

Ver el informe de la tarea para la evidencia de ruptura deliberada de cada
una (incluida la trampa especifica de la prueba 1, mostrada explicitamente).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from codex_caliper.aruco_tools import CalibrationState
from codex_caliper.config import AppConfig
from codex_caliper.homography import compute_plane_calibration
from codex_caliper.intrinsics import INTRINSICS_MAX_RMS_PX, Intrinsics
from codex_caliper.ui.calibration_view import IntrinsicCalibrationView, PlaneCalibrationView
from codex_caliper.ui.main_window import MainWindow
from tests.fixtures.synthetic import make_banana_scene

# ---------------------------------------------------------------------------
# Datos sinteticos compartidos
# ---------------------------------------------------------------------------

PATTERN_SIZE = (9, 6)
SQUARE_MM = 25.0
IMAGE_SIZE = (1920, 1080)

KNOWN_FX = KNOWN_FY = 1400.0
KNOWN_CX, KNOWN_CY = IMAGE_SIZE[0] / 2.0, IMAGE_SIZE[1] / 2.0
KNOWN_K = np.array(
    [[KNOWN_FX, 0, KNOWN_CX], [0, KNOWN_FY, KNOWN_CY], [0, 0, 1]], dtype=np.float64
)
KNOWN_DIST = np.array([-0.22, 0.09, 0.0006, -0.0004, 0.015], dtype=np.float64)

# Mismas 15 poses variadas en angulo y profundidad que tests/test_intrinsics.py:
# un set casi frontal y casi identico deja el sistema mal condicionado.
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
    cols, rows = PATTERN_SIZE
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_MM
    return objp


def _clean_poses() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Las 15 poses proyectadas con K/dist conocidos, sin ruido -- RMS bajo."""
    objp = _board_object_points()
    obj_pts, img_pts = [], []
    for rx, ry, rz, tx, ty, tz in SYNTHETIC_POSES:
        rvec = np.deg2rad([rx, ry, rz]).reshape(3, 1)
        tvec = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1)
        projected, _ = cv2.projectPoints(objp, rvec, tvec, KNOWN_K, KNOWN_DIST)
        obj_pts.append(objp)
        img_pts.append(projected.reshape(-1, 2).astype(np.float32))
    return obj_pts, img_pts


def _noisy_poses(seed: int = 7, noise_px: float = 3.0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Las mismas 15 poses, con ruido Gaussiano agregado a cada esquina --
    suficiente para que ``cv2.calibrateCamera`` devuelva un RMS por encima
    de ``INTRINSICS_MAX_RMS_PX`` (0.5 px)."""
    rng = np.random.default_rng(seed)
    obj_pts, img_pts = _clean_poses()
    noisy_img_pts = [pts + rng.normal(scale=noise_px, size=pts.shape).astype(np.float32) for pts in img_pts]
    return obj_pts, noisy_img_pts


def _gradient_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Cuadro con textura real (no plano): un degradado en dos canales, para
    que la correccion de distorsion produzca una diferencia de pixeles
    MEDIBLE -- un cuadro de color uniforme no la produciria (verificado)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = np.tile(np.linspace(0, 255, width), (height, 1)).astype(np.uint8)
    frame[..., 1] = np.tile(np.linspace(0, 255, height), (width, 1)).T.astype(np.uint8)
    frame[..., 2] = 128
    return frame


def _distorted_intrinsics(image_size: tuple[int, int] = (640, 480)) -> Intrinsics:
    fx = fy = 500.0
    cx, cy = image_size[0] / 2.0, image_size[1] / 2.0
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([-0.3, 0.08, 0.0, 0.0, 0.0], dtype=np.float64)
    return Intrinsics(camera_matrix=k, dist_coeffs=dist, rms_px=0.2, image_size=image_size)


def _fake_finder(results: list):
    """Doble inyectado de ``find_chessboard``: consume ``results`` en orden,
    ignorando el contenido real del cuadro. No requiere una foto real de
    tablero (prohibido por el entorno de esta tarea)."""
    it = iter(results)

    def _finder(frame, pattern_size):
        return next(it)

    return _finder


MARKERS_MM = {1: (0.0, 0.0), 2: (300.0, 0.0), 3: (0.0, 200.0), 4: (300.0, 200.0)}


def _plane_cfg(**overrides) -> AppConfig:
    base = dict(marker_thickness_mm=2.0, marker_layout=dict(MARKERS_MM))
    base.update(overrides)
    return AppConfig(**base)


def _four_marker_frame(mm_per_px: float = 0.3) -> np.ndarray:
    return make_banana_scene([], mm_per_px, markers=MARKERS_MM, marker_dict="DICT_6X6_50")


def _three_marker_frame(mm_per_px: float = 0.3) -> np.ndarray:
    three = {k: v for k, v in MARKERS_MM.items() if k != 4}
    return make_banana_scene([], mm_per_px, markers=three, marker_dict="DICT_6X6_50")


# ---------------------------------------------------------------------------
# IntrinsicCalibrationView -- Ruling 2, prueba 1: el toggle antes/despues
# ---------------------------------------------------------------------------


def test_undistortion_toggle_actually_changes_the_displayed_pixels(qtbot):
    """El toggle tiene que CORREGIR el cuadro, no solo fijar una bandera que
    su propia consulta lee de vuelta -- ver el informe de la tarea para la
    evidencia de que la version del brief pasa con una implementacion que no
    hace nada."""
    view = IntrinsicCalibrationView(image_size=(640, 480))
    qtbot.addWidget(view)
    view._intrinsics = _distorted_intrinsics((640, 480))  # calibracion ya aceptada

    frame = _gradient_frame()
    view.set_live_frame(frame)

    assert view.undistort_preview is False
    before = view.displayed_frame.copy()
    np.testing.assert_array_equal(before, frame)
    assert view.displayed_frame_is_undistorted() is False

    view.undistort_preview = True
    after = view.displayed_frame

    assert not np.array_equal(before, after), "el cuadro mostrado no cambio -- el toggle no corrigio nada"
    assert view.displayed_frame_is_undistorted() is True

    # Viaje de vuelta: apagar el toggle tiene que restaurar el crudo.
    view.undistort_preview = False
    np.testing.assert_array_equal(view.displayed_frame, frame)
    assert view.displayed_frame_is_undistorted() is False


def test_undistortion_toggle_without_intrinsics_shows_raw_and_reports_false(qtbot):
    """Sin una calibracion ACEPTADA todavia, activar el toggle no puede
    inventar una correccion: debe seguir mostrando el crudo."""
    view = IntrinsicCalibrationView(image_size=(640, 480))
    qtbot.addWidget(view)
    view.set_live_frame(_gradient_frame())

    view.undistort_preview = True

    np.testing.assert_array_equal(view.displayed_frame, view._raw_frame)
    assert view.displayed_frame_is_undistorted() is False


# ---------------------------------------------------------------------------
# IntrinsicCalibrationView -- Ruling 2, prueba 2: umbral de RMS
# ---------------------------------------------------------------------------


def test_calibration_is_accepted_at_or_below_the_rms_threshold(qtbot):
    view = IntrinsicCalibrationView(image_size=IMAGE_SIZE)
    qtbot.addWidget(view)
    view._object_points, view._image_points = _clean_poses()

    accepted = view.run_calibration()

    assert accepted is True
    assert view.result_accepted is True
    assert view.rejection_message() == ""
    assert view.intrinsics is not None
    assert view.intrinsics.rms_px < INTRINSICS_MAX_RMS_PX


def test_calibration_is_rejected_above_the_rms_threshold_and_pins_the_real_numbers(qtbot):
    view = IntrinsicCalibrationView(image_size=IMAGE_SIZE)
    qtbot.addWidget(view)
    view._object_points, view._image_points = _noisy_poses()

    accepted = view.run_calibration()

    assert accepted is False
    assert view.result_accepted is False
    assert view.intrinsics is None, "una calibracion rechazada no debe quedar activa para medir"

    message = view.rejection_message()
    # No alcanza con que el mensaje CONTENGA "0.5" en cualquier parte (eso lo
    # satisface cualquier mensaje con un 0.5 de casualidad) -- se fija el
    # umbral real exportado por el modulo de intrinsecos...
    assert f"{INTRINSICS_MAX_RMS_PX:.2f}" in message
    # ...y el RMS real que de verdad produjo ESTE intento, no un numero fijo.
    assert view._last_attempt is not None
    assert f"{view._last_attempt.rms_px:.2f}" in message
    assert view._last_attempt.rms_px >= INTRINSICS_MAX_RMS_PX


def test_run_calibration_requires_at_least_one_captured_pose(qtbot):
    view = IntrinsicCalibrationView(image_size=IMAGE_SIZE)
    qtbot.addWidget(view)

    with pytest.raises(ValueError):
        view.run_calibration()


# ---------------------------------------------------------------------------
# IntrinsicCalibrationView -- Ruling 2, prueba 3: contador de poses
# ---------------------------------------------------------------------------


def test_capture_counter_only_increases_on_poses_where_the_chessboard_is_found(qtbot, monkeypatch):
    found_corners = np.zeros((54, 1, 2), dtype=np.float32)
    # Secuencia: encontrado, encontrado, NO encontrado, encontrado, NO encontrado.
    monkeypatch.setattr(
        "codex_caliper.ui.calibration_view.find_chessboard",
        _fake_finder([found_corners, found_corners, None, found_corners, None]),
    )

    view = IntrinsicCalibrationView(image_size=IMAGE_SIZE)
    qtbot.addWidget(view)
    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    results = [view.capture_pose(dummy_frame) for _ in range(5)]

    assert results == [True, True, False, True, False]
    assert view.pose_count == 3, "una pose SIN tablero no debe contarse como 'accepted'"


# ---------------------------------------------------------------------------
# PlaneCalibrationView -- Ruling 2, prueba 4: residual real en mm
# ---------------------------------------------------------------------------


def test_plane_view_reports_the_real_residual_value_in_millimetres(qtbot):
    cfg = _plane_cfg()
    view = PlaneCalibrationView(cfg)
    qtbot.addWidget(view)

    frame = _four_marker_frame()
    view.update_frame(frame)

    expected_plane = compute_plane_calibration(view._detections, cfg)
    assert expected_plane is not None
    assert view.residual_text() == f"{expected_plane.residual_mm:.3f} mm"
    # No alcanza con que la cadena CONTENGA "mm" (cualquier texto con esas
    # letras la satisface) -- el residual real medido en esta escena NO es
    # cero, asi que ademas se fija que el numero reportado sea ese mismo.
    assert expected_plane.residual_mm > 0.0


def test_plane_view_lists_detected_marker_ids_and_ok_state(qtbot):
    cfg = _plane_cfg()
    view = PlaneCalibrationView(cfg)
    qtbot.addWidget(view)

    view.update_frame(_four_marker_frame())

    assert view.marker_ids() == [1, 2, 3, 4]
    assert view.state is CalibrationState.OK
    # ``mm_per_px`` es la escala del LIENZO rectificado (``cfg.canvas_mm_per_px``),
    # no la escala a la que se renderizo la escena sintetica de entrada --
    # son dos numeros distintos a proposito (ver homography.py).
    assert view.mm_per_px_text() == f"{cfg.canvas_mm_per_px:.3f} mm/px"


def test_plane_view_shows_degraded_state_with_three_markers(qtbot):
    """Carry-over de la Ruling 4: con 3 marcadores la calibracion sigue
    funcionando (DEGRADED), y esta vista -- que es donde un operador mira el
    detalle de calibracion -- tiene que decirlo."""
    cfg = _plane_cfg()
    view = PlaneCalibrationView(cfg)
    qtbot.addWidget(view)

    view.update_frame(_three_marker_frame())

    assert view.marker_ids() == [1, 2, 3]
    assert view.state is CalibrationState.DEGRADED


def test_plane_view_reports_blocked_for_a_processed_frame_with_zero_markers(qtbot):
    """Un cuadro YA PROCESADO sin ningun marcador detectado es BLOCKED, no
    "todavia sin datos" -- esas dos cosas no pueden confundirse en
    ``state`` (se distinguen con un cuadro procesado de verdad, blanco
    liso, de las que ``detect_markers`` no encuentra nada)."""
    cfg = _plane_cfg()
    view = PlaneCalibrationView(cfg)
    qtbot.addWidget(view)

    blank_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    view.update_frame(blank_frame)

    assert view.marker_ids() == []
    assert view.state is CalibrationState.BLOCKED


def test_plane_view_before_any_frame_reports_empty_state_without_crashing(qtbot):
    cfg = _plane_cfg()
    view = PlaneCalibrationView(cfg)
    qtbot.addWidget(view)

    assert view.state is None
    assert view.marker_ids() == []
    assert "mm" not in view.residual_text().lower() or "—" in view.residual_text()


# ---------------------------------------------------------------------------
# Ruling 3 -- measurement_enabled / blocked_reason ya existen (Task 18);
# aqui se ejercitan, no se reimplementan.
# ---------------------------------------------------------------------------


def test_measurement_mode_is_blocked_without_valid_intrinsics(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window.load_intrinsics(None)

    assert window.measurement_enabled is False
    assert "calibration" in window.blocked_reason().lower()


def test_accepting_a_calibration_in_the_view_unblocks_measurement_in_the_window(qtbot):
    """La vista de calibracion intrinseca tiene que poder desbloquear la
    medicion de ``MainWindow`` a traves de ``load_intrinsics`` -- sin
    reimplementar esa compuerta (Ruling 3), solo conectandola."""
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.measurement_enabled is False

    window.intrinsic_calibration_view._object_points, window.intrinsic_calibration_view._image_points = (
        _clean_poses()
    )
    window.intrinsic_calibration_view.image_size = IMAGE_SIZE
    accepted = window.intrinsic_calibration_view.run_calibration()

    assert accepted is True
    assert window.measurement_enabled is True
    assert window.blocked_reason() == ""


def test_rejecting_a_calibration_does_not_unblock_measurement(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window.intrinsic_calibration_view._object_points, window.intrinsic_calibration_view._image_points = (
        _noisy_poses()
    )
    window.intrinsic_calibration_view.image_size = IMAGE_SIZE
    accepted = window.intrinsic_calibration_view.run_calibration()

    assert accepted is False
    assert window.measurement_enabled is False
    assert "calibration" in window.blocked_reason().lower()


# ---------------------------------------------------------------------------
# Navegacion entre vistas (spec S8: pantallas separadas de calibracion)
# ---------------------------------------------------------------------------


def test_main_window_navigates_to_the_calibration_views_and_back(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.current_view_name() == "live"

    window.show_view("intrinsic_calibration")
    assert window.current_view_name() == "intrinsic_calibration"
    assert window._stack.currentWidget() is window.intrinsic_calibration_view

    window.show_view("plane_calibration")
    assert window.current_view_name() == "plane_calibration"
    assert window._stack.currentWidget() is window.plane_calibration_view

    window.show_view("live")
    assert window.current_view_name() == "live"

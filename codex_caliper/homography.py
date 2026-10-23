"""Homografia del plano metrico: ajuste, residual de confianza y rectificacion.

Cuatro marcadores ArUco aportan 4 esquinas cada uno -- 16 correspondencias
pixel/mundo para 8 incognitas -- por eso el sistema queda sobredeterminado y
el desacuerdo entre esos 16 puntos es una metrica de confianza medible
(``residual_mm``), no un detalle interno. ``cv2.findHomography`` se llama con
``method=0`` (ajuste por minimos cuadrados, sin RANSAC): las 16 esquinas
vienen de marcadores decodificados e identificados por id -- no hay
outliers que descartar -- y RANSAC sobre puntos ya confiables arriesga
descartar datos buenos por un ajuste peor.

La homografia se recalcula cuadro a cuadro: los 4 marcadores estan siempre
en el campo de vision, asi que recalcularla no cuesta nada, y hace que el
sistema se autorrepare si el soporte de la camara se golpea -- el siguiente
cuadro simplemente se recalibra solo, en vez de quedar corrompido en
silencio hasta un reinicio manual.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np

from codex_caliper.aruco_tools import CalibrationState, MarkerDetection, calibration_state
from codex_caliper.config import AppConfig


class DegenerateMarkerLayout(Exception):
    """Se lanza cuando ``cv2.findHomography`` no puede ajustar NINGUNA
    homografia porque los puntos de mundo que implica ``cfg.marker_layout``
    (combinado con ``cfg.marker_length_mm``) son degenerados -- tipicamente
    colineales -- y ``findHomography`` devuelve ``None`` en vez de una
    matriz. Sin esta guardia, ese ``None`` se propaga a
    ``cv2.perspectiveTransform`` y revienta con un assert crudo de OpenCV
    C++ ("scn + 1 == m.cols"), ilegible para quien opera el sistema.

    A diferencia de ``BLOCKED`` (menos de 3 marcadores visibles, una
    condicion TRANSITORIA que se resuelve sola en el siguiente cuadro si el
    marcador vuelve a aparecer), esto es un error de CONFIGURACION: un
    ``marker_layout`` con centros y/o un ``marker_length_mm`` que colapsan
    los puntos de mundo a una linea va a fallar exactamente igual en todos
    los cuadros siguientes hasta que alguien corrija el archivo de config.
    Devolver ``None`` aqui, igual que en ``BLOCKED``, confundiria a quien
    consume la API -- "esperá, el marcador va a reaparecer" -- cuando en
    realidad el problema no se autocorrige nunca. Por eso se lanza, igual
    que ``IntrinsicsMismatch`` en ``intrinsics.py`` para un problema de
    configuracion igualmente persistente (un anillo de zoom movido).
    """


@dataclasses.dataclass
class PlaneCalibration:
    """Resultado de ajustar la homografia del plano metrico en un frame.

    ``H`` mapea pixeles de imagen a pixeles del lienzo rectificado (no a
    milimetros directamente): el lienzo tiene una escala fija y conocida
    (``mm_per_px``, igual a ``cfg.canvas_mm_per_px``), asi que una distancia
    medida sobre el lienzo se convierte a milimetros con una simple
    multiplicacion. ``residual_mm`` es el RMS, en milimetros, del error de
    reproyeccion de los puntos usados para ajustar ``H`` -- la metrica de
    confianza que ve el operador. ``state`` es ``OK`` con los 4 marcadores o
    ``DEGRADED`` con 3; con menos no existe esta instancia en absoluto (ver
    ``compute_plane_calibration``).

    ADVERTENCIA sobre que tan sensible es ``residual_mm``: es un RMS
    GLOBAL sobre las 16 (o 12) esquinas, y el ajuste de minimos cuadrados
    absorbe parte del error de UN solo marcador mal ubicado repartiendolo
    entre los demas puntos, que siguen siendo correctos. Medido sobre un
    layout de 300x200 mm con una camara sintetica conocida: desplazar UN
    marcador produce apenas ~0.065 mm de ``residual_mm`` por mm de error (un
    marcador desplazado 12 mm -- medio cuerpo de marcador -- sube el
    residual a solo ~0.77-0.96 mm, segun cual). Se midio tambien el RMS por
    marcador (las 4 esquinas de cada uno) y su maximo entre los 4: resulta
    apenas 1.2x-1.6x mas sensible que el RMS global, no una mejora
    cualitativa -- por eso NO se agrega como campo aparte. ``residual_mm``
    es un indicador honesto de "que tan bien ajusta el conjunto", NO un
    indicador preciso de "hay UN marcador fuera de lugar": un numero bajo
    no descarta esa posibilidad. Ver el informe de Task 5 (fix round 1,
    Finding 3) para la tabla de sensibilidad medida.
    """

    H: np.ndarray
    residual_mm: float
    state: CalibrationState
    mm_per_px: float


def world_points_for(detections: list[MarkerDetection], cfg: AppConfig) -> np.ndarray:
    """Esquinas de cada marcador detectado, en milimetros del mundo.

    Para cada deteccion, a partir del centro declarado en
    ``cfg.marker_layout`` y del lado ``cfg.marker_length_mm``, genera sus 4
    esquinas en el MISMO orden que entrega ``cv2.aruco``: superior-izquierda,
    superior-derecha, inferior-derecha, inferior-izquierda del propio
    marcador (horario, partiendo de la esquina superior izquierda vista en
    el marco del marcador). Los marcadores del rig se montan sin rotacion
    respecto al layout declarado, asi que esa esquina "superior izquierda
    del marcador" coincide con la esquina de menor x y menor y en milimetros.

    Si este orden no coincidiera con el de ``detection.corners``, la
    homografia se ajustaria con correspondencias desparejadas -- la esquina
    de imagen de un punto emparejada con la esquina de mundo de otro -- y
    produciria una transformacion de apariencia plausible pero equivocada
    (ver ``tests/test_homography.py::test_world_points_for_corner_order_matches_opencv_detection``,
    que verifica el orden real de OpenCV en vez de asumirlo).
    """
    half = cfg.marker_length_mm / 2.0
    rows: list[list[list[float]]] = []
    for detection in detections:
        cx, cy = cfg.marker_layout[detection.id]
        rows.append(
            [
                [cx - half, cy - half],
                [cx + half, cy - half],
                [cx + half, cy + half],
                [cx - half, cy + half],
            ]
        )
    if not rows:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2)


def _image_points_for(detections: list[MarkerDetection]) -> np.ndarray:
    """Las esquinas de imagen de ``detections``, en el mismo orden y cantidad
    que entrega ``world_points_for`` para la misma lista."""
    if not detections:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray([d.corners for d in detections], dtype=np.float64).reshape(-1, 2)


def _canvas_bounds_mm(cfg: AppConfig) -> tuple[float, float, float, float]:
    """Rectangulo ``(min_x, min_y, max_x, max_y)`` en mm que cubre el lienzo.

    Compartido por ``canvas_size_for`` y ``compute_plane_calibration``: ambos
    deben acordar el mismo origen de mundo, para que las esquinas de
    marcador usadas al ajustar la homografia caigan en el mismo pixel de
    lienzo que luego describe ``canvas_size_for``. Cubre el rectangulo que
    abren los CENTROS declarados en ``cfg.marker_layout``, mas un margen de
    medio lado de marcador por borde, para que el cuerpo entero de cada
    marcador quede dentro del lienzo.
    """
    margin = cfg.marker_length_mm / 2.0
    xs = [pos[0] for pos in cfg.marker_layout.values()]
    ys = [pos[1] for pos in cfg.marker_layout.values()]
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def canvas_size_for(cfg: AppConfig) -> tuple[int, int]:
    """Tamano en pixeles ``(ancho, alto)`` del lienzo cenital rectificado.

    Se deriva del rectangulo que abren los marcadores declarados en
    ``cfg.marker_layout`` (ver ``_canvas_bounds_mm``) a la escala fija
    ``cfg.canvas_mm_per_px``.
    """
    min_x, min_y, max_x, max_y = _canvas_bounds_mm(cfg)
    width_px = max(1, math.ceil((max_x - min_x) / cfg.canvas_mm_per_px))
    height_px = max(1, math.ceil((max_y - min_y) / cfg.canvas_mm_per_px))
    return width_px, height_px


def compute_plane_calibration(detections: list[MarkerDetection], cfg: AppConfig) -> PlaneCalibration | None:
    """Ajusta la homografia imagen->lienzo a partir de las esquinas detectadas.

    Devuelve ``None`` si ``calibration_state`` clasifica las detecciones como
    ``BLOCKED`` (menos de 3 marcadores distintos): el caller nunca debe
    recibir una ``PlaneCalibration`` que no pueda sostener una medicion. Con
    3 marcadores (``DEGRADED``) el ajuste sigue adelante con los 12 puntos
    disponibles, y lo declara en ``state``.

    El destino de ``cv2.findHomography`` son pixeles de LIENZO, no
    milimetros: los puntos de mundo (mm, de ``world_points_for``) se
    convierten primero a pixeles de lienzo restando el origen de
    ``_canvas_bounds_mm`` y dividiendo por ``cfg.canvas_mm_per_px`` -- el
    mismo origen y escala que usa ``canvas_size_for``. ``residual_mm``
    reproyecta las esquinas de imagen con la ``H`` ajustada y mide el RMS del
    desacuerdo contra esos mismos puntos de lienzo, convertido a milimetros.

    Lanza ``DegenerateMarkerLayout`` si ``cv2.findHomography`` no puede
    ajustar ninguna homografia (puntos de mundo colineales, tipicamente por
    un ``cfg.marker_layout``/``marker_length_mm`` con un error de tipeo) --
    ver esa excepcion para por que se lanza en vez de devolver ``None``.
    """
    state = calibration_state(detections)
    if state is CalibrationState.BLOCKED:
        return None

    world_mm = world_points_for(detections, cfg)
    image_px = _image_points_for(detections)

    min_x, min_y, _max_x, _max_y = _canvas_bounds_mm(cfg)
    mm_per_px = cfg.canvas_mm_per_px
    canvas_px = (world_mm - np.array([min_x, min_y])) / mm_per_px

    H, _inlier_mask = cv2.findHomography(image_px, canvas_px, method=0)
    if H is None:
        raise DegenerateMarkerLayout(
            "Could not fit a plane homography: the marker layout in the configuration "
            "produces degenerate (collinear) world points, so no homography can explain "
            "them. This is a configuration error -- check cfg.marker_layout and "
            "cfg.marker_length_mm for a typo -- not a transient loss of markers, and it "
            "will not resolve on its own on the next frame."
        )

    reprojected = cv2.perspectiveTransform(
        image_px.reshape(-1, 1, 2).astype(np.float64), H
    ).reshape(-1, 2)
    error_px = reprojected - canvas_px
    residual_mm = float(np.sqrt(np.mean(np.sum(error_px**2, axis=1))) * mm_per_px)

    return PlaneCalibration(H=H, residual_mm=residual_mm, state=state, mm_per_px=mm_per_px)


def rectify(frame: np.ndarray, plane: PlaneCalibration, canvas_size_px: tuple[int, int]) -> np.ndarray:
    """Rectifica ``frame`` al lienzo cenital usando la homografia ya ajustada.

    Simple envoltura de ``cv2.warpPerspective``: toda la logica de armar la
    homografia correcta vive en ``compute_plane_calibration``.
    """
    return cv2.warpPerspective(frame, plane.H, canvas_size_px)

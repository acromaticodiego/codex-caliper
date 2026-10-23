"""Deteccion de marcadores ArUco y clasificacion del estado de calibracion.

Los cuatro marcadores del rig (``DICT_6X6_50``, ids 1-4) se imprimen en FDM:
su "blanco" es en realidad un gris sucio con textura visible de capas, lo que
rompe la *decodificacion* (no solo la deteccion de candidatos). Medido sobre
las capturas reales del montaje, el defecto de OpenCV para
``perspectiveRemovePixelPerCell`` (4 px/celda) deja el marcador id 3
indecodificable en todos los frames; subirlo a 8 lo arregla en los cuatro.
Ver ``build_detector_params`` para el detalle de por que.
"""

from __future__ import annotations

import dataclasses
import enum

import cv2
import numpy as np

from codex_caliper.config import AppConfig


@dataclasses.dataclass
class MarkerDetection:
    """Un marcador ArUco decodificado en un frame.

    ``corners`` tiene forma 4x2, en el orden que entrega OpenCV (esquina
    superior izquierda primero, en sentido horario). No se usa
    ``frozen=True``: la igualdad/hash por defecto de un dataclass frozen con
    un campo ``np.ndarray`` falla ("truth value ambiguous") en cuanto algo lo
    compara o lo mete en un set, y ``corners`` es mutable de todos modos.
    """

    id: int
    corners: np.ndarray


class CalibrationState(enum.Enum):
    """Cuanta confianza de calibracion metrica sostiene el frame actual.

    ``OK`` exige los 4 marcadores del rig; con 3 la calibracion es posible
    pero ``DEGRADED`` (menos redundancia frente a error de deteccion); con
    menos de 3 no hay suficiente geometria y queda ``BLOCKED``.
    """

    OK = "ok"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


def build_detector_params() -> cv2.aruco.DetectorParameters:
    """Parametros de deteccion ajustados para marcadores impresos en 3D.

    Al decodificar, ArUco endereza el marcador detectado a un cuadrado
    canonico y decide cada bit muestreando una celda de NxN pixeles del
    resultado. Con el defecto de OpenCV (N=4, 16 px/celda), una sola linea
    de capa de la impresion FDM -- o una mota de la superficie -- puede
    dominar esa muestra y voltear el bit. Con N=8 (64 px/celda) la textura
    se promedia y el muestreo deja de ser ruidoso: esto es lo que hace
    decodificable el marcador id 3 del montaje real, que con el defecto no
    decodifica en ningun frame.

    Subir ``errorCorrectionRate`` (p.ej. a 0.8) tambien hace "aparecer" el
    marcador id 3, pero es la palanca equivocada: no mejora la muestra, sino
    que acepta mas bits erroneos como validos. La superficie real produce
    entre 35 y 50 cuadrilateros candidatos espurios por frame (veta de la
    impresion), asi que aflojar la validacion de bits invita a falsos
    positivos sobre esos candidatos. Se prefiere muestrear mejor, no validar
    menos -- de ahi que aqui no se toque ``errorCorrectionRate``.

    ``adaptiveThreshWinSizeMin/Max/Step`` se ensanchan porque el "blanco" de
    la impresion es un gris sucio con textura, no un blanco uniforme: un
    rango mas amplio de ventanas de umbralado adaptativo sostiene la
    binarizacion bajo esa textura e iluminacion desigual. El refinamiento
    subpixel (``CORNER_REFINE_SUBPIX``) es lo que sostiene el objetivo de
    +/-1 mm del spec sobre la posicion de las esquinas; no tiene relacion
    con el problema de decodificacion de bits.
    """
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 6
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.perspectiveRemovePixelPerCell = 8
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


def detect_markers(frame: np.ndarray, cfg: AppConfig) -> list[MarkerDetection]:
    """Detecta los marcadores ArUco del rig en ``frame``.

    Usa ``cfg.aruco_dictionary`` y los parametros de ``build_detector_params``
    para decodificar, y descarta cualquier id decodificado que no figure en
    ``cfg.marker_ids`` -- un decode espurio de otro id (ruido de la textura
    de la impresion) no debe llegar al resto del pipeline.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.aruco_dictionary))
    detector = cv2.aruco.ArucoDetector(dictionary, build_detector_params())
    corners, ids, _rejected = detector.detectMarkers(frame)

    if ids is None:
        return []

    allowed_ids = set(cfg.marker_ids)
    detections = []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)
        if marker_id not in allowed_ids:
            continue
        detections.append(MarkerDetection(id=marker_id, corners=marker_corners.reshape(4, 2)))
    return detections


def calibration_state(detections: list[MarkerDetection]) -> CalibrationState:
    """Clasifica cuanta confianza de calibracion sostienen las detecciones."""
    count = len(detections)
    if count >= 4:
        return CalibrationState.OK
    if count == 3:
        return CalibrationState.DEGRADED
    return CalibrationState.BLOCKED

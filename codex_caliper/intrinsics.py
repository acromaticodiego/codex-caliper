"""Calibracion intrinseca de camara: recuperacion de la matriz de camara y los
coeficientes de distorsion, su persistencia en JSON, y la guardia que impide
usarlos con una captura de otra resolucion.

La lente del montaje real (ELP-USBFHD01M-BFV, varifocal 2.8-12 mm, zoom y foco
ajustados a mano) solo produce intrinsecos validos para la posicion exacta de
los anillos en que se calibro. Si un anillo se mueve -- incluso sin querer, al
manipular el montaje -- la matriz de camara y la distorsion dejan de
corresponder a la realidad y todas las medidas se corrompen en silencio. El
software no puede detectar un anillo movido directamente; lo unico que puede
verificar de forma automatica es que la resolucion guardada coincida con la
que la camara entrega ahora, que es lo que hace ``require_compatible``. El
resto del protocolo (spec sec 5.2: anillos atornillados y marcados con pintura
antes de calibrar) es manual y ``zoom_note`` lo registra en texto libre.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np

# Umbral de aceptacion de la calibracion intrinseca (restriccion global del plan).
INTRINSICS_MAX_RMS_PX = 0.5


class IntrinsicsMismatch(Exception):
    """Se lanza cuando los intrinsecos guardados se calibraron a una resolucion
    distinta de la que la camara esta entregando ahora."""


@dataclasses.dataclass(kw_only=True)
class Intrinsics:
    """Resultado de una calibracion intrinseca, listo para persistir.

    ``image_size`` es ``(width, height)``, igual que lo que recibe
    ``cv2.calibrateCamera``. ``zoom_note`` no participa en ningun calculo:
    es el registro humano de la posicion de los anillos de zoom/foco.
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms_px: float
    image_size: tuple[int, int]
    zoom_note: str = ""

    @property
    def is_acceptable(self) -> bool:
        """True si el RMS de reproyeccion esta bajo el umbral de aceptacion."""
        return self.rms_px < INTRINSICS_MAX_RMS_PX

    def require_compatible(self, current_size: tuple[int, int]) -> None:
        """Lanza ``IntrinsicsMismatch`` si ``current_size`` no coincide con la
        resolucion con la que se calibraron estos intrinsecos.

        Esta es la unica verificacion automatica posible de un cambio de zoom:
        software no puede ver que el anillo se movio, pero si puede ver que la
        imagen que entrega la camara ahora tiene otro tamano.
        """
        stored = tuple(self.image_size)
        current = tuple(current_size)
        if stored != current:
            raise IntrinsicsMismatch(
                f"Stored intrinsics were calibrated at resolution {stored}, but the "
                f"camera is currently delivering {current}. Recalibrate before "
                "measuring -- a zoom or focus ring may have moved."
            )

    def save(self, path: str | Path) -> None:
        """Serializa a JSON. ``camera_matrix``/``dist_coeffs`` son ``ndarray``;
        JSON no tiene tipo de arreglo, asi que se convierten con ``.tolist()``.
        """
        data = {
            "camera_matrix": np.asarray(self.camera_matrix).tolist(),
            "dist_coeffs": np.asarray(self.dist_coeffs).tolist(),
            "rms_px": self.rms_px,
            "image_size": list(self.image_size),
            "zoom_note": self.zoom_note,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Intrinsics":
        """Carga desde JSON, reconstruyendo los ``ndarray`` con ``np.asarray``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64),
            rms_px=float(data["rms_px"]),
            image_size=tuple(data["image_size"]),
            zoom_note=data.get("zoom_note", ""),
        )


def calibrate_from_corners(object_points, image_points, image_size: tuple[int, int]) -> Intrinsics:
    """Envuelve ``cv2.calibrateCamera``: ``object_points``/``image_points`` son
    listas (una entrada por vista/pose) de arreglos de puntos 3D/2D.
    """
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return Intrinsics(
        camera_matrix=camera_matrix,
        # cv2.calibrateCamera devuelve dist_coeffs como fila/columna 2D; se
        # aplana a 1D para que dist_coeffs[0] sea k1 y no el vector entero.
        dist_coeffs=np.asarray(dist_coeffs).ravel(),
        rms_px=float(rms),
        image_size=tuple(image_size),
    )


def undistort(frame: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Corrige la distorsion de lente de ``frame`` usando los intrinsecos dados."""
    return cv2.undistort(frame, intr.camera_matrix, intr.dist_coeffs)


def find_chessboard(frame: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    """Busca un tablero de ajedrez de ``pattern_size`` esquinas internas
    (columnas, filas) en ``frame`` y refina las esquinas a subpixel.

    Devuelve ``None`` si el tablero no se encuentra, en vez de lanzar: no
    encontrarlo es un resultado normal durante la captura interactiva, no un
    error.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

"""Vista de video central: lienzo rectificado a pantalla casi completa con el
cuadro crudo como PiP en una esquina (Task 18, spec S8 variante 1).

Dos decisiones de la encargante gobiernan este modulo:

1. **El PiP se renderiza a resolucion reducida.** El pipeline ya paga un
   ``warpPerspective`` por cuadro para producir ``rectified`` (Task 14); el
   PiP muestra el cuadro CRUDO (``raw_undistorted``), que no necesita ningun
   warp adicional, pero convertirlo a ``QPixmap`` a resolucion completa en
   cada repintado si costaria memoria y ancho de banda de composicion que
   nadie necesita para una miniatura de esquina. Por eso se reescala con
   ``cv2.resize`` ANTES de construir el ``QPixmap``, a una fraccion fija del
   ancho nativo del cuadro crudo (``PIP_SCALE``), con un piso en pixeles
   (``PIP_MIN_WIDTH_PX``) para que la miniatura no degenere a algo
   ilegible en una toma de video real.

2. **``rectified`` puede ser ``None``.** ``FrameResult.rectified`` (Task 14)
   es deliberadamente ``None`` -- no un lienzo en negro -- cuando la
   calibracion esta ``BLOCKED``: no existe homografia, no existe lienzo
   metrico. ``raw_undistorted`` siempre esta presente (la correccion de
   distorsion no depende de los marcadores) precisamente para que la
   interfaz siempre tenga algo que mostrar. Cuando llega ``None``,
   ``set_frame`` muestra el cuadro crudo como area PRINCIPAL en vez de
   dejar la ventana en negro o lanzar, con un aviso dibujado sobre el
   mismo cuadro para que quede claro que no es la vista rectificada.

Dos capas, dos relojes (ronda de correcciones finales, C1)
-----------------------------------------------------------
La vista mantiene las dos capas SEPARADAS -- el lienzo rectificado y el
cuadro crudo -- porque no llegan al mismo ritmo. El crudo llega a la
cadencia de la CAMARA (``MainWindow._on_camera_frame``, un cuadro cada
~33 ms); el lienzo rectificado llega a la cadencia del PIPELINE
(``update_frame_result``, 3-10 por segundo en CPU). La version anterior
tenia una sola entrada (``set_frame``) que reconstruia ambas capas de una
vez, asi que la unica forma de refrescar el crudo era volver a entregar
tambien un lienzo -- y por eso ``MainWindow`` solo podia pintar cuando el
pipeline terminaba: el video entero corria al ritmo de la medicion.

Con ``set_raw_frame`` y ``set_rectified_frame`` separadas, el PiP crudo se
mueve a la cadencia de la camara y el lienzo a la del pipeline, sin que
uno pise al otro. ``set_raw_frame`` solo reconstruye el area PRINCIPAL
cuando no hay lienzo rectificado que mostrar (el camino de respaldo de la
Ruling 2): con un lienzo presente, reconvertirlo a ``QPixmap`` en cada
cuadro de camara seria trabajo puro sin ningun pixel nuevo.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QWidget

from codex_caliper.ui.imaging import bgr_to_qpixmap
from codex_caliper.ui.theme import BG_VOID_HEX

#: Fraccion del ancho nativo del cuadro crudo que ocupa el PiP.
PIP_SCALE = 0.25

#: Piso en pixeles del PiP: por debajo de esto una miniatura de esquina deja
#: de ser legible en una toma de video real, sin importar cuanto reduzca
#: ``PIP_SCALE`` el ancho nativo.
PIP_MIN_WIDTH_PX = 160

#: Margen, en pixeles de widget, entre el PiP y el borde de la vista.
_PIP_MARGIN_PX = 12

#: Texto de aviso dibujado sobre el cuadro crudo cuando no hay lienzo
#: rectificado que mostrar. Texto de interfaz: en ingles.
RAW_FALLBACK_TEXT = "NO CALIBRATION - showing raw feed"

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.7
_BANNER_ORIGIN = (16, 36)
_BANNER_COLOR_BGR = (60, 170, 217)  # ambar/acero, coherente con el aviso de la barra de estado


def _downscale_for_pip(frame: np.ndarray, *, scale: float = PIP_SCALE, min_width_px: int = PIP_MIN_WIDTH_PX) -> np.ndarray:
    """Reescala ``frame`` para el PiP, antes de convertirlo a ``QPixmap``.

    Si el cuadro de entrada ya es mas chico que el piso (tipico solo en
    pruebas con cuadros sinteticos pequenos), se devuelve tal cual: no
    tiene sentido AGRANDAR una miniatura.
    """
    height, width = frame.shape[:2]
    target_width = max(min_width_px, round(width * scale))
    if target_width >= width:
        return frame
    target_height = max(1, round(height * target_width / width))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _draw_fallback_banner(frame: np.ndarray, text: str) -> np.ndarray:
    """Dibuja ``text`` con un trazo de contorno negro sobre una COPIA de ``frame``."""
    out = frame.copy()
    cv2.putText(out, text, _BANNER_ORIGIN, _FONT, _FONT_SCALE, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, text, _BANNER_ORIGIN, _FONT, _FONT_SCALE, _BANNER_COLOR_BGR, 2, cv2.LINE_AA)
    return out


class VideoView(QWidget):
    """Lienzo central: rectificado a pantalla casi completa + PiP crudo en una esquina.

    ``set_frame`` es la unica via de entrada de pixeles nuevos; el dibujo
    real ocurre en ``paintEvent`` a partir de los ``QPixmap`` ya preparados
    (``main_pixmap``, ``pip_pixmap``), nunca reconstruyendolos en cada
    repintado -- la conversion desde ``numpy`` ocurre una vez por cuadro,
    no una vez por repintado.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pip_visible = True
        self.main_pixmap = QPixmap()
        self.pip_pixmap = QPixmap()
        self.is_showing_raw_fallback = False
        self._rectified_pixmap: QPixmap | None = None
        self._raw_frame: np.ndarray | None = None

        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(BG_VOID_HEX))
        self.setPalette(palette)

    @property
    def pip_visible(self) -> bool:
        """``True`` si el PiP se dibuja. Controlado solo por ``toggle_pip``."""
        return self._pip_visible

    def toggle_pip(self) -> None:
        """Alterna la visibilidad del PiP (tecla P en ``MainWindow``).

        El autor necesita poder esconderlo para tomas de cierre limpias
        (spec S8) Y volver a mostrarlo -- un alternador que solo apagara
        seria una tecla rota a mitad de camino.
        """
        self._pip_visible = not self._pip_visible
        self.update()

    def set_frame(self, rectified: np.ndarray | None, raw_frame: np.ndarray) -> None:
        """Actualiza las DOS capas de una vez, a partir de un cuadro del pipeline.

        ``rectified`` puede ser ``None`` (calibracion ``BLOCKED``, Task 14);
        en ese caso el area principal muestra ``raw_frame`` con un aviso
        dibujado encima, en vez de quedar en negro. El PiP SIEMPRE se
        construye a partir de ``raw_frame`` (esa es su definicion, spec S8:
        "el crudo como PiP en la esquina"), independientemente de si
        tambien es lo que se usa como area principal.
        """
        self._store_rectified(rectified)
        self._store_raw(raw_frame)
        self._refresh_main_pixmap()
        self.update()

    def set_raw_frame(self, raw_frame: np.ndarray) -> None:
        """Refresca SOLO la capa cruda (el PiP), a la cadencia de la camara.

        Es la via que usa ``MainWindow._on_camera_frame`` en cada cuadro que
        entrega la camara, sin esperar al pipeline. El area principal se
        reconstruye unicamente si no hay lienzo rectificado vigente: en ese
        caso el crudo ES el area principal (camino de respaldo de la Ruling
        2) y tiene que moverse; con un lienzo presente, reconvertirlo en
        cada cuadro de camara seria trabajo sin ningun pixel nuevo.
        """
        self._store_raw(raw_frame)
        if self._rectified_pixmap is None:
            self._refresh_main_pixmap()
        self.update()

    def set_rectified_frame(self, rectified: np.ndarray | None) -> None:
        """Refresca SOLO la capa rectificada, a la cadencia del pipeline.

        ``None`` la DESCARTA (calibracion ``BLOCKED``): el area principal
        pasa al camino de respaldo y desde ese momento sigue a la camara,
        en vez de congelarse en el ultimo lienzo valido -- que es lo que
        haria si un ``None`` simplemente se ignorara.
        """
        self._store_rectified(rectified)
        self._refresh_main_pixmap()
        self.update()

    def _store_rectified(self, rectified: np.ndarray | None) -> None:
        self._rectified_pixmap = None if rectified is None else bgr_to_qpixmap(rectified)

    def _store_raw(self, raw_frame: np.ndarray) -> None:
        self._raw_frame = raw_frame
        self.pip_pixmap = bgr_to_qpixmap(_downscale_for_pip(raw_frame))

    def _refresh_main_pixmap(self) -> None:
        """Recompone el area principal: el lienzo si existe, el crudo con su
        aviso si no."""
        self.is_showing_raw_fallback = self._rectified_pixmap is None
        if self._rectified_pixmap is not None:
            self.main_pixmap = self._rectified_pixmap
        elif self._raw_frame is not None:
            self.main_pixmap = bgr_to_qpixmap(
                _draw_fallback_banner(self._raw_frame, RAW_FALLBACK_TEXT)
            )

    def paintEvent(self, event) -> None:  # noqa: D102 - override de Qt
        painter = QPainter(self)
        try:
            if not self.main_pixmap.isNull():
                scaled = self.main_pixmap.scaled(
                    self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                x = (self.width() - scaled.width()) // 2
                y = (self.height() - scaled.height()) // 2
                painter.drawPixmap(x, y, scaled)

            if self._pip_visible and not self.pip_pixmap.isNull():
                x = max(0, self.width() - self.pip_pixmap.width() - _PIP_MARGIN_PX)
                y = max(0, self.height() - self.pip_pixmap.height() - _PIP_MARGIN_PX)
                painter.drawPixmap(x, y, self.pip_pixmap)
        finally:
            painter.end()

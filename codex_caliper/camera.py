"""Captura de camara en su propio hilo Qt.

El modelo de hilos del plan es explicito: hilo de captura -> hilo de
procesamiento -> senales Qt hacia la interfaz, y nunca procesar en el hilo de
la GUI -- eso es lo que congela una interfaz con video. Este modulo es el
primero de esos tres: el unico punto por el que entran frames reales al
sistema.

``open_capture`` esta aislado de ``CameraThread`` a proposito, para poder
sustituirlo por un capture falso en los tests sin levantar hardware ni tocar
el bucle de lectura.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

# Entre fallos de lectura consecutivos, cuanto esperar antes de reintentar.
# La espera es sobre un threading.Event, asi que stop() la interrumpe de
# inmediato en vez de tener que agotarla.
_READ_RETRY_DELAY_S = 0.1

# Fallos de lectura consecutivos tolerados antes de declarar la camara
# perdida. A 0.1s de espera entre intentos, son ~1s de gracia -- suficiente
# para absorber un hipo puntual de USB sin que una desconexion real tarde
# en notificarse.
_MAX_CONSECUTIVE_READ_FAILURES = 10

# Cuanto espera stop() a que el bucle termine antes de retornar igual. Un
# stop() que no esta acotado se cuelga si el hilo esta parado dentro de
# cap.read() con la camara desenchufada -- justo el momento en que alguien
# quiere cerrar la ventana.
_STOP_WAIT_TIMEOUT_MS = 2000


def open_capture(
    index: int,
    *,
    width: int = 1920,
    height: int = 1080,
    fourcc: str = "MJPG",
    backend: int = cv2.CAP_DSHOW,
) -> cv2.VideoCapture:
    """Abre la camara en ``index`` pidiendo MJPEG a ``width``x``height``.

    La ELP-USBFHD01M-BFV entrega 1920x1080 @ 30 fps por USB 2.0; a esa
    resolucion MJPEG es la unica opcion con ancho de banda para 30 fps, YUY2
    sin comprimir no alcanza. Lo que el driver concede de verdad puede no
    coincidir con lo pedido -- por eso esta funcion solo pide, y quien la
    llama debe leer las propiedades reales despues de abrir.
    """
    capture = cv2.VideoCapture(index, backend)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


class CameraThread(QThread):
    """Hilo dedicado a leer frames de una camara y emitirlos por senal Qt.

    ``frame_ready`` lleva siempre una COPIA del arreglo que devolvio
    ``cap.read()``, nunca la referencia cruda: OpenCV puede reusar el mismo
    buffer interno entre lecturas, y si se emitiera esa referencia el hilo
    receptor leeria memoria que el hilo de captura sigue reescribiendo --
    frames rasgados, de forma intermitente, sin ninguna excepcion que lo
    delate. El costo es una copia por frame (~6 MB a 1080p); es exactamente
    lo que cuesta no tener una carrera de datos.

    ``capture_lost`` se emite si la lectura falla de forma persistente (ver
    ``_MAX_CONSECUTIVE_READ_FAILURES``): la camara real puede desconectarse
    en cualquier momento, y el hilo no debe quedarse reintentando para
    siempre en silencio ni dejar que una excepcion lo tumbe sin que la
    interfaz se entere. Al perder la camara, el bucle termina por su cuenta
    y libera el capture -- igual que si se hubiera llamado a ``stop()``.
    """

    frame_ready = Signal(np.ndarray)
    capture_lost = Signal(str)

    def __init__(
        self,
        index: int,
        width: int = 1920,
        height: int = 1080,
        fourcc: str = "MJPG",
        backend: int = cv2.CAP_DSHOW,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._index = index
        self._width = width
        self._height = height
        self._fourcc = fourcc
        self._backend = backend
        self._stop_requested = threading.Event()
        self._actual_size: tuple[int, int] | None = None

    @property
    def actual_size(self) -> tuple[int, int]:
        """Resolucion que la camara concedio de verdad, leida tras abrir y
        configurar el capture -- puede no coincidir con lo pedido al
        constructor, y los intrinsecos (Task 3) dependen de la real."""
        if self._actual_size is None:
            raise RuntimeError("actual_size no esta disponible antes de iniciar la captura")
        return self._actual_size

    def run(self) -> None:
        """Bucle de lectura. Corre en este hilo, nunca en el de la GUI."""
        self._stop_requested.clear()
        capture = open_capture(
            self._index,
            width=self._width,
            height=self._height,
            fourcc=self._fourcc,
            backend=self._backend,
        )
        try:
            if not capture.isOpened():
                self.capture_lost.emit(f"No se pudo abrir la camara {self._index}")
                return

            self._actual_size = (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )

            consecutive_failures = 0
            while not self._stop_requested.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= _MAX_CONSECUTIVE_READ_FAILURES:
                        self.capture_lost.emit(
                            f"La camara {self._index} dejo de responder tras "
                            f"{consecutive_failures} lecturas fallidas seguidas"
                        )
                        return
                    # Espera interrumpible: si stop() llega durante la espera,
                    # el bucle sale de inmediato en vez de agotar el retraso.
                    if self._stop_requested.wait(_READ_RETRY_DELAY_S):
                        return
                    continue

                consecutive_failures = 0
                self.frame_ready.emit(frame.copy())
        finally:
            capture.release()

    def stop(self) -> None:
        """Pide al bucle que termine y espera, acotado, a que lo haga.

        La espera tiene limite de tiempo a proposito: si el hilo esta
        bloqueado dentro de ``cap.read()`` con la camara desenchufada,
        esperar sin limite colgaria a quien llama a ``stop()`` -- justo
        cuando alguien quiere cerrar la ventana. Forzar la salida de una
        llamada bloqueante de OpenCV desde otro hilo (p. ej. liberando el
        capture de forma concurrente) es comportamiento no definido en
        OpenCV y podria crashear, asi que no se intenta: ``stop()`` deja de
        bloquear dentro del plazo, aunque en el caso extremo de una lectura
        realmente colgada el hilo seguiria vivo de fondo.
        """
        self._stop_requested.set()
        if self.isRunning():
            self.wait(_STOP_WAIT_TIMEOUT_MS)

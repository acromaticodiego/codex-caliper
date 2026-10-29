"""Ventana principal: barra de estado superior, video central y panel lateral
(Task 18, spec S8 variante 1 -- "el video es lo que vende").

Tres decisiones de la encargante gobiernan este modulo, no el brief original:

**El panel es 240 px fijos, no un factor de estiramiento.** Al maximizar la
ventana tiene que crecer el video, nunca el panel; por eso ``SidePanel`` se
agrega al layout de contenido con ``stretch=0`` (su ancho ya es fijo via
``setFixedWidth``) y ``VideoView`` con ``stretch=1``. Los margenes y el
espaciado de TODOS los layouts de nivel superior quedan en cero: con
cualquier margen distinto de cero, ``video_view.width()`` deja de ser
exactamente ``window.width() - 240`` sin que el layout este "mal", solo con
un respiro visual que una prueba exacta no puede distinguir de un error.

**El procesamiento nunca corre en el hilo de la GUI (Ruling 4).**
``Pipeline.process`` hace un warp, la deteccion y toda la cadena de
geometria; llamarlo desde un slot que corre en el hilo de la GUI congela el
repintado de la ventana durante ese tiempo -- exactamente la causa que la
spec nombra como la mas comun de una interfaz de video que se congela. Por
eso ``_PipelineWorker`` vive en su propio ``QThread`` (``moveToThread``), y
``submit_frame`` solo EMITE una senal (``_frame_submitted``) hacia el: Qt
encola esa llamada en el hilo del worker automaticamente porque el emisor y
el receptor tienen afinidad de hilo distinta (``Qt.AutoConnection``, el
default). ``result_ready`` vuelve al reves por el mismo mecanismo, as que
``update_frame_result`` si corre en el hilo de la GUI -- es pura pintura,
nunca medicion.

**La resolucion real de la camara se verifica contra los intrinsecos
(Ruling 3).** ``actual_size`` (Task 16) solo se conoce despues de que el
hilo de captura abrio el dispositivo; la primera vez que llega un cuadro se
verifica con ``Intrinsics.require_compatible`` (Task 3). Si no coincide, la
medicion se BLOQUEA (``measurement_enabled`` pasa a ``False``) y la razon se
muestra en la interfaz -- jamas se mide con unos intrinsecos que no
corresponden a la resolucion real, que es exactamente el tipo de corrupcion
silenciosa que esa guardia existe para evitar. La misma compuerta cubre la
ausencia total de intrinsecos (``load_intrinsics(None)``), que es el
"Review Focus 5" del plan: la Task 19 construira las vistas de calibracion
que llaman a ``load_intrinsics`` con un resultado real; aqui solo se deja
el engranaje que las bloquea mientras no haya una calibracion valida.

Correcciones de la ronda 1 de revision
-----------------------------------------
1. **``draw_results`` (Task 17) se aplica aqui, no en ningun otro sitio.**
   ``update_frame_result`` recibia ``result.rectified`` crudo y se lo
   pasaba directo a ``VideoView`` -- las cotas solo vivian en las tarjetas
   del panel, exactamente el anti-objetivo que la spec S8 nombra ("si los
   numeros viven unicamente en la tabla, el video pasa a ser decorativo").
2. **El video no se congela mientras la medicion esta bloqueada.**
   ``_on_camera_frame`` retornaba sin tocar ``video_view`` cuando
   ``measurement_enabled`` es ``False`` (sin intrinsecos, o resolucion no
   compatible); como el video solo se actualizaba al volver de
   ``submit_frame`` -> worker -> ``result_ready``, el cuadro se congelaba
   aunque la camara siguiera entregando. Ahora ese camino usa
   ``VideoView.set_frame(None, frame)`` directamente -- el mismo camino de
   respaldo de la Ruling 2, reutilizado: "sin calibracion, muestro el
   crudo" es exactamente lo que corresponde aqui tambien.
3. **Se elimino el conteo exacto de marcadores en el caso BLOCKED.**
   La version anterior le sacaba el conteo exacto con una expresion
   regular al texto en ingles del aviso de ``pipeline.py`` -- acoplado a
   una redaccion ajena y sin ninguna prueba que lo fijara. Con 3
   marcadores el estado es DEGRADED y el conteo ya esta disponible
   estructuralmente (``plane.state``); BLOCKED es el UNICO caso que
   necesitaba la regex, y ahi lo unico que el operador necesita saber es
   que esta bloqueado y por que (destapar los marcadores) -- el conteo
   exacto no cambia esa accion. Se cambio por un mensaje generico y se
   borro el acoplamiento en vez de solo fijarlo con una prueba.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from codex_caliper.aruco_tools import CalibrationState
from codex_caliper.camera import CameraThread
from codex_caliper.intrinsics import Intrinsics, IntrinsicsMismatch
from codex_caliper.pipeline import MARKER_COUNT, FrameResult, Pipeline
from codex_caliper.ui.overlay import draw_results
from codex_caliper.ui.side_panel import SidePanel
from codex_caliper.ui.video_view import VideoView

# Paleta: instrumento de medicion, no tablero SaaS -- mismos tokens que
# ``video_view.py``/``side_panel.py``. Ver el informe de la tarea para la
# justificacion completa (por que NO es el fondo "#111" generico, por que
# los colores de senal son desaturados, etc.).
_BG_VOID_HEX = "#171B20"
_SURFACE_HEX = "#1D2227"
_RULE_HEX = "#333A42"
_INK_PRIMARY_HEX = "#E7EAED"
_SIGNAL_WARN_HEX = "#D9A54B"

#: Escala del lienzo usada para el panel lateral cuando no hay calibracion
#: de plano (ver docstring de ``_DEFAULT_MM_PER_PX`` en ``side_panel.py``).
_DEFAULT_MM_PER_PX = 0.2

_STATUS_BAR_HEIGHT_PX = 30

_CHECK = "✓"  # ✓ -- Qt renderiza Unicode real, a diferencia de cv2.putText (ver overlay.py)
_CROSS = "✗"  # ✗
_WARN = "⚠"  # ⚠
_SEPARATOR = " │ "  # │ -- la forma exacta del mockup de la spec S8


class _PipelineWorker(QObject):
    """Envuelve un ``Pipeline`` para correr ``process`` fuera del hilo de la GUI.

    Vive en su propio ``QThread`` (ver ``MainWindow.__init__``); no corre
    nada por si solo, solo reacciona a ``process_frame`` cuando
    ``MainWindow`` le entrega un cuadro.
    """

    result_ready = Signal(object)

    def __init__(self, pipeline: Pipeline) -> None:
        super().__init__()
        self._pipeline = pipeline

    def process_frame(self, frame: np.ndarray) -> None:
        """Slot: procesa un cuadro y emite su ``FrameResult``.

        Este metodo corre en el hilo del worker (por la afinidad que le dio
        ``moveToThread``), nunca en el de la GUI -- es la pieza central de
        la Ruling 4 del encargo.
        """
        result = self._pipeline.process(frame)
        self.result_ready.emit(result)


class MainWindow(QMainWindow):
    """Ventana principal: barra de estado, video central, panel a la derecha.

    ``pipeline`` es opcional a proposito: las pruebas de layout de esta
    tarea no necesitan medir nada, solo alimentar ``update_frame_result``
    con un ``FrameResult`` construido a mano. Cuando se entrega un
    ``pipeline`` real, ``submit_frame``/``connect_camera`` quedan listos
    para que la Task 20 (el punto de entrada) conecte una camara real sin
    tener que resolver otra vez el hilo de procesamiento.
    """

    _frame_submitted = Signal(np.ndarray)

    def __init__(
        self,
        intrinsics: Intrinsics | None = None,
        pipeline: Pipeline | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("codex-caliper")

        self._intrinsics = intrinsics
        self._plane = None
        self._camera_error: str | None = None
        self._resolution_blocked_reason: str | None = None
        self._resolution_checked = False
        self._camera_thread: CameraThread | None = None
        self._worker_thread: QThread | None = None
        self._worker: _PipelineWorker | None = None
        self._status_text = ""
        self._status_warning = False

        self.video_view = VideoView()
        self.side_panel = SidePanel()

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self.video_view, 1)
        content_layout.addWidget(self.side_panel, 0)

        self._status_label = QLabel()
        self._status_label.setFont(QFont("Consolas", 10))
        status_bar_widget = QWidget()
        status_bar_widget.setFixedHeight(_STATUS_BAR_HEIGHT_PX)
        status_bar_widget.setStyleSheet(
            f"background-color: {_SURFACE_HEX}; border-bottom: 1px solid {_RULE_HEX};"
        )
        status_layout = QHBoxLayout(status_bar_widget)
        status_layout.setContentsMargins(12, 0, 12, 0)
        status_layout.setSpacing(0)
        status_layout.addWidget(self._status_label)
        status_layout.addStretch(1)
        self._status_bar_widget = status_bar_widget

        outer = QWidget()
        outer.setStyleSheet(f"background-color: {_BG_VOID_HEX};")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        outer_layout.addWidget(status_bar_widget)
        outer_layout.addWidget(content, 1)
        self.setCentralWidget(outer)

        if pipeline is not None:
            self._worker_thread = QThread(self)
            self._worker = _PipelineWorker(pipeline)
            self._worker.moveToThread(self._worker_thread)
            self._worker_thread.start()
            # Conexion "auto": el emisor (MainWindow) y el receptor (el
            # worker, movido a otro hilo) tienen afinidad de hilo distinta,
            # asi que Qt la encola sola -- no hace falta pedir
            # Qt.QueuedConnection a mano.
            self._frame_submitted.connect(self._worker.process_frame)
            self._worker.result_ready.connect(self.update_frame_result)

        self._refresh_status()

    # -- Entrada de cuadros ------------------------------------------------

    def submit_frame(self, frame: np.ndarray) -> None:
        """Entrega un cuadro para procesar, sin bloquear el hilo de la GUI.

        Solo emite una senal; quien procesa de verdad es
        ``_PipelineWorker.process_frame``, en su propio hilo (Ruling 4).
        Sin un ``pipeline`` configurado no hay nada conectado del otro
        lado -- la senal se emite igual, sin efecto, en vez de lanzar.
        """
        self._frame_submitted.emit(frame)

    def connect_camera(self, camera_thread: CameraThread) -> None:
        """Conecta un ``CameraThread`` real (Task 16) a esta ventana.

        ``capture_lost`` se conecta de forma directa a ``on_capture_lost``
        (Ruling 3): sin esto la senal no tiene ningun oyente y una camara
        desconectada no se nota en la interfaz. La verificacion de
        ``actual_size`` ocurre en el primer cuadro que llegue, no aqui --
        ``actual_size`` todavia no existe hasta que el hilo de captura
        termina de abrir el dispositivo (ver docstring de
        ``CameraThread.actual_size``).
        """
        self._camera_thread = camera_thread
        self._resolution_checked = False
        camera_thread.capture_lost.connect(self.on_capture_lost)
        camera_thread.frame_ready.connect(self._on_camera_frame)

    def _on_camera_frame(self, frame: np.ndarray) -> None:
        """Slot de ``CameraThread.frame_ready``: verifica resolucion una sola
        vez y despues reenvia a ``submit_frame`` mientras la medicion no
        este bloqueada (Ruling 3)."""
        if not self._resolution_checked:
            self._resolution_checked = True
            if self._intrinsics is not None and self._camera_thread is not None:
                try:
                    self._intrinsics.require_compatible(self._camera_thread.actual_size)
                except IntrinsicsMismatch as exc:
                    self._resolution_blocked_reason = str(exc)
                    self._refresh_status()

        if not self.measurement_enabled:
            # Finding 2 (ronda de revision 1): la camara sigue entregando
            # cuadros aunque la medicion este bloqueada (sin intrinsecos
            # todavia, o resolucion no compatible) -- eso no puede congelar
            # el video. ``rectified=None`` reusa el mismo camino de
            # respaldo que la Ruling 2 le dio a ``VideoView`` para el
            # cuadro BLOCKED del pipeline: mostrar el crudo en vez de nada.
            self.video_view.set_frame(None, frame)
            return
        self.submit_frame(frame)

    def on_capture_lost(self, message: str) -> None:
        """Slot de ``CameraThread.capture_lost`` (Ruling 3): una camara
        perdida tiene que notarse en la interfaz, no fallar en silencio."""
        self._camera_error = message
        self._refresh_status()

    # -- Resultados ---------------------------------------------------------

    def update_frame_result(self, result: FrameResult) -> None:
        """Slot de ``_PipelineWorker.result_ready``: solo pintura, nunca medicion.

        Corre en el hilo de la GUI (la senal vuelve encolada desde el hilo
        del worker), pero lo unico que hace es actualizar widgets ya
        construidos -- ``VideoView.set_frame`` ya maneja
        ``rectified is None`` (Ruling 2) sin que esta ventana tenga que
        saber nada de ese caso.

        Finding 1 (ronda de revision 1): ``draw_results`` (Task 17) se
        aplica AQUI, sobre el lienzo, antes de entregarselo a
        ``VideoView`` -- sin esto las cotas solo vivian en las tarjetas del
        panel y el video, que es lo que la spec S8 dice que "vende", quedaba
        decorativo. Se usa ``result.plane.mm_per_px`` (la escala REAL de
        esta calibracion) en vez de la constante del modulo siempre que hay
        un plano -- ``draw_results``/``format_measurement`` ya distinguen
        ese piso de incertidumbre del default generico (ver ``overlay.py``).
        """
        self._plane = result.plane
        mm_per_px = result.plane.mm_per_px if result.plane is not None else _DEFAULT_MM_PER_PX

        rectified = result.rectified
        if rectified is not None:
            rectified = draw_results(rectified, result.results, mm_per_px)

        self.video_view.set_frame(rectified, result.raw_undistorted)
        self.side_panel.update_results(result.results, mm_per_px)
        self._refresh_status()

    # -- Calibracion / bloqueo de medicion -----------------------------------

    def load_intrinsics(self, intrinsics: Intrinsics | None) -> None:
        """Reemplaza los intrinsecos activos (Task 19 los llamara con un
        resultado real de calibracion). Vuelve a verificar la resolucion
        del cuadro siguiente en vez de confiar en una verificacion vieja."""
        self._intrinsics = intrinsics
        self._resolution_checked = False
        self._resolution_blocked_reason = None
        self._refresh_status()

    @property
    def measurement_enabled(self) -> bool:
        """``False`` mientras falte una calibracion valida para la resolucion
        real de la camara, o mientras la camara este perdida (Ruling 3)."""
        return not (self._camera_error or self._resolution_blocked_reason or self._intrinsics is None)

    def blocked_reason(self) -> str:
        """Por que ``measurement_enabled`` es ``False`` en este momento, en
        ingles, lista para mostrarse tal cual. Cadena vacia si no esta
        bloqueado."""
        if self._camera_error:
            return f"Measurement is blocked: {self._camera_error}"
        if self._resolution_blocked_reason:
            return f"Measurement is blocked: {self._resolution_blocked_reason}"
        if self._intrinsics is None:
            return "Measurement is blocked: no intrinsic calibration is loaded."
        return ""

    # -- Barra de estado ------------------------------------------------------

    def status_text(self) -> str:
        """Texto completo de la barra de estado superior, en ingles."""
        return self._status_text

    def status_is_warning(self) -> bool:
        """``True`` si la barra de estado esta mostrando un aviso (calibracion
        degradada o bloqueada, intrinsecos invalidos, camara perdida).
        ``False`` bajo una calibracion sana -- Ruling 1: esta propiedad
        tiene que distinguir "avisa cuando corresponde" de "avisa
        siempre"."""
        return self._status_warning

    def _refresh_status(self) -> None:
        text, warning = self._compute_status()
        self._status_text = text
        self._status_warning = warning
        self._status_label.setText(text)
        color = _SIGNAL_WARN_HEX if warning else _INK_PRIMARY_HEX
        self._status_label.setStyleSheet(f"color: {color};")

    def _compute_status(self) -> tuple[str, bool]:
        if self._camera_error:
            return f"{_CROSS} Camera error: {self._camera_error}", True
        if self._resolution_blocked_reason:
            return f"{_CROSS} {self._resolution_blocked_reason}", True

        intrinsics_text, intrinsics_warning = self._intrinsics_segment()
        markers_text, markers_warning = self._markers_segment()
        scale_text = self._scale_segment()

        text = intrinsics_text + _SEPARATOR + markers_text + _SEPARATOR + scale_text
        return text, intrinsics_warning or markers_warning

    def _intrinsics_segment(self) -> tuple[str, bool]:
        if self._intrinsics is None:
            return f"{_CROSS} Intrinsics: not loaded", True
        if not self._intrinsics.is_acceptable:
            return f"{_CROSS} Intrinsics RMS {self._intrinsics.rms_px:.2f}px", True
        return f"{_CHECK} Intrinsics RMS {self._intrinsics.rms_px:.2f}px", False

    def _markers_segment(self) -> tuple[str, bool]:
        """Segmento de marcadores de la barra de estado.

        Finding 3 (ronda de revision 1): la version anterior recuperaba el
        conteo EXACTO del caso BLOCKED con una expresion regular sobre el
        texto en ingles del aviso de ``pipeline.py`` -- acoplada a una
        redaccion ajena y sin ninguna prueba que la fijara. Con 3
        marcadores el estado es DEGRADED y el conteo ya esta disponible
        estructuralmente (``plane.state``, sin parsear nada); BLOCKED es el
        UNICO caso que necesitaba la regex, y ahi lo unico que importa
        comunicar es que la medicion esta bloqueada -- el conteo exacto (2
        vs. 1 marcador visible) no cambia la accion del operador (destapar
        los marcadores). Se prefiere un mensaje generico, sin esa
        dependencia, en vez de fijar el acoplamiento con una prueba.
        """
        if self._plane is None:
            return f"{_CROSS} calibration blocked", True
        if self._plane.state is CalibrationState.DEGRADED:
            return f"{_WARN} 3/{MARKER_COUNT} markers", True
        return f"{_CHECK} {MARKER_COUNT}/{MARKER_COUNT} markers", False

    def _scale_segment(self) -> str:
        if self._plane is None:
            return "— mm/px"
        return f"{self._plane.mm_per_px:.3f} mm/px"

    # -- Eventos Qt -----------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: D102 - override de Qt
        if event.key() == Qt.Key.Key_P:
            self.video_view.toggle_pip()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: D102 - override de Qt
        """Cierra hilos propios antes de cerrar, para no dejar nada corriendo
        de fondo (``QThread: Destroyed while thread is still running``)."""
        if self._camera_thread is not None:
            self._camera_thread.stop()
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait()
        super().closeEvent(event)

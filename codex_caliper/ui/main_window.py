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

import tempfile
import uuid
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from codex_caliper.aruco_tools import CalibrationState
from codex_caliper.camera import CameraThread
from codex_caliper.config import AppConfig
from codex_caliper.homography import DegenerateMarkerLayout
from codex_caliper.intrinsics import Intrinsics, IntrinsicsMismatch
from codex_caliper.pipeline import MARKER_COUNT, FrameResult, Pipeline
from codex_caliper.ui.calibration_view import IntrinsicCalibrationView, PlaneCalibrationView
from codex_caliper.ui.overlay import draw_marker_overlay, draw_results
from codex_caliper.ui.side_panel import SidePanel
from codex_caliper.ui.theme import (
    BG_VOID_HEX,
    INK_PRIMARY_HEX,
    RULE_HEX,
    SIGNAL_WARN_HEX,
    SURFACE_HEX,
)
from codex_caliper.ui.validation_view import ValidationView
from codex_caliper.ui.video_view import VideoView
from codex_caliper.validation import ValidationLog

_STATUS_BAR_HEIGHT_PX = 30

#: Alto de la linea de avisos de operacion, debajo del video.
_NOTICE_BAR_HEIGHT_PX = 26

#: Separador entre varios avisos en la misma linea.
_NOTICE_SEPARATOR = "   •   "

#: Residual de reproyeccion, en mm, a partir del cual la barra de estado
#: avisa.
#:
#: Es el mismo numero que la lista de verificacion de hardware del README
#: declara como objetivo ("Homography reprojection residual below 1 mm"), no
#: un umbral nuevo inventado aqui. Sin un umbral, el campo volveria a ser un
#: numero que no puede indicar salud -- exactamente el defecto que reemplazar
#: ``mm_per_px`` por ``residual_mm`` existe para cerrar.
RESIDUAL_WARN_MM = 1.0

_CHECK = "✓"  # ✓ -- Qt renderiza Unicode real, a diferencia de cv2.putText (ver overlay.py)
_CROSS = "✗"  # ✗
_WARN = "⚠"  # ⚠
_SEPARATOR = " │ "  # │ -- la forma exacta del mockup de la spec S8

_NAV_BAR_HEIGHT_PX = 36

#: Nombres de pagina validos para ``show_view``/``current_view_name`` (Task
#: 19: navegacion entre la operacion en vivo y las vistas de calibracion/
#: validacion, spec S8 -- pantallas SEPARADAS, nunca compitiendo por
#: espacio con el video). El orden de esta lista es el orden de los
#: botones de la barra de navegacion.
_VIEW_NAMES = ["live", "intrinsic_calibration", "plane_calibration", "validation"]
_VIEW_LABELS = {
    "live": "Live",
    "intrinsic_calibration": "Calibrate camera",
    "plane_calibration": "Calibrate plane",
    "validation": "Validation",
}


class _PipelineWorker(QObject):
    """Envuelve un ``Pipeline`` para correr ``process`` fuera del hilo de la GUI.

    Vive en su propio ``QThread`` (ver ``MainWindow.__init__``); no corre
    nada por si solo, solo reacciona a ``process_frame`` cuando
    ``MainWindow`` le entrega un cuadro.
    """

    result_ready = Signal(object)

    #: Emitida cuando ``Pipeline.process`` lanza una de las DOS excepciones
    #: que su propio docstring declara (ver ``process_frame``). Lleva el
    #: mensaje ya redactado y una bandera que dice si es un problema de
    #: resolucion/intrinsecos (``True``) o de la config del montaje
    #: (``False``) -- los dos se bloquean, pero solo el primero se puede
    #: deshacer cargando una calibracion nueva.
    pipeline_failed = Signal(str, bool)

    def __init__(self, pipeline: Pipeline) -> None:
        super().__init__()
        self._pipeline = pipeline

    def process_frame(self, frame: np.ndarray) -> None:
        """Slot: procesa un cuadro y emite su ``FrameResult``.

        Este metodo corre en el hilo del worker (por la afinidad que le dio
        ``moveToThread``), nunca en el de la GUI -- es la pieza central de
        la Ruling 4 del encargo.

        Captura las DOS excepciones que ``Pipeline.process`` DOCUMENTA que
        puede lanzar, y nada mas (ronda de correcciones finales, I5). Sin
        esto, un ``marker_layout`` colineal -- un error de tipeo en la
        config que pasa la validacion de arranque -- imprimia un traceback
        por CADA cuadro en la consola, con el video congelado y la barra de
        estado en verde: el contrato "nunca un traceback" del punto de
        entrada valia solo hasta que existia una ventana.

        NO es un ``except Exception``: un error de programacion aguas
        arriba tiene que verse. Este proyecto ya pago por un except ciego
        (Task 8, ``np.trapezoid`` devolviendo ``None`` para siempre en
        silencio), y ``pipeline.py`` deja propagar a proposito todo lo que
        no este en esta lista.
        """
        try:
            result = self._pipeline.process(frame)
        except IntrinsicsMismatch as exc:
            self.pipeline_failed.emit(str(exc), True)
            return
        except DegenerateMarkerLayout as exc:
            self.pipeline_failed.emit(
                "Measurement stopped: the configured marker_layout is degenerate "
                f"(collinear or coincident marker positions), so no homography can be "
                f"fitted to it. Fix marker_layout in the config file and restart. ({exc})",
                False,
            )
            return
        self.result_ready.emit(result)


class MainWindow(QMainWindow):
    """Ventana principal: barra de navegacion, y debajo la pagina activa de
    un ``QStackedWidget`` -- la operacion en vivo (barra de estado, video
    central, panel) mas las tres vistas de esta tarea (spec S8: las
    pantallas de calibracion y de validacion son SEPARADAS, no compiten
    por espacio con el video en vivo, que sigue siendo la unica pagina que
    tiene que ganarse pixel a pixel contra el panel de 240 px).

    ``pipeline`` es opcional a proposito: las pruebas de layout de esta
    tarea no necesitan medir nada, solo alimentar ``update_frame_result``
    con un ``FrameResult`` construido a mano. Cuando se entrega un
    ``pipeline`` real, ``submit_frame``/``connect_camera`` quedan listos
    para que la Task 20 (el punto de entrada) conecte una camara real sin
    tener que resolver otra vez el hilo de procesamiento.

    ``cfg``/``validation_log_path`` son OPCIONALES: ``MainWindow()`` sin
    argumentos (como construyen todas las pruebas de layout de la Task 18)
    sigue funcionando sin tocar disco ni exigir un montaje real -- sin
    ``cfg`` se usa un ``AppConfig`` placeholder (ningun test de esta tarea
    detecta marcadores a traves de el sin pasar un ``cfg`` explicito), y
    sin ``validation_log_path`` el log de validacion queda en un archivo
    temporal con un nombre unico por instancia (nunca se CREA hasta el
    primer ``record_sample`` real, asi que una ventana construida sin
    validar nada no deja ningun rastro en disco). Task 20 pasara los dos
    reales.
    """

    _frame_submitted = Signal(np.ndarray)

    def __init__(
        self,
        intrinsics: Intrinsics | None = None,
        pipeline: Pipeline | None = None,
        cfg: AppConfig | None = None,
        validation_log_path: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("codex-caliper")

        self._intrinsics = intrinsics
        self._pipeline = pipeline
        self._plane = None
        self._camera_error: str | None = None
        self._resolution_blocked_reason: str | None = None
        self._config_error: str | None = None
        self._resolution_checked = False
        self._camera_thread: CameraThread | None = None
        self._worker_thread: QThread | None = None
        self._worker: _PipelineWorker | None = None
        self._status_text = ""
        self._status_warning = False
        self._notice_text = ""
        self._cfg = cfg if cfg is not None else AppConfig(marker_thickness_mm=0.0)

        # -- Contrapresion de un solo cuadro en vuelo (C1) -------------------
        #
        # ``_pipeline_busy`` es ``True`` desde que un cuadro se entrega al
        # worker hasta que su resultado (o su fallo) vuelve. Mientras lo
        # este, un cuadro nuevo NO se encola: SOBRESCRIBE
        # ``_pending_frame``. Ver ``submit_frame`` para por que.
        self._pipeline_busy = False
        self._pending_frame: np.ndarray | None = None

        #: Cuantos cuadros se descartaron por llegar con el pipeline ocupado.
        #: Es el unico testigo observable de que la contrapresion existe --
        #: una implementacion sin ella deja este contador en cero y entrega
        #: TODOS los cuadros al worker.
        self.dropped_frame_count = 0

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
            f"background-color: {SURFACE_HEX}; border-bottom: 1px solid {RULE_HEX};"
        )
        status_layout = QHBoxLayout(status_bar_widget)
        status_layout.setContentsMargins(12, 0, 12, 0)
        status_layout.setSpacing(0)
        status_layout.addWidget(self._status_label)
        status_layout.addStretch(1)
        self._status_bar_widget = status_bar_widget

        # -- Linea de avisos de operacion (C2) --------------------------------
        #
        # ``FrameResult.warnings`` no tenia NINGUN consumidor: el pipeline
        # escribe ahi cinco mensajes distintos para el operador -- entre
        # ellos "Separate them on the tray." -- y ninguno podia verse. La
        # spec S7 etapa 4 no lo deja opcional: con dos mascaras fusionadas
        # el sistema tiene que negarse A MEDIR *y mostrar el aviso*, porque
        # "una medida falsa es peor que ninguna". Sin este canal, el caso
        # normal de dos bananos pegados se veia como "No fruit detected" en
        # el panel y una barra de estado perfectamente sana.
        #
        # Por que una linea PROPIA y no la barra de estado: esa barra es un
        # readout de instrumento de tres campos de ancho fijo cuya forma la
        # fija el mockup de la spec S8 (y la fijan sus pruebas). Meter ahi
        # un mensaje de longitud variable -- "Fruits 3 and 7 are closer
        # than 2.0 mm, so they cannot be measured separately. Separate them
        # on the tray." -- empuja los tres campos y los recorta. Esta linea
        # va pegada DEBAJO del video, que es donde la mirada del operador
        # ya esta (spec S8: el video es lo que vende), crece para varios
        # avisos a la vez, y desaparece cuando no hay ninguno en vez de
        # dejar un hueco permanente.
        self._notice_label = QLabel()
        self._notice_label.setFont(QFont("Segoe UI", 9))
        self._notice_label.setWordWrap(False)
        self._notice_label.setStyleSheet(f"color: {SIGNAL_WARN_HEX};")
        notice_bar = QWidget()
        notice_bar.setFixedHeight(_NOTICE_BAR_HEIGHT_PX)
        notice_bar.setStyleSheet(
            f"background-color: {SURFACE_HEX}; border-top: 1px solid {RULE_HEX};"
        )
        notice_layout = QHBoxLayout(notice_bar)
        notice_layout.setContentsMargins(12, 0, 12, 0)
        notice_layout.setSpacing(0)
        notice_layout.addWidget(self._notice_label)
        notice_layout.addStretch(1)
        self._notice_bar = notice_bar
        notice_bar.setVisible(False)

        outer = QWidget()
        outer.setStyleSheet(f"background-color: {BG_VOID_HEX};")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        outer_layout.addWidget(status_bar_widget)
        outer_layout.addWidget(content, 1)
        outer_layout.addWidget(notice_bar)

        # -- Navegacion entre vistas (Task 19, spec S8) -----------------------
        #
        # ``outer`` (la pagina de operacion en vivo de arriba, sin cambios)
        # pasa a ser la pagina "live" de un ``QStackedWidget``; las otras
        # tres paginas son las vistas de calibracion/validacion de esta
        # tarea. Solo una pagina es visible a la vez -- eso es, literalmente,
        # "pantalla separada, no compite por espacio" (spec S8): no un panel
        # lateral mas, una vista entera distinta.
        self.intrinsic_calibration_view = IntrinsicCalibrationView()
        self.plane_calibration_view = PlaneCalibrationView(self._cfg)
        if validation_log_path is None:
            # Unico por instancia y NUNCA creado hasta el primer
            # ``record_sample`` real (``ValidationLog.__init__`` solo LEE si
            # el archivo ya existe) -- una ``MainWindow`` construida sin un
            # log real (todas las pruebas de layout de la Task 18, y la
            # mayoria de las de esta tarea) no deja ningun rastro en disco.
            validation_log_path = Path(tempfile.gettempdir()) / f"codex_caliper_validation_{uuid.uuid4().hex}.csv"
        self.validation_view = ValidationView(ValidationLog(validation_log_path))

        # La vista de calibracion intrinseca desbloquea la medicion a
        # traves de la MISMA compuerta que ya construyo la Task 18
        # (``load_intrinsics``/``measurement_enabled``) -- Ruling 3 del
        # encargo: esta ventana no reimplementa esa logica, solo la conecta.
        self.intrinsic_calibration_view.calibration_accepted.connect(self.load_intrinsics)

        self._stack = QStackedWidget()
        self._stack.addWidget(outer)
        self._stack.addWidget(self.intrinsic_calibration_view)
        self._stack.addWidget(self.plane_calibration_view)
        self._stack.addWidget(self.validation_view)

        self._nav_buttons: dict[str, QPushButton] = {}
        nav_bar = QWidget()
        nav_bar.setFixedHeight(_NAV_BAR_HEIGHT_PX)
        nav_bar.setStyleSheet(f"background-color: {SURFACE_HEX}; border-bottom: 1px solid {RULE_HEX};")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(8, 0, 8, 0)
        nav_layout.setSpacing(4)
        for name in _VIEW_NAMES:
            button = QPushButton(_VIEW_LABELS[name])
            button.setCheckable(True)
            button.setFlat(True)
            button.clicked.connect(lambda _checked=False, n=name: self.show_view(n))
            self._nav_buttons[name] = button
            nav_layout.addWidget(button)
        nav_layout.addStretch(1)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(nav_bar)
        central_layout.addWidget(self._stack, 1)
        self.setCentralWidget(central)

        self.show_view("live")

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
            self._worker.pipeline_failed.connect(self.on_pipeline_failed)

        self._refresh_status()

    # -- Navegacion entre vistas (Task 19, spec S8) --------------------------

    def show_view(self, name: str) -> None:
        """Muestra la pagina ``name`` (``"live"``, ``"intrinsic_calibration"``,
        ``"plane_calibration"`` o ``"validation"``) y nada mas: cambiar de
        pagina no pausa ni reinicia la camara ni el pipeline -- siguen
        corriendo si estan conectados, exactamente como un operador
        esperaria de pestanas, no de pantallas modales."""
        if name not in _VIEW_NAMES:
            raise ValueError(f"unknown view {name!r}; expected one of {_VIEW_NAMES}")
        self._stack.setCurrentIndex(_VIEW_NAMES.index(name))
        for button_name, button in self._nav_buttons.items():
            button.setChecked(button_name == name)

    def current_view_name(self) -> str:
        """El nombre de la pagina actualmente visible."""
        return _VIEW_NAMES[self._stack.currentIndex()]

    # -- Entrada de cuadros ------------------------------------------------

    def submit_frame(self, frame: np.ndarray) -> None:
        """Entrega un cuadro para procesar, sin bloquear el hilo de la GUI.

        Solo emite una senal; quien procesa de verdad es
        ``_PipelineWorker.process_frame``, en su propio hilo (Ruling 4).
        Sin un ``pipeline`` configurado no hay nada conectado del otro
        lado -- la senal se emite igual, sin efecto, en vez de lanzar.

        Contrapresion: UN SOLO cuadro en vuelo (C1)
        --------------------------------------------
        La conexion al worker es encolada por Qt (afinidad de hilo
        distinta) y esa cola NO tiene tope. La version anterior emitia por
        cada cuadro de camara: a 30 cuadros/s de entrada contra 3-10
        cuadros/s de rendimiento real del pipeline en CPU, se acumulaban
        ~25 cuadros por segundo de ~6.2 MB cada uno (~155 MB/s) y la
        memoria del proceso se agotaba en 30-60 segundos. No es una
        degradacion: es una caida, y ocurre dentro del primer minuto de la
        unica demostracion que el proyecto publica (spec S13, punto 1).

        Mientras el worker esta ocupado, un cuadro nuevo SOBRESCRIBE al
        pendiente en vez de encolarse. Eso es lo correcto y no solo lo
        barato: para medir fruta quieta sobre una bandeja, el cuadro util
        es el MAS RECIENTE -- procesar una cola atrasada entrega medidas
        de hace varios segundos sobre una escena que ya cambio.

        La bandera solo se levanta si hay un worker conectado: sin
        pipeline nadie emitiria ``result_ready`` para bajarla otra vez, y
        la primera llamada dejaria la ventana sorda para siempre.
        """
        if self._worker is None:
            self._frame_submitted.emit(frame)
            return
        if self._pipeline_busy:
            if self._pending_frame is not None:
                self.dropped_frame_count += 1
            self._pending_frame = frame
            return
        self._pipeline_busy = True
        self._pending_frame = None
        self._frame_submitted.emit(frame)

    def _release_pipeline(self) -> None:
        """Baja la bandera de ocupado y entrega el cuadro pendiente, si hay.

        Lo llaman los DOS caminos de retorno del worker
        (``update_frame_result`` y ``on_pipeline_failed``): si solo lo
        hiciera el camino feliz, una excepcion del pipeline dejaria la
        bandera arriba y la ventana no volveria a entregar ni un cuadro.
        """
        self._pipeline_busy = False
        pending = self._pending_frame
        self._pending_frame = None
        if pending is not None and self.measurement_enabled:
            self.submit_frame(pending)

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
        vez, PINTA el cuadro crudo y despues lo entrega a ``submit_frame``
        mientras la medicion no este bloqueada (Ruling 3).

        El video se refresca desde AQUI, en cada cuadro de camara (C1)
        ---------------------------------------------------------------
        La version anterior solo tocaba ``video_view`` en este slot cuando
        la medicion estaba BLOQUEADA; con la medicion activa, la unica via
        de pintura era ``update_frame_result`` -- o sea que la cadencia del
        video ERA la cadencia del pipeline (3-10/s en CPU), no la de la
        camara. ``set_raw_frame`` es la capa que puede ir al ritmo de la
        camara sin pagar nada del pipeline: el PiP crudo se mueve siempre,
        y cuando no hay lienzo rectificado vigente (calibracion BLOCKED, o
        medicion bloqueada) el area principal tambien.

        Lo que esta linea NO arregla, y hay que decirlo: el LIENZO
        rectificado sigue llegando a la cadencia del pipeline. Rectificarlo
        aqui exigiria ademas corregir distorsion en el hilo de la GUI (18
        ms por cuadro a 1080p con ``cv2.undistort``, medido en esta
        maquina), que es justamente lo que la Ruling 4 saco de este hilo.
        El README se corrigio para decir lo que de verdad se mueve a la
        cadencia de la camara, en vez de afirmar lo contrario.
        """
        if not self._resolution_checked:
            self._resolution_checked = True
            if self._intrinsics is not None and self._camera_thread is not None:
                try:
                    self._intrinsics.require_compatible(self._camera_thread.actual_size)
                except IntrinsicsMismatch as exc:
                    self._resolution_blocked_reason = str(exc)
                    self._refresh_status()

        self.video_view.set_raw_frame(frame)

        if not self.measurement_enabled:
            # Finding 2 (ronda de revision 1): la camara sigue entregando
            # cuadros aunque la medicion este bloqueada (sin intrinsecos
            # todavia, o resolucion no compatible) -- eso no puede congelar
            # el video. Descartar el lienzo reusa el mismo camino de
            # respaldo que la Ruling 2 le dio a ``VideoView`` para el
            # cuadro BLOCKED del pipeline: mostrar el crudo en vez de nada,
            # y mostrarlo MOVIENDOSE.
            self.video_view.set_rectified_frame(None)
            self._set_notices([self.blocked_reason()])
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
        decorativo. Se usa ``result.plane.mm_per_px``, la escala REAL de
        esta calibracion.

        Ya no hay un default de 0.2 mm/px para el caso sin plano (ronda de
        correcciones finales): era codigo muerto demostrable -- el pipeline
        devuelve ``results`` VACIO en todo cuadro sin plano, asi que no
        habia nada que formatear con el -- y un default plausible en el
        camino de la escala es una invitacion a medir con una escala que
        nadie calibro.

        ``draw_marker_overlay`` (Task 17) se aplica tambien AQUI, sobre el
        MISMO lienzo rectificado -- encargo posterior al plan de 20 tareas.
        Antes solo se dibujaba en ``PlaneCalibrationView`` (una pantalla
        aparte, de solo calibracion, que la spec documenta como
        display-only), asi que la vista principal -- la que spec S13 pone
        PRIMERO entre lo que se publica -- nunca mostraba los marcadores que
        sostienen la metrica. Se dibuja sobre el lienzo RECTIFICADO, no
        sobre el PiP crudo, por una razon de CADENCIA, no de gusto:
        ``result.markers`` llega a la cadencia del PIPELINE (esta misma
        senal), pero el PiP se refresca aparte, a la cadencia de la CAMARA
        (``set_raw_frame``, ver docstring de ``video_view.py``) -- dibujar
        marcadores ahi los dejaria desincronizados del cuadro que
        realmente los contiene, parpadeando entre un cuadro con marcadores
        pintados y varios sin ellos hasta el siguiente resultado del
        pipeline. El lienzo rectificado, en cambio, SOLO se actualiza
        junto con ``result`` (misma llamada), asi que los marcadores que se
        pintan son siempre los de ESE cuadro. Se dibuja ANTES de
        ``draw_results``: los marcadores viven en las esquinas del lienzo y
        la fruta en el centro, pero si algo cambia esa geometria el
        protagonista de este video es la fruta, no el marcador.

        ``result.warnings`` se enruta a la linea de avisos (C2). Era el
        unico campo de ``FrameResult`` sin ningun consumidor en todo el
        proyecto, con cinco mensajes distintos de operador escritos ahi.
        """
        self._plane = result.plane
        mm_per_px = result.plane.mm_per_px if result.plane is not None else None

        rectified = result.rectified
        if rectified is not None:
            rectified = draw_marker_overlay(rectified, result.markers)
            if mm_per_px is not None:
                rectified = draw_results(rectified, result.results, mm_per_px)

        self.video_view.set_rectified_frame(rectified)
        self.video_view.set_raw_frame(result.raw_undistorted)
        self.side_panel.update_results(result.results, mm_per_px)
        self._set_notices(result.warnings)
        self._refresh_status()
        self._release_pipeline()

    def on_pipeline_failed(self, message: str, is_resolution_problem: bool) -> None:
        """Slot de ``_PipelineWorker.pipeline_failed`` (I5).

        Las dos excepciones que ``Pipeline.process`` documenta son errores
        de CONFIGURACION que no se arreglan en el cuadro siguiente, asi que
        no basta con mostrar el mensaje una vez: la medicion se BLOQUEA,
        igual que ya hacia la guardia de resolucion. Sin ese bloqueo, el
        cuadro pendiente se volveria a entregar, volveria a fallar, y la
        ventana quedaria girando sobre el mismo error un cuadro a la vez.

        El problema de resolucion se guarda en ``_resolution_blocked_reason``
        -- el mismo campo que la guardia de ``actual_size``, misma causa y
        mismo mensaje -- para que ``load_intrinsics`` lo limpie igual que
        limpia el otro. El layout degenerado va a ``_config_error``, que
        NADIE limpia: ninguna calibracion nueva arregla una config
        colineal.
        """
        if is_resolution_problem:
            self._resolution_blocked_reason = message
        else:
            self._config_error = message
        self._set_notices([message])
        self._refresh_status()
        self._release_pipeline()

    # -- Avisos de operacion (C2) ---------------------------------------------

    def notice_text(self) -> str:
        """Los avisos de operacion que la interfaz esta mostrando ahora
        mismo, en ingles, exactamente como los lee el operador. Cadena
        vacia si no hay ninguno."""
        return self._notice_text

    def _set_notices(self, messages: list[str]) -> None:
        texts = [message for message in messages if message]
        self._notice_text = _NOTICE_SEPARATOR.join(texts)
        self._notice_label.setText(self._notice_text)
        # Sin avisos la linea se esconde entera, en vez de dejar una franja
        # vacia permanente robandole alto al video (spec S8).
        self._notice_bar.setVisible(bool(texts))

    # -- Calibracion / bloqueo de medicion -----------------------------------

    def load_intrinsics(self, intrinsics: Intrinsics | None) -> None:
        """Reemplaza los intrinsecos activos Y los del pipeline.

        Vuelve a verificar la resolucion del cuadro siguiente en vez de
        confiar en una verificacion vieja.

        Propagar al pipeline no es opcional (I4)
        ----------------------------------------
        ``Pipeline`` guarda su PROPIA referencia a los intrinsecos y la usa
        para ``undistort`` en cada cuadro. La version anterior actualizaba
        solo la de la ventana, asi que la barra de estado habria mostrado
        el RMS de la calibracion NUEVA mientras cada medida se seguia
        corrigiendo con la VIEJA -- en verde, sin ningun aviso. Es la
        corrupcion silenciosa que ``require_compatible`` existe para
        evitar, por un camino que esa guardia no cubre: dos calibraciones
        de la MISMA resolucion y distinta distorsion pasan la guardia sin
        objetar nada.

        Hoy es inalcanzable (nadie llama a ``set_live_frame``, ver "Known
        limits" del README), y por eso mismo se arregla ahora: el README
        apunta al lector hacia "cablear la pantalla de calibracion" como
        la tarea pendiente, y hacerlo sin esta linea no deja una pantalla
        incompleta -- deja un sistema que mide mal en silencio.

        El cambio es un reemplazo de referencia desde el hilo de la GUI
        mientras el worker puede estar leyendola: atomico bajo el GIL, y
        el peor caso es que el cuadro en vuelo (hay a lo sumo UNO, ver
        ``submit_frame``) se corrija con los intrinsecos anteriores. Si la
        resolucion no corresponde, ``undistort`` lanza
        ``IntrinsicsMismatch`` y ``on_pipeline_failed`` lo bloquea -- el
        mismo desenlace que la guardia de ``actual_size``, no uno nuevo.
        """
        self._intrinsics = intrinsics
        if self._pipeline is not None and intrinsics is not None:
            self._pipeline.intrinsics = intrinsics
        self._resolution_checked = False
        self._resolution_blocked_reason = None
        self._refresh_status()

    @property
    def measurement_enabled(self) -> bool:
        """``False`` mientras falte una calibracion valida para la resolucion
        real de la camara, mientras la camara este perdida (Ruling 3), o
        mientras el pipeline haya fallado por un error de configuracion
        que no se arregla solo (I5)."""
        return not (
            self._camera_error
            or self._config_error
            or self._resolution_blocked_reason
            or self._intrinsics is None
        )

    def blocked_reason(self) -> str:
        """Por que ``measurement_enabled`` es ``False`` en este momento, en
        ingles, lista para mostrarse tal cual. Cadena vacia si no esta
        bloqueado."""
        if self._camera_error:
            return f"Measurement is blocked: {self._camera_error}"
        if self._config_error:
            return self._config_error
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
        color = SIGNAL_WARN_HEX if warning else INK_PRIMARY_HEX
        self._status_label.setStyleSheet(f"color: {color};")

    def _compute_status(self) -> tuple[str, bool]:
        if self._camera_error:
            return f"{_CROSS} Camera error: {self._camera_error}", True
        if self._config_error:
            return f"{_CROSS} {self._config_error}", True
        if self._resolution_blocked_reason:
            return f"{_CROSS} {self._resolution_blocked_reason}", True

        intrinsics_text, intrinsics_warning = self._intrinsics_segment()
        markers_text, markers_warning = self._markers_segment()
        residual_text, residual_warning = self._residual_segment()

        text = intrinsics_text + _SEPARATOR + markers_text + _SEPARATOR + residual_text
        return text, intrinsics_warning or markers_warning or residual_warning

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

    def _residual_segment(self) -> tuple[str, bool]:
        """Tercer campo de la barra de estado: el residual de reproyeccion
        de la homografia de ESTE cuadro, en mm (I8).

        Antes el campo mostraba ``plane.mm_per_px``, que
        ``compute_plane_calibration`` copia literalmente de
        ``cfg.canvas_mm_per_px``: imprimia "0.200 mm/px" para siempre, no
        podia indicar salud, y el README lo llamaba "la escala en vivo".
        Una constante disfrazada de lectura en vivo es, en un instrumento
        de medicion, peor que no mostrar nada -- da la impresion de un
        chequeo que no existe.

        ``residual_mm`` SI se mide en cada cuadro (el RMS del error de
        reproyeccion de las 16 o 12 esquinas, Task 5), es el numero que la
        spec S8 pide en esa barra, y es el que de verdad se puede ir a
        mal: sube si alguien mueve un marcador, si el layout de la config
        no corresponde al montaje, o si la deteccion de esquinas se
        degrada. El mockup de la spec pone un valor MEDIDO en ese campo
        ("0.183 mm/px"), no el parametro de configuracion.

        Advertencia que hereda de Task 5 y que el README ya documenta: es
        un RMS global, asi que atenua el error de UN marcador suelto
        (~0.065 mm de residual por mm de desplazamiento). Un residual bajo
        no descarta un marcador mal ubicado; uno alto si descarta que todo
        este bien.
        """
        if self._plane is None:
            return "— mm residual", False
        residual_mm = self._plane.residual_mm
        warning = residual_mm >= RESIDUAL_WARN_MM
        mark = _WARN if warning else _CHECK
        return f"{mark} {residual_mm:.3f} mm residual", warning

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

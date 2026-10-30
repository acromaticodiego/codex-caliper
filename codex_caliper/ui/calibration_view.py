"""Vistas de calibracion intrinseca y de plano (Task 19).

Pantallas SEPARADAS de la operacion en vivo (spec S8): no compiten por
espacio con el video de medicion, que es lo que ``main_window.py`` ya
dedica por completo al fruto. ``MainWindow`` las aloja en un
``QStackedWidget`` y navega hacia ellas, no al reves.

``IntrinsicCalibrationView`` -- el toggle antes/despues no es un extra
---------------------------------------------------------------------------
Spec S7 etapa 1 llama al toggle antes/despues de la correccion de
distorsion "el recurso visual mas eficaz del proyecto" y pide que exista
aunque solo se use en el video de presentacion (S13 lo lista SEGUNDO,
justo despues del video en vivo). Por eso ``undistort_preview`` no es una
bandera decorativa: su setter vuelve a calcular ``displayed_frame`` de
verdad, aplicando ``codex_caliper.intrinsics.undistort`` sobre el cuadro
crudo cuando hay una calibracion ACEPTADA cargada. ``displayed_frame_is_undistorted``
inspecciona el cuadro MOSTRADO contra el crudo -- nunca devuelve la bandera
de vuelta -- precisamente para que una implementacion que fije la bandera
sin corregir nada (el toggle mas vacio que se puede escribir) no pueda
pasar por una vista que de verdad corrige distorsion. Ver el informe de la
tarea para la ruptura deliberada que demuestra esto.

``run_calibration`` rechaza por encima de ``INTRINSICS_MAX_RMS_PX`` (0.5 px,
restriccion global del plan, Task 3): una calibracion rechazada NUNCA queda
cargada en ``self._intrinsics`` (la que esta vista expone para alimentar
``MainWindow.load_intrinsics``) -- solo se recuerda el intento
(``_last_attempt``) para poder reportar su RMS real en
``rejection_message()``.

``PlaneCalibrationView`` reusa ``draw_marker_overlay`` (Task 17) para
mostrar las esquinas/ids detectados con el mismo color que ya codifica el
estado de calibracion (verde/naranja/rojo) en el resto de la interfaz --
nunca una segunda convencion de color para la misma informacion.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from codex_caliper.aruco_tools import CalibrationState, MarkerDetection, detect_markers
from codex_caliper.config import AppConfig
from codex_caliper.homography import PlaneCalibration, compute_plane_calibration
from codex_caliper.intrinsics import (
    INTRINSICS_MAX_RMS_PX,
    Intrinsics,
    calibrate_from_corners,
    find_chessboard,
    undistort,
)
from codex_caliper.ui.imaging import bgr_to_qpixmap
from codex_caliper.ui.overlay import draw_marker_overlay
from codex_caliper.ui.theme import (
    BG_VOID_HEX,
    INK_MUTED_HEX,
    INK_PRIMARY_HEX,
    SIGNAL_FAIL_HEX,
    SIGNAL_PASS_HEX,
    SIGNAL_WARN_HEX,
    SURFACE_HEX,
)

#: Esquinas internas por defecto del tablero de calibracion (columnas, filas).
_DEFAULT_PATTERN_SIZE = (9, 6)

#: Lado del cuadro del tablero por defecto, en mm.
_DEFAULT_SQUARE_MM = 25.0

#: Resolucion por defecto de la camara del rig (Task 16).
_DEFAULT_IMAGE_SIZE = (1920, 1080)

#: Ancho maximo de la vista previa en pixeles de widget -- estas vistas no
#: necesitan pantalla completa como el video en vivo (spec S8 solo lo exige
#: para la operacion, no para la calibracion).
_PREVIEW_MAX_WIDTH_PX = 640


def _preview_pixmap(frame: np.ndarray) -> QPixmap:
    """``frame`` como ``QPixmap``, limitado a ``_PREVIEW_MAX_WIDTH_PX`` de ancho.

    Las dos vistas de este modulo tenian este cuerpo duplicado LITERALMENTE
    (``_update_preview_pixmap``, una en cada clase, en el mismo archivo).
    Una sola funcion aqui es lo que hace que las dos vistas previas no
    puedan divergir en el ancho maximo ni en el modo de reescalado.
    """
    pixmap = bgr_to_qpixmap(frame)
    if pixmap.width() > _PREVIEW_MAX_WIDTH_PX:
        pixmap = pixmap.scaledToWidth(_PREVIEW_MAX_WIDTH_PX, Qt.TransformationMode.SmoothTransformation)
    return pixmap


class IntrinsicCalibrationView(QWidget):
    """Calibracion intrinseca interactiva: acumula poses del tablero,
    calibra y expone el toggle antes/despues de la correccion de
    distorsion.

    No conecta una camara por si misma (Task 20, el punto de entrada, le
    entrega cuadros reales con ``set_live_frame``/``capture_pose``); esta
    vista no necesita saber de donde vienen.
    """

    #: Emitida cuando ``run_calibration`` ACEPTA una calibracion nueva, con
    #: el ``Intrinsics`` resultante. ``MainWindow`` la conecta a
    #: ``load_intrinsics`` (Task 18) para desbloquear la medicion sin que
    #: esta vista tenga que conocer esa compuerta (Ruling 3 del encargo).
    calibration_accepted = Signal(object)

    def __init__(
        self,
        pattern_size: tuple[int, int] = _DEFAULT_PATTERN_SIZE,
        square_mm: float = _DEFAULT_SQUARE_MM,
        image_size: tuple[int, int] = _DEFAULT_IMAGE_SIZE,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.pattern_size = pattern_size
        self.square_mm = square_mm
        self.image_size = image_size

        self._object_points: list[np.ndarray] = []
        self._image_points: list[np.ndarray] = []
        self.pose_count = 0

        self._intrinsics: Intrinsics | None = None
        self._last_attempt: Intrinsics | None = None
        self._result_accepted: bool | None = None

        self._raw_frame: np.ndarray | None = None
        self._undistort_preview = False
        self.displayed_frame: np.ndarray | None = None

        self.setStyleSheet(f"background-color: {BG_VOID_HEX};")

        self._preview_label = QLabel("No frame yet")
        self._preview_label.setStyleSheet(f"color: {INK_MUTED_HEX}; background-color: {SURFACE_HEX};")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(320, 240)

        self._pose_label = QLabel()
        self._pose_label.setFont(QFont("Consolas", 10))
        self._pose_label.setStyleSheet(f"color: {INK_PRIMARY_HEX};")

        self._status_label = QLabel()
        self._status_label.setFont(QFont("Consolas", 10))
        self._status_label.setWordWrap(True)

        self._toggle_checkbox = QCheckBox("Show undistorted (before/after)")
        self._toggle_checkbox.setStyleSheet(f"color: {INK_PRIMARY_HEX};")
        self._toggle_checkbox.toggled.connect(self._on_toggle_checked)

        self._calibrate_button = QPushButton("Run calibration")
        self._calibrate_button.clicked.connect(self._on_calibrate_clicked)

        controls = QHBoxLayout()
        controls.addWidget(self._toggle_checkbox)
        controls.addStretch(1)
        controls.addWidget(self._calibrate_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self._preview_label, 1)
        layout.addWidget(self._pose_label)
        layout.addLayout(controls)
        layout.addWidget(self._status_label)

        self._update_pose_label()
        self._refresh_status_label()

    # -- Acumulacion de poses -----------------------------------------------

    def _board_object_points(self) -> np.ndarray:
        cols, rows = self.pattern_size
        objp = np.zeros((cols * rows, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_mm
        return objp

    def capture_pose(self, frame: np.ndarray) -> bool:
        """Busca el tablero en ``frame`` y, si lo encuentra, acumula la pose.

        Devuelve ``True`` solo si se acumulo una pose nueva. "accepted" en
        el nombre del test original de esta tarea significa exactamente
        esto -- una pose SIN tablero (``find_chessboard`` devuelve ``None``,
        un resultado normal durante la captura interactiva) no cuenta, y
        ``pose_count`` no debe subir por ella.
        """
        self.set_live_frame(frame)
        corners = find_chessboard(frame, self.pattern_size)
        if corners is None:
            return False
        self._object_points.append(self._board_object_points())
        self._image_points.append(np.asarray(corners, dtype=np.float32).reshape(-1, 2))
        self.pose_count += 1
        self._update_pose_label()
        return True

    # -- Calibracion ---------------------------------------------------------

    def run_calibration(self) -> bool:
        """Calibra con todas las poses acumuladas y rechaza por encima del
        umbral de aceptacion (``INTRINSICS_MAX_RMS_PX``, 0.5 px, Task 3).

        Una calibracion rechazada NUNCA queda en ``self.intrinsics``
        (``self._intrinsics`` solo guarda intentos ACEPTADOS): emitir
        ``calibration_accepted`` con un resultado por encima del umbral
        desbloquearia la medicion con una calibracion que el propio sistema
        ya juzgo insuficiente.
        """
        if not self._object_points:
            raise ValueError("run_calibration needs at least one captured pose (none accepted yet)")

        attempt = calibrate_from_corners(self._object_points, self._image_points, self.image_size)
        self._last_attempt = attempt
        self._result_accepted = attempt.is_acceptable

        if attempt.is_acceptable:
            self._intrinsics = attempt
            self._refresh_displayed_frame()
            self.calibration_accepted.emit(attempt)

        self._refresh_status_label()
        return self._result_accepted

    def _on_calibrate_clicked(self) -> None:
        """Slot del boton "Run calibration": captura el ``ValueError`` que
        ``run_calibration`` lanza sin ninguna pose capturada (fix round 1,
        Finding 1), y lo enruta a la etiqueta de estado en vez de dejarlo
        llegar a stderr como un traceback crudo.

        El nivel de API no cambia -- ``run_calibration()`` llamado
        directo (como hace ``tests/test_calibration_view.py::test_run_calibration_requires_at_least_one_captured_pose``)
        sigue lanzando, y eso esta bien: un llamador programatico (Task 20,
        u otra prueba) puede querer decidir que hacer con ese error. Esta
        captura vive SOLO en el limite de la interfaz, donde hacer clic sin
        haber capturado nada todavia no es un camino exotico -- es
        literalmente lo que un operador que recien abre esta pantalla va a
        probar primero. Spec S4 nombra "parecer un script" como
        anti-objetivo explicito, y un traceback en consola es precisamente
        esa senal; la guia de ``frontend-design`` tambien es directa sobre
        esto: un fallo es un momento para dar direccion, no para callar o
        para explotar.

        Se elige capturar en el slot del click, no deshabilitar el boton
        mientras ``pose_count == 0``: deshabilitar exige mantener ese
        estado sincronizado en cada punto que cambia ``pose_count``
        (``capture_pose``, y cualquier reinicio futuro), mientras que
        capturar en el unico limite por el que entra cualquier click deja
        un solo lugar que puede quedar desincronizado -- ninguno.
        """
        try:
            self.run_calibration()
        except ValueError as exc:
            self._status_label.setText(f"Cannot calibrate yet: {exc}")
            self._status_label.setStyleSheet(f"color: {SIGNAL_WARN_HEX};")

    @property
    def result_accepted(self) -> bool | None:
        """``True``/``False`` tras la ultima ``run_calibration``, ``None``
        si todavia no se corrio ninguna."""
        return self._result_accepted

    @property
    def intrinsics(self) -> Intrinsics | None:
        """Los intrinsecos ACEPTADOS (listos para
        ``MainWindow.load_intrinsics``), o ``None`` si ninguna calibracion
        paso el umbral todavia."""
        return self._intrinsics

    def rejection_message(self) -> str:
        """Por que la ultima calibracion se rechazo, con el RMS real que la
        produjo y el umbral real del proyecto -- cadena vacia si la ultima
        calibracion fue aceptada o si todavia no se corrio ninguna.

        No es una subcadena que cualquier mensaje con un "0.5" de casualidad
        satisfaga: el umbral se interpola desde ``INTRINSICS_MAX_RMS_PX``
        (la unica fuente de verdad, Task 3) y el RMS desde el intento real.
        """
        if self._result_accepted is not False or self._last_attempt is None:
            return ""
        return (
            f"Calibration rejected: RMS {self._last_attempt.rms_px:.2f} px exceeds "
            f"the {INTRINSICS_MAX_RMS_PX:.2f} px acceptance threshold. Capture more "
            "poses, or poses with more varied angles, and recalibrate."
        )

    # -- Toggle antes/despues -------------------------------------------------

    def set_live_frame(self, frame: np.ndarray) -> None:
        """Entrega un cuadro en vivo a la vista previa (Task 20 lo llamara
        desde la camara real; los tests le pasan cuadros sinteticos)."""
        self._raw_frame = frame
        self._refresh_displayed_frame()

    @property
    def undistort_preview(self) -> bool:
        """``True`` mientras la vista previa muestre el cuadro corregido."""
        return self._undistort_preview

    @undistort_preview.setter
    def undistort_preview(self, value: bool) -> None:
        self._undistort_preview = bool(value)
        self._toggle_checkbox.blockSignals(True)
        self._toggle_checkbox.setChecked(self._undistort_preview)
        self._toggle_checkbox.blockSignals(False)
        self._refresh_displayed_frame()

    def _on_toggle_checked(self, checked: bool) -> None:
        self.undistort_preview = checked

    def _refresh_displayed_frame(self) -> None:
        """Recalcula ``displayed_frame`` de verdad -- nunca solo la bandera.

        Esta es la unica funcion que decide que pixeles se muestran: aplica
        ``undistort`` sobre el cuadro CRUDO cuando el toggle esta activo Y
        hay una calibracion aceptada cargada; en cualquier otro caso
        muestra el crudo, sin fingir una correccion que no puede hacer
        todavia (sin intrinsecos no hay nada que corregir).
        """
        if self._raw_frame is None:
            self.displayed_frame = None
            self._preview_label.setText("No frame yet")
            return
        if self._undistort_preview and self._intrinsics is not None:
            self.displayed_frame = undistort(self._raw_frame, self._intrinsics)
        else:
            self.displayed_frame = self._raw_frame.copy()
        self._update_preview_pixmap()

    def displayed_frame_is_undistorted(self) -> bool:
        """Inspecciona el cuadro MOSTRADO contra el crudo -- nunca devuelve
        la bandera ``undistort_preview`` de vuelta (Ruling 2 del encargo:
        esa version se lee a si misma y pasa con una implementacion que no
        corrige nada en absoluto). Ver el informe de la tarea para la
        ruptura deliberada que demuestra la diferencia.
        """
        if self.displayed_frame is None or self._raw_frame is None:
            return False
        return not np.array_equal(self.displayed_frame, self._raw_frame)

    # -- Widgets ---------------------------------------------------------------

    def _update_preview_pixmap(self) -> None:
        if self.displayed_frame is None:
            return
        self._preview_label.setPixmap(_preview_pixmap(self.displayed_frame))

    def _update_pose_label(self) -> None:
        self._pose_label.setText(f"Poses captured: {self.pose_count}")

    def _refresh_status_label(self) -> None:
        if self._result_accepted is None:
            text, warning = "No calibration run yet.", False
        elif self._result_accepted:
            text = f"Calibration accepted: RMS {self._intrinsics.rms_px:.2f} px."
            warning = False
        else:
            text, warning = self.rejection_message(), True
        self._status_label.setText(text)
        color = SIGNAL_WARN_HEX if warning else INK_PRIMARY_HEX
        self._status_label.setStyleSheet(f"color: {color};")


class PlaneCalibrationView(QWidget):
    """Vista de calibracion de plano: detecta los marcadores ArUco del rig,
    ajusta la homografia y muestra el residual (mm) y la escala (mm/px) que
    esa calibracion sostiene -- un resultado publicable (spec S7 etapa 10),
    no un panel de depuracion.

    Pantalla separada de la operacion en vivo (spec S8): esta vista no
    rectifica ni dibuja cotas de fruta, solo lo que hace falta para que un
    operador confirme que los 4 (o 3) marcadores se ven y en que estado
    queda la calibracion metrica.
    """

    def __init__(self, cfg: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._detections: list[MarkerDetection] = []
        self._plane: PlaneCalibration | None = None
        self._has_frame = False

        self.setStyleSheet(f"background-color: {BG_VOID_HEX};")

        self._preview_label = QLabel("No frame yet")
        self._preview_label.setStyleSheet(f"color: {INK_MUTED_HEX}; background-color: {SURFACE_HEX};")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(320, 240)

        self._markers_label = QLabel()
        self._residual_label = QLabel()
        self._scale_label = QLabel()
        for label in (self._markers_label, self._residual_label, self._scale_label):
            label.setFont(QFont("Consolas", 10))
            label.setStyleSheet(f"color: {INK_PRIMARY_HEX};")

        info = QVBoxLayout()
        info.addWidget(self._markers_label)
        info.addWidget(self._residual_label)
        info.addWidget(self._scale_label)
        info.addStretch(1)

        layout = QHBoxLayout(self)
        layout.addWidget(self._preview_label, 1)
        layout.addLayout(info)

        self._refresh_labels()

    def update_frame(self, frame: np.ndarray) -> None:
        """Detecta marcadores en ``frame``, ajusta el plano si hay
        suficiente geometria, y refresca la vista previa con el mismo
        overlay de marcadores que ``draw_marker_overlay`` (Task 17) ya
        dibuja en la ventana en vivo -- una sola convencion de color para
        el estado de calibracion, no dos."""
        self._has_frame = True
        self._detections = detect_markers(frame, self._cfg)
        self._plane = compute_plane_calibration(self._detections, self._cfg)
        self.displayed_frame = draw_marker_overlay(frame, self._detections)
        self._update_preview_pixmap()
        self._refresh_labels()

    def marker_ids(self) -> list[int]:
        """Ids distintos detectados en el ultimo cuadro, ordenados."""
        return sorted(detection.id for detection in self._detections)

    @property
    def state(self) -> CalibrationState | None:
        """``None`` antes del primer cuadro; si no, el estado de la ultima
        calibracion de plano (``OK``/``DEGRADED``), o ``BLOCKED`` si
        ``compute_plane_calibration`` no pudo ajustar nada (incluido el
        caso de CERO marcadores detectados -- un cuadro ya procesado sin
        geometria sigue siendo ``BLOCKED``, no "todavia sin datos"; por eso
        se distingue con ``_has_frame``, no con si ``_detections`` esta
        vacia)."""
        if self._plane is not None:
            return self._plane.state
        if self._has_frame:
            return CalibrationState.BLOCKED
        return None

    def residual_text(self) -> str:
        """Residual de reproyeccion real, en mm -- cadena de relleno antes
        del primer cuadro medible, nunca un numero inventado."""
        if self._plane is None:
            return "— mm"
        return f"{self._plane.residual_mm:.3f} mm"

    def mm_per_px_text(self) -> str:
        if self._plane is None:
            return "— mm/px"
        return f"{self._plane.mm_per_px:.3f} mm/px"

    def _update_preview_pixmap(self) -> None:
        if self.displayed_frame is None:
            return
        self._preview_label.setPixmap(_preview_pixmap(self.displayed_frame))

    def _refresh_labels(self) -> None:
        ids = self.marker_ids()
        state = self.state
        if state is None:
            self._markers_label.setText("Markers: — (no frame yet)")
            color = INK_PRIMARY_HEX
        elif state is CalibrationState.OK:
            self._markers_label.setText(f"Markers: {ids} OK")
            color = SIGNAL_PASS_HEX
        elif state is CalibrationState.DEGRADED:
            self._markers_label.setText(f"Markers: {ids} DEGRADED (3/4)")
            color = SIGNAL_WARN_HEX
        else:
            self._markers_label.setText(f"Markers: {ids} BLOCKED")
            color = SIGNAL_FAIL_HEX
        self._markers_label.setStyleSheet(f"color: {color};")
        self._residual_label.setText(f"Residual: {self.residual_text()}")
        self._scale_label.setText(f"Scale: {self.mm_per_px_text()}")

"""Tests de la ventana principal, la vista de video y el panel lateral (Task 18).

El brief de esta tarea tenia cuatro pruebas de las seis que no discriminaban
entre una implementacion correcta y una incorrecta (rulings del encargo);
las correcciones se verifican aqui, no solo se mencionan:

1. **PiP de ida y vuelta.** El brief solo presionaba P una vez y
   comprobaba ``pip_visible is False``. Una implementacion que solo
   apagara el PiP (y nunca lo encendiera de nuevo) pasaria igual. Aqui se
   verifica el estado inicial Y el viaje de vuelta.

2. **Banda de dos lados para el PiP.** ``pip.width() < main.width() // 3``
   lo satisface un PiP de un solo pixel. Se agrega un piso absoluto: el PiP
   tiene que ser chico Y seguir siendo legible en un video real.

3. **Aviso bloqueado vs. sano.** El brief solo probaba el caso ``True``, lo
   que no distingue "avisa cuando corresponde" de "avisa siempre". Se
   prueban ambos lados.

4. **Ancho exacto, sin aflojar la aserticion.** En vez de relajar
   ``window.width() - 240`` a una tolerancia, los layouts de nivel
   superior quedan con margen y espaciado CERO -- eso es, literalmente,
   lo que pide la spec S8 (el video domina, maximizar solo crece el
   video). Si la igualdad exacta no se sostiene, se reporta el numero
   real en vez de ensanchar la prueba.

Mas alla del brief, tres Rulings del encargo exigen comportamiento nuevo
que ningun test original cubria: ``rectified is None`` no puede dejar la
ventana en negro (Ruling 2), ``capture_lost``/``actual_size`` tienen que
notarse en la interfaz (Ruling 3), y ``Pipeline.process`` nunca puede
correr en el hilo de la GUI (Ruling 4).

Ronda de revision 1 (tres hallazgos sobre el commit e8b4afc) agrega tres
pruebas mas, al final del archivo: ``draw_results`` (Task 17) tiene que
aplicarse sobre el lienzo que se muestra (Finding 1, critico -- sin esto
las cotas solo viven en el panel, el anti-objetivo exacto de la spec S8);
el video no puede congelarse mientras la medicion esta bloqueada (Finding
2); y el panel lateral no deja tarjetas viejas visibles tras una
actualizacion (encontrado revisando una CAPTURA DE PANTALLA, no un conteo
-- ver su docstring para por que la primera version de esa prueba no
discriminaba).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel

import threading

from codex_caliper.aruco_tools import CalibrationState, MarkerDetection
from codex_caliper.camera import CameraThread
from codex_caliper.codex import CodexVerdict
from codex_caliper.homography import DegenerateMarkerLayout, PlaneCalibration
from codex_caliper.intrinsics import Intrinsics, IntrinsicsMismatch
from codex_caliper.pipeline import MARKER_COUNT, BananaResult, FrameResult
from codex_caliper.aggregation import TEMPORAL_WINDOW, TrackAggregate
from codex_caliper.ui.main_window import RESIDUAL_WARN_MM, MainWindow
from codex_caliper.ui.overlay import text_anchor_for
from codex_caliper.ui.side_panel import SIDE_PANEL_WIDTH_PX, SidePanel
from codex_caliper.ui.video_view import (
    _PIP_MARGIN_PX,
    _pip_origin_px,
    PIP_MIN_WIDTH_PX,
    PIP_SCALE,
    VideoView,
)


# ---------------------------------------------------------------------------
# Helpers de datos sinteticos -- ningun test de esta tarea necesita hardware
# ni un Pipeline real.
# ---------------------------------------------------------------------------


def _frame(width: int, height: int, value: int = 128) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _bare_qpixmap(frame: np.ndarray) -> QPixmap:
    """Conversion BGR -> QPixmap INDEPENDIENTE de ``video_view.py``, para
    comparar contra lo que ``MainWindow`` termina mostrando sin acoplarse a
    como ``video_view`` hace esa misma conversion internamente (Finding 1,
    ronda de revision 1)."""
    frame = np.ascontiguousarray(frame)
    height, width = frame.shape[:2]
    image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888)
    return QPixmap.fromImage(image.copy())


def _plane(
    *,
    state: CalibrationState = CalibrationState.OK,
    mm_per_px: float = 0.2,
    residual_mm: float = 0.18,
    H: np.ndarray | None = None,
) -> PlaneCalibration:
    # ``H`` default es la identidad: la MAYORIA de las pruebas de este
    # archivo prueban layout/paneles, no geometria de homografia, y una
    # identidad deja el lienzo igual al cuadro de entrada para esas. La
    # prueba del marcador (fix de la demo de LinkedIn, ver mas abajo) pasa
    # una ``H`` real para poder discriminar "se transformo" de "no se
    # transformo" -- con la identidad las dos son indistinguibles.
    return PlaneCalibration(
        H=H if H is not None else np.eye(3), residual_mm=residual_mm, state=state, mm_per_px=mm_per_px
    )


def _intrinsics(*, rms_px: float = 0.21, image_size: tuple[int, int] = (1920, 1080)) -> Intrinsics:
    return Intrinsics(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros(5),
        rms_px=rms_px,
        image_size=image_size,
    )


def _aggregate(*, track_id: int = 1, locked: bool = True, samples: int = 30) -> TrackAggregate:
    return TrackAggregate(
        track_id=track_id,
        samples=samples,
        locked=locked,
        length_mm=187.3,
        length_std_mm=0.8,
        caliper_mm=38.4,
        caliper_std_mm=0.5,
        chord_mm=152.0,
        volume_mm3=1.0e5,
        window=TEMPORAL_WINDOW,
    )


def _banana_result(*, track_id: int = 1, locked: bool = True, passes: bool = True) -> BananaResult:
    mask = np.zeros((200, 300), dtype=bool)
    mask[50:150, 50:250] = True
    # Esta tarea prueba layout de ventana/paneles, no la geometria del
    # overlay (encargo posterior al plan de 20 tareas) -- sin curvas que
    # dibujar aqui, los cuatro campos quedan vacios.
    empty_path = np.empty((0, 2), dtype=np.float64)
    return BananaResult(
        track_id=track_id,
        aggregate=_aggregate(track_id=track_id, locked=locked),
        verdict=CodexVerdict(passes=passes, length_ok=passes, caliper_ok=passes, commercial_grade=38),
        estimated_weight_g=145.0,
        mask=mask,
        contour_px=empty_path,
        axis_path_px=empty_path,
        axis_tips_px=(np.zeros(2), np.zeros(2)),
        outer_curve_px=empty_path,
    )


def _frame_result(
    *,
    rectified: np.ndarray | None,
    raw: np.ndarray | None = None,
    plane: PlaneCalibration | None = None,
    results: list[BananaResult] | None = None,
    warnings: list[str] | None = None,
    markers: list | None = None,
) -> FrameResult:
    if raw is None:
        raw = _frame(640, 480, value=200)
    return FrameResult(
        rectified=rectified,
        raw_undistorted=raw,
        plane=plane,
        results=results or [],
        warnings=warnings or [],
        markers=markers or [],
    )


class RecordingPipeline:
    """Sustituto de ``Pipeline``: registra en que ``QThread`` corrio ``process``.

    Es lo unico que puede probar la Ruling 4 (nunca procesar en el hilo de
    la GUI): un ``Pipeline`` real tardaria y traeria YOLO, que no esta
    instalado en esta maquina a proposito.
    """

    def __init__(self) -> None:
        self.threads: list[QThread] = []
        self.frames: list[np.ndarray] = []

    def process(self, frame: np.ndarray) -> FrameResult:
        self.threads.append(QThread.currentThread())
        self.frames.append(frame)
        # ``plane`` no puede ser ``None`` junto con un ``rectified`` real:
        # el ``Pipeline`` de verdad nunca produce esa combinacion (BLOCKED
        # es la UNICA razon para ``rectified is None``, y ahi ``plane``
        # tambien es ``None`` -- ver docstring de ``FrameResult``).
        # ``update_frame_result`` (Fix 1, demo de LinkedIn) ahora lee
        # ``plane.H`` para proyectar los marcadores al espacio del
        # lienzo; un sustituto que rompiera esa invariante (como la
        # version anterior de este metodo, sin ``plane``) la haria
        # explotar con un ``AttributeError`` sobre ``None`` -- fiel al
        # contrato real, no un defecto de este sustituto.
        return _frame_result(rectified=frame, raw=frame, plane=_plane())


class BlockingPipeline:
    """``Pipeline`` sustituto que se QUEDA dentro de ``process`` hasta que la
    prueba lo suelta.

    Es la unica forma de ejercitar la contrapresion (C1): con un pipeline
    instantaneo, el worker vacia la cola entre una entrega y la siguiente y
    nunca se llega al estado "ocupado" que la contrapresion gobierna -- la
    prueba quedaria en verde con y sin bandera.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.frames: list[np.ndarray] = []
        self._lock = threading.Lock()

    def process(self, frame: np.ndarray) -> FrameResult:
        with self._lock:
            self.frames.append(frame)
        self.entered.set()
        self.release.wait(timeout=5.0)
        return _frame_result(rectified=frame, raw=frame, plane=_plane())

    def frame_values(self) -> list[int]:
        with self._lock:
            return [int(f.flat[0]) for f in self.frames]


class RaisingPipeline:
    """``Pipeline`` sustituto que lanza una de las DOS excepciones que
    ``Pipeline.process`` documenta (I5)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    def process(self, frame: np.ndarray) -> FrameResult:
        self.calls += 1
        raise self._exc


def _pip_color(window: MainWindow):
    """Color de un pixel del PiP, que es la capa CRUDA de ``VideoView``.

    Los cuadros sinteticos de este archivo son uniformes, asi que el valor
    de un pixel del PiP identifica inequivocamente de QUE cuadro de camara
    se construyo -- que es como se distingue "el video sigue a la camara"
    de "el video sigue al pipeline" sin leer ninguna bandera interna.
    """
    image = window.video_view.pip_pixmap.toImage()
    return image.pixelColor(image.width() // 2, image.height() // 2)


# ---------------------------------------------------------------------------
# Layout (spec S8 variante 1)
# ---------------------------------------------------------------------------


def test_side_panel_width_is_fixed_at_240_px(qtbot):
    """Al maximizar debe crecer el video, no el panel -- el panel usa
    ``setFixedWidth``, nunca un factor de estiramiento."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    window.resize(900, 600)
    qtbot.wait(50)
    width_before = window.side_panel.width()

    window.resize(2400, 1400)
    qtbot.wait(50)

    assert width_before == SIDE_PANEL_WIDTH_PX
    assert window.side_panel.width() == SIDE_PANEL_WIDTH_PX


def test_video_view_takes_the_remaining_width(qtbot):
    """Los layouts de nivel superior tienen margen y espaciado cero (spec
    S8: el video domina), asi que la igualdad es EXACTA, no aproximada."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    window.resize(1200, 800)
    qtbot.wait(50)

    assert window.video_view.width() == window.width() - SIDE_PANEL_WIDTH_PX


def test_pip_toggles_with_the_keyboard_in_both_directions(qtbot):
    """Ruling 1: una implementacion que solo apague el PiP (y nunca lo
    vuelva a mostrar) no puede pasar esto -- se verifica el estado
    inicial y el viaje de vuelta."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    assert window.video_view.pip_visible is True

    qtbot.keyClick(window, Qt.Key.Key_P)
    assert window.video_view.pip_visible is False

    qtbot.keyClick(window, Qt.Key.Key_P)
    assert window.video_view.pip_visible is True


def test_pip_width_is_a_fraction_of_the_widget_not_of_the_source_frame(qtbot):
    """Fix 2 (pedido explicito del autor: "haz que la ventana donde se ve
    el banano normal ocupe menos espacio"). La implementacion ANTERIOR
    calculaba ``PIP_SCALE`` sobre el ANCHO NATIVO del cuadro crudo
    (``frame.shape[1]``), no sobre el widget donde se pinta -- con una
    camara de 1920 px pintada en un widget de 800 px eso fijaba el PiP en
    ``round(1920 * 0.25) = 480`` px SIEMPRE, el 60% del widget, sin
    importar cuanto se redimensionara la ventana (la queja real del
    autor: en su pantalla el PiP ocupaba ~40% y tapaba el extremo del
    tallo).

    Se eligen un ancho de widget (800) y un ancho de cuadro crudo (1920)
    deliberadamente DISTINTOS para que las dos bases de calculo diverjan:
    la implementacion vieja da 480, la correcta da
    ``round(800 * PIP_SCALE) = 200``. Afirmar solo "el PiP es chico" no
    discrimina -- 480 TAMBIEN es chico frente a un widget de 800 px; lo
    que discrimina es afirmar el numero EXACTO que cada base produce, y
    rechazar el de la base equivocada.
    """
    view = VideoView()
    qtbot.addWidget(view)
    view.resize(800, 600)

    raw = _frame(1920, 1080)
    rectified = _frame(1900, 1000)
    view.set_frame(rectified, raw)

    wrong_frame_relative_width = round(raw.shape[1] * PIP_SCALE)  # 480 -- el bug
    correct_widget_relative_width = round(view.width() * PIP_SCALE)  # 200 -- el fix
    assert correct_widget_relative_width != wrong_frame_relative_width, "el fixture debe discriminar"
    assert view.pip_pixmap.width() == correct_widget_relative_width
    assert view.pip_pixmap.width() != wrong_frame_relative_width


def test_pip_is_rendered_small_but_still_legible(qtbot):
    """Ruling 1: ``pip.width() < main.width() // 3`` por si solo lo satisface
    un PiP de un pixel. Se agrega un piso absoluto de legibilidad: el PiP
    tiene que ser chico Y seguir sirviendo como miniatura en un video real.

    Widget redimensionado EXPLICITAMENTE (Fix 2): el ancho del PiP ahora
    depende de ``self.width()``, no del tamano por defecto (no mostrado)
    de un ``QWidget`` -- fijar la ventana a un tamano realista de la
    pantalla del README (1300x820) hace que esta prueba no dependa de un
    default implicito de Qt que podria cambiar entre versiones/plataformas.
    """
    view = VideoView()
    qtbot.addWidget(view)
    view.resize(1300, 820)

    raw = _frame(1920, 1080)
    rectified = _frame(1900, 1000)
    view.set_frame(rectified, raw)

    assert view.pip_pixmap.width() < view.width() // 3
    assert view.pip_pixmap.width() >= PIP_MIN_WIDTH_PX, (
        "el PiP no puede degenerar a algo ilegible en video"
    )


def test_pip_floor_binds_on_a_narrow_widget(qtbot):
    """I11 / Fix 2: el caso donde ``PIP_MIN_WIDTH_PX`` es la restriccion
    ACTIVA -- ahora sobre el ancho del WIDGET, no del cuadro crudo (ver
    test de arriba: esa es la base que cambio en este fix).

    Con un widget de 500 px, ``round(500 * PIP_SCALE) = 125`` esta por
    DEBAJO del piso, asi que el piso es lo que decide el ancho. Rebajar
    ``PIP_MIN_WIDTH_PX`` (a 80, a 1) cambia el resultado de esta prueba y la
    pone roja -- cosa que la prueba de arriba, con su widget de 800/1300 px,
    no puede hacer ni con el piso en cero.

    El cuadro crudo sigue siendo GRANDE (1920 px) a proposito: si el piso
    todavia se calculara (por error) sobre el ancho del cuadro en vez del
    widget, ``round(1920 * PIP_SCALE) = 480`` NO caeria bajo el piso, y
    esta prueba fallaria -- es la misma discriminacion de la primera
    prueba de esta seccion, aplicada al piso en vez de a la fraccion.

    Se exige ademas que el piso no AGRANDE la miniatura por encima del
    cuadro crudo (``_downscale_for_pip`` devuelve el cuadro tal cual en ese
    caso): un piso que escalara hacia arriba inventaria pixeles.
    """
    view = VideoView()
    qtbot.addWidget(view)
    view.resize(500, 350)

    raw = _frame(1920, 1080)
    view.set_frame(_frame(1900, 1000), raw)

    assert round(view.width() * PIP_SCALE) < PIP_MIN_WIDTH_PX, "el fixture debe caer bajo el piso"
    assert view.pip_pixmap.width() == PIP_MIN_WIDTH_PX
    assert view.pip_pixmap.width() <= raw.shape[1], "el piso no puede agrandar la miniatura"


def test_status_bar_shows_calibration_state_in_english(qtbot):
    window = MainWindow(intrinsics=_intrinsics())
    qtbot.addWidget(window)

    window.update_frame_result(_frame_result(rectified=_frame(900, 600), plane=_plane()))

    assert "4/4 markers" in window.status_text()
    assert "calibracion" not in window.status_text().lower()
    assert "calibración" not in window.status_text().lower()


def test_blocked_calibration_is_a_warning_but_healthy_calibration_is_not(qtbot):
    """Ruling 1: el brief solo probaba el lado ``True``, que no distingue
    "avisa cuando corresponde" de "avisa siempre"."""
    window = MainWindow(intrinsics=_intrinsics())
    qtbot.addWidget(window)

    window.update_frame_result(
        _frame_result(
            rectified=None,
            plane=None,
            warnings=["Calibration blocked: only 2/4 ArUco markers are visible..."],
        )
    )
    assert window.status_is_warning() is True
    # C2 (ronda de correcciones finales): esta prueba PASABA una lista de
    # ``warnings`` poblada y nunca comprobaba que llegara a ninguna parte --
    # creaba la apariencia de cobertura de un canal que no existia
    # (``update_frame_result`` leia ``plane`` y ``results`` e ignoraba
    # ``warnings``). El aviso del pipeline tiene que poder LEERSE.
    assert "only 2/4 ArUco markers" in window.notice_text()

    window.update_frame_result(_frame_result(rectified=_frame(900, 600), plane=_plane()))
    assert window.status_is_warning() is False
    # Y el lado sano no deja un aviso viejo colgado en pantalla: un canal que
    # solo acumula es tan inutil como uno que no muestra nada.
    assert window.notice_text() == ""


def test_status_bar_shows_degraded_markers_as_a_warning_with_three_markers(qtbot):
    """Carry-over de la Task 18 (Ruling 4 de esta tarea): el brief original
    y todas las pruebas de ``_plane(...)`` en este archivo solo ejercitan
    ``CalibrationState.OK`` -- la rama DEGRADED (3/4 marcadores) de
    ``_markers_segment`` nunca corrio bajo una prueba, a pesar de ser la
    rama que spec S7 etapa 2 nombra explicitamente ("con 3 marcadores el
    sistema sigue funcionando y lo indica")."""
    window = MainWindow(intrinsics=_intrinsics())
    qtbot.addWidget(window)

    window.update_frame_result(
        _frame_result(
            rectified=_frame(900, 600),
            plane=_plane(state=CalibrationState.DEGRADED),
            warnings=["Degraded calibration: 3/4 ArUco markers visible..."],
        )
    )

    assert f"3/{MARKER_COUNT} markers" in window.status_text()
    assert window.status_is_warning() is True
    # C2: igual que la prueba de arriba, esta pasaba ``warnings`` y no
    # comprobaba que se vieran. El texto del aviso de degradacion dice algo
    # que la barra de estado NO dice -- sobre cuantas correspondencias de
    # esquina se esta midiendo -- asi que perderlo no es redundante.
    assert "Degraded calibration" in window.notice_text()

    # Y el lado sano sigue sin avisar -- la misma distincion que la Ruling 1
    # ya exigio para el caso bloqueado/sano (no "avisa siempre").
    window.update_frame_result(_frame_result(rectified=_frame(900, 600), plane=_plane()))
    assert f"{MARKER_COUNT}/{MARKER_COUNT} markers" in window.status_text()
    assert window.status_is_warning() is False
    assert window.notice_text() == ""


def test_side_panel_does_not_leave_stale_cards_visible_after_an_update(qtbot):
    """Encontrado revisando una captura de pantalla, no un conteo: ``takeAt``
    saca un widget de la CONTABILIDAD del layout, pero sigue siendo un hijo
    visible de ``_cards_container`` con su geometria vieja hasta que de
    verdad se destruye -- y ``deleteLater`` por si sola solo agenda esa
    destruccion para una vuelta futura del bucle de eventos. Sin
    ``setParent(None)`` la tarjeta (o el aviso "No fruit detected") del
    cuadro anterior queda dibujada encima de las nuevas durante ese hueco:
    invisible para cualquier prueba que solo cuente tarjetas."""
    panel = SidePanel()
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitExposed(panel)

    # Sin ningun ``qtbot.wait``/``processEvents`` entre medio A PROPOSITO:
    # ``deleteLater`` agenda la destruccion para una vuelta futura del
    # bucle de eventos, asi que darle tiempo a correr esconderia el defecto
    # (el "No fruit detected" original terminaria desapareciendo solo, de
    # casualidad, antes de que la prueba lo mirara). Lo que tiene que ser
    # INMEDIATO es la desconexion del arbol de widgets visible
    # (``setParent(None)``), no la liberacion de memoria.
    # La escala se pasa EXPLICITA: ``update_results`` ya no tiene un
    # default de 0.2 mm/px (era codigo muerto, y un default plausible en el
    # camino de la escala invita a medir con una escala que nadie calibro).
    panel.update_results([_banana_result(track_id=1)], 0.2)
    stale_labels_before = [
        child for child in panel._cards_container.findChildren(QLabel) if child.text() == "No fruit detected"
    ]
    assert stale_labels_before == [], "el aviso vacio inicial no debe seguir visible tras la primera actualizacion"

    panel.update_results([])
    # ``QLabel`` hereda de ``QFrame`` en Qt, asi que el aviso vacio legitimo
    # ("No fruit detected") tambien aparece en ``findChildren(QFrame)``; se
    # filtra por tipo EXACTO para distinguirlo de una tarjeta real
    # (``_build_card`` siempre devuelve un ``QFrame`` liso, nunca un ``QLabel``).
    visible_cards = [child for child in panel._cards_container.findChildren(QFrame) if type(child) is QFrame]
    assert visible_cards == [], "una tarjeta del cuadro anterior no debe seguir visible tras vaciarse"


# ---------------------------------------------------------------------------
# Ruling 2 -- rectified is None no puede dejar la ventana en negro
# ---------------------------------------------------------------------------


def test_video_view_shows_the_raw_frame_with_a_warning_when_rectified_is_none(qtbot):
    """``FrameResult.rectified`` es ``None`` -- no un lienzo en negro -- cuando
    la calibracion esta BLOCKED (Task 14). Sin este manejo la ventana
    queda en negro o lanza en el primer cuadro sin los 4 marcadores."""
    view = VideoView()
    qtbot.addWidget(view)

    raw = _frame(640, 480, value=222)  # valor distintivo, lejos de 0 (negro)
    view.set_frame(None, raw)

    assert view.is_showing_raw_fallback is True
    assert not view.main_pixmap.isNull()
    # El lienzo principal tiene que venir del CRUDO, no de un placeholder en
    # negro: se muestrea una esquina lejos de donde se dibuja el aviso.
    image = view.main_pixmap.toImage()
    sampled = image.pixelColor(image.width() - 5, image.height() - 5)
    assert sampled.red() > 100, "el area principal debe mostrar el cuadro crudo, no un lienzo en negro"

    # Con calibracion sana, no se activa el camino de respaldo.
    view.set_frame(_frame(900, 600), raw)
    assert view.is_showing_raw_fallback is False


def test_main_window_update_frame_result_does_not_crash_when_rectified_is_none(qtbot):
    """Extremo a extremo: ``MainWindow`` tiene que sobrevivir un ``FrameResult``
    BLOCKED sin lanzar y sin dejar el video en negro."""
    window = MainWindow(intrinsics=_intrinsics())
    qtbot.addWidget(window)

    window.update_frame_result(
        _frame_result(rectified=None, plane=None, warnings=["Calibration blocked: only 1/4 ArUco markers..."])
    )

    assert window.video_view.is_showing_raw_fallback is True
    assert not window.video_view.main_pixmap.isNull()


# ---------------------------------------------------------------------------
# Ruling 3 -- capture_lost y actual_size tienen que notarse en la interfaz
# ---------------------------------------------------------------------------


def test_capture_lost_is_surfaced_in_the_status_bar(qtbot, monkeypatch):
    """Una camara que se desconecta no puede fallar en silencio: tiene que
    notarse en la barra de estado."""
    from codex_caliper import camera as camera_module

    class _FakeCapture:
        def isOpened(self):
            return True

        def set(self, *a, **kw):
            return True

        def get(self, prop_id):
            return 1920.0

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(camera_module, "open_capture", lambda *a, **kw: _FakeCapture())

    window = MainWindow(intrinsics=_intrinsics())
    qtbot.addWidget(window)

    thread = CameraThread(0)
    window.connect_camera(thread)

    with qtbot.waitSignal(thread.capture_lost, timeout=5000):
        thread.start()
    qtbot.waitUntil(lambda: thread.isFinished(), timeout=2000)
    qtbot.wait(50)

    assert window.status_is_warning() is True
    assert "camera" in window.status_text().lower() or "stopped responding" in window.status_text().lower()

    window.close()


def test_resolution_mismatch_blocks_measurement_instead_of_measuring_wrong(qtbot, monkeypatch):
    """Ruling 3: si la camara concede una resolucion distinta de la calibrada,
    ``require_compatible`` rechaza -- el sistema no debe medir con
    parametros que no corresponden, y la interfaz tiene que decirlo."""
    from codex_caliper import camera as camera_module

    class _FakeCapture:
        def __init__(self):
            self._served = False

        def isOpened(self):
            return True

        def set(self, *a, **kw):
            return True

        def get(self, prop_id):
            import cv2

            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 1280.0
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 720.0
            return 0.0

        def read(self):
            import time

            if self._served:
                time.sleep(0.05)
                return False, None
            self._served = True
            return True, _frame(1280, 720)

        def release(self):
            pass

    monkeypatch.setattr(camera_module, "open_capture", lambda *a, **kw: _FakeCapture())

    # Calibrado a 1920x1080; la camara falsa concede 1280x720.
    pipeline = RecordingPipeline()
    window = MainWindow(intrinsics=_intrinsics(image_size=(1920, 1080)), pipeline=pipeline)
    qtbot.addWidget(window)

    thread = CameraThread(0)
    window.connect_camera(thread)

    with qtbot.waitSignal(thread.frame_ready, timeout=5000):
        thread.start()
    qtbot.wait(100)
    thread.stop()

    assert window.measurement_enabled is False
    assert "1280" in window.blocked_reason() or "720" in window.blocked_reason()
    assert window.status_is_warning() is True
    # I10 (ronda de correcciones finales): "bloquea la medicion" tiene que
    # significar que NINGUN cuadro llega al pipeline, no solo que una
    # propiedad devuelva ``False``. Esta prueba ya tenia un
    # ``RecordingPipeline`` conectado y nunca lo miraba; el ``qtbot.wait(100)``
    # de arriba ya le dio al worker tiempo de sobra para haber procesado el
    # cuadro si la compuerta no estuviera.
    assert pipeline.threads == [], "no se puede medir con unos intrinsecos de otra resolucion"

    window.close()


def test_load_intrinsics_none_blocks_measurement_with_a_calibration_reason(qtbot):
    """Review Focus 5 del plan: sin intrinsecos no puede haber medicion."""
    window = MainWindow(intrinsics=_intrinsics())
    qtbot.addWidget(window)
    assert window.measurement_enabled is True

    window.load_intrinsics(None)

    assert window.measurement_enabled is False
    assert "calibration" in window.blocked_reason().lower()


# ---------------------------------------------------------------------------
# Ruling 4 -- Pipeline.process nunca corre en el hilo de la GUI
# ---------------------------------------------------------------------------


def test_frame_processing_never_runs_on_the_gui_thread(qtbot):
    pipeline = RecordingPipeline()
    window = MainWindow(pipeline=pipeline)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    gui_thread = QThread.currentThread()

    window.submit_frame(_frame(16, 12))
    qtbot.waitUntil(lambda: len(pipeline.threads) == 1, timeout=2000)

    assert pipeline.threads[0] is not gui_thread, "Pipeline.process no debe correr en el hilo de la GUI"

    window.close()


def test_frame_result_reaches_the_window_after_processing(qtbot):
    """El resultado procesado en el worker si tiene que llegar de vuelta a
    la ventana (en el hilo de la GUI) para pintarse."""
    pipeline = RecordingPipeline()
    window = MainWindow(pipeline=pipeline)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    frame = _frame(32, 24, value=50)
    window.submit_frame(frame)

    qtbot.waitUntil(lambda: not window.video_view.main_pixmap.isNull(), timeout=2000)

    window.close()


# ---------------------------------------------------------------------------
# Ronda de revision 1 -- tres hallazgos sobre el commit e8b4afc
# ---------------------------------------------------------------------------


def test_update_frame_result_draws_the_measurement_overlay_on_the_video(qtbot):
    """Finding 1 (CRITICO, ronda de revision 1): ``update_frame_result``
    pasaba ``result.rectified`` crudo a ``VideoView`` -- ``draw_results``
    (Task 17) nunca se llamaba, asi que las cotas solo vivian en las
    tarjetas del panel. La spec S8 es explicita sobre por que eso es el
    anti-objetivo exacto del proyecto: "si los numeros viven unicamente en
    la tabla, el video pasa a ser decorativo y la mirada se va al panel".

    La forma que muerde (sugerida en la revision): comparar el lienzo
    MOSTRADO contra el lienzo CRUDO de entrada -- si son iguales, el
    overlay nunca se aplico, sin importar cuantas pruebas cuenten
    resultados o lean texto del panel."""
    window = MainWindow(intrinsics=_intrinsics())
    qtbot.addWidget(window)

    rectified = _frame(900, 600, value=235)
    result = _banana_result(track_id=7, locked=True, passes=True)
    window.update_frame_result(_frame_result(rectified=rectified, plane=_plane(), results=[result]))

    displayed = window.video_view.main_pixmap.toImage()
    bare = _bare_qpixmap(rectified).toImage()
    assert displayed != bare, "el lienzo mostrado debe llevar las cotas dibujadas, no ser el lienzo crudo"

    # Mas especifico que "algo cambio en algun lado": al menos un pixel
    # DENTRO del bloque de etiqueta de ESTE resultado tiene que haber
    # cambiado. No se compara el pixel exacto de ``text_anchor_for`` --
    # ``cv2.putText`` dibuja con ese punto como base de la linea, asi que
    # la tinta cae arriba/a la derecha del origen, no encima de el.
    x, y = text_anchor_for(result.mask)
    region_differs = any(
        displayed.pixelColor(px, py) != bare.pixelColor(px, py)
        for py in range(max(0, y - 20), min(displayed.height(), y + 90))
        for px in range(max(0, x), min(displayed.width(), x + 220))
    )
    assert region_differs, "el bloque de etiqueta de este banano no cambio ningun pixel -- el overlay no se aplico ahi"

    window.close()


def test_update_frame_result_draws_the_marker_overlay_in_the_rectified_canvas_space(qtbot):
    """Fix 1 (demo de LinkedIn, EL bug real de esta ronda): ``result.markers``
    trae las esquinas de deteccion en pixeles de la imagen CRUDA sin
    distorsion (``detect_markers`` corre sobre ``raw_undistorted``, ver
    docstring de ``FrameResult``), no en pixeles del lienzo RECTIFICADO
    que esta prueba pinta -- dos espacios de pixeles DISTINTOS,
    relacionados por ``plane.H`` (``PlaneCalibration.H``: "mapea pixeles de
    imagen a pixeles del lienzo rectificado", ver su docstring).

    La version de esta prueba en la ronda anterior usaba ``_plane()`` con
    su ``H`` por defecto (identidad): con una identidad, dibujar las
    esquinas SIN transformar es PIXEL POR PIXEL indistinguible de
    transformarlas -- exactamente el tipo de prueba que el encargo nombra
    como de las trece que pasaban igual con una implementacion correcta y
    con una incorrecta. Visible en la grabacion real de la demo: un solo
    cuadrilatero verde flotando sobre el lomo del banano (un marcador
    pintado en su coordenada de imagen cruda, que cae ahi por pura
    coincidencia geometrica), mientras los 4 marcadores reales -- en las
    esquinas del lienzo -- no tenian ningun contorno encima.

    Aqui ``H`` es una TRASLACION real (``(+260, +200)``, sin rotacion para
    que la aritmetica de regiones sea trivial de leer), asi que una
    implementacion correcta y una incorrecta (sin transformar) DIFIEREN en
    donde pintan, y la prueba lo comprueba en AMBAS regiones -- no solo
    que "algo se dibujo en la region correcta" (eso solo, una
    implementacion que dibujara en los DOS sitios seguiria pasandolo): el
    marcador tiene que aparecer en sus esquinas YA TRANSFORMADAS
    (300..360, 240..300) Y NO en sus esquinas CRUDAS (30..110, 30..110),
    que es exactamente donde lo deja el bug.
    """
    window = MainWindow(intrinsics=_intrinsics())
    qtbot.addWidget(window)

    rectified = _frame(900, 600, value=235)
    marker = MarkerDetection(
        id=1,
        corners=np.array([[40.0, 40.0], [100.0, 40.0], [100.0, 100.0], [40.0, 100.0]], dtype=np.float32),
    )
    translation_H = np.array([[1.0, 0.0, 260.0], [0.0, 1.0, 200.0], [0.0, 0.0, 1.0]])
    window.update_frame_result(
        _frame_result(rectified=rectified, plane=_plane(H=translation_H), results=[], markers=[marker])
    )

    displayed = window.video_view.main_pixmap.toImage()
    bare = _bare_qpixmap(rectified).toImage()

    def _region_differs(x0: int, y0: int, x1: int, y1: int) -> bool:
        return any(
            displayed.pixelColor(px, py) != bare.pixelColor(px, py)
            for py in range(y0, y1)
            for px in range(x0, x1)
        )

    assert _region_differs(300, 240, 360, 300), (
        "el marcador no se dibujo en su posicion TRANSFORMADA (image_px -> canvas_px via H)"
    )
    assert not _region_differs(30, 30, 110, 110), (
        "el marcador se dibujo en sus esquinas CRUDAS (sin transformar) -- el bug real: "
        "pintar coordenadas de imagen cruda directo sobre un lienzo que es otro espacio"
    )

    window.close()


def test_video_keeps_updating_while_measurement_is_gated_off(qtbot):
    """Finding 2 (Importante, ronda de revision 1): ``_on_camera_frame``
    retornaba sin tocar ``video_view`` cuando ``measurement_enabled`` es
    ``False`` -- como el video solo se actualizaba al volver de
    ``submit_frame`` -> worker -> ``result_ready``, el cuadro se congelaba
    aunque la camara siguiera entregando. Es exactamente la razon que la
    Ruling 2 le dio a ``raw_undistorted``: "siempre esta presente... para
    que la interfaz siempre tenga algo que mostrar" -- y el primer estado
    que un operador ve, sin intrinsecos cargados todavia, es justo este.

    Se llama a ``_on_camera_frame`` directamente (sin levantar un
    ``CameraThread`` real): la mecanica de la camara ya esta cubierta por
    otras pruebas de este archivo, lo que esta sin cubrir es la reaccion
    de ``MainWindow`` cuando la medicion esta bloqueada."""
    pipeline = RecordingPipeline()
    window = MainWindow(intrinsics=None, pipeline=pipeline)
    qtbot.addWidget(window)
    assert window.measurement_enabled is False

    frame = _frame(320, 240, value=180)
    window._on_camera_frame(frame)

    assert not window.video_view.main_pixmap.isNull(), "el video no puede quedar vacio mientras la camara entrega cuadros"
    assert window.video_view.is_showing_raw_fallback is True

    # I10 (ronda de correcciones finales): sin esta espera la asercion de
    # abajo NO PODIA FALLAR. ``submit_frame`` emite una senal encolada hacia
    # otro hilo; en el instante siguiente a la llamada, el worker todavia no
    # pudo haber sacado nada de la cola, asi que ``pipeline.threads == []``
    # se cumplia con la compuerta y sin ella. 100 ms es mas de lo que tarda
    # ``RecordingPipeline.process`` por tres ordenes de magnitud.
    qtbot.wait(100)
    assert pipeline.threads == [], "ningun cuadro debe llegar al pipeline mientras la medicion esta bloqueada"
    # El operador tambien tiene que poder LEER por que no se esta midiendo.
    assert "no intrinsic calibration" in window.notice_text().lower()

    window.close()


# ---------------------------------------------------------------------------
# C1 -- contrapresion y cadencia del video
# ---------------------------------------------------------------------------


def test_only_one_frame_is_in_flight_and_the_backlog_is_coalesced(qtbot):
    """C1 (CRITICO): ``_on_camera_frame`` llamaba a ``submit_frame`` por CADA
    cuadro de camara sobre una conexion encolada de Qt SIN TOPE -- ni bandera
    de ocupado, ni descarte, ni ``BlockingQueuedConnection``. A 30 cuadros/s
    de entrada contra 3-10 cuadros/s de rendimiento real del pipeline en CPU
    se acumulan ~25 cuadros/s de ~6.2 MB cada uno (~155 MB/s): la memoria del
    proceso se agota en 30-60 segundos, dentro del primer minuto de la unica
    demostracion que el proyecto publica.

    La implementacion equivocada que esta prueba RECHAZA es exactamente la
    anterior: sin bandera de ocupado, los diez cuadros que se entregan
    mientras el worker esta bloqueado se encolan todos y el pipeline termina
    viendo los once. Con contrapresion ve DOS: el que atrapo primero y el
    ULTIMO que llego mientras estaba ocupado -- los nueve intermedios se
    descartan, que es lo correcto y no solo lo barato (para fruta quieta, el
    cuadro util es el mas reciente).

    Una asercion de tipo "paso algo" no distingue las dos: hay que fijar el
    conjunto EXACTO de cuadros que vio el pipeline.
    """
    pipeline = BlockingPipeline()
    window = MainWindow(intrinsics=_intrinsics(), pipeline=pipeline)
    qtbot.addWidget(window)

    # Cuadro 0: lo toma el worker y se queda dentro de ``process``.
    window.submit_frame(_frame(32, 24, value=0))
    assert pipeline.entered.wait(timeout=5.0), "el worker nunca entro a process"

    # Cuadros 1..10 con el worker ocupado: nueve se tienen que perder.
    for value in range(1, 11):
        window.submit_frame(_frame(32, 24, value=value))

    assert window.dropped_frame_count == 9, (
        f"se descartaron {window.dropped_frame_count} cuadros, deberian ser 9 "
        "(sin contrapresion se descartan 0 y los once se encolan)"
    )

    pipeline.release.set()
    qtbot.waitUntil(lambda: len(pipeline.frames) == 2, timeout=5000)
    qtbot.wait(100)

    assert pipeline.frame_values() == [0, 10], (
        "el pipeline debe ver el primer cuadro y el ULTIMO, nunca la cola entera"
    )

    window.close()


def test_the_video_follows_the_camera_not_the_pipeline(qtbot):
    """C1, segunda mitad: con la medicion activa, ``video_view`` se
    actualizaba UNICAMENTE desde ``update_frame_result``, asi que la cadencia
    del video ERA la cadencia del pipeline (3-10/s en CPU) -- lo contrario de
    lo que el README afirmaba y de lo que la spec S13 publica como su
    artefacto de mayor impacto.

    La implementacion equivocada que esta prueba RECHAZA: la anterior, en la
    que ``_on_camera_frame`` reenvia el cuadro y no pinta nada. Aqui el
    pipeline se queda BLOQUEADO dentro de ``process`` -- no puede devolver
    ningun resultado -- y a pesar de eso cada cuadro nuevo de camara tiene
    que cambiar los pixeles de la capa cruda. Si la pintura dependiera del
    pipeline, los tres cuadros se verian identicos.
    """
    pipeline = BlockingPipeline()
    window = MainWindow(intrinsics=_intrinsics(), pipeline=pipeline)
    qtbot.addWidget(window)

    # El primer cuadro entra al worker, que se queda DENTRO de ``process``:
    # a partir de aqui el pipeline no puede devolver nada, y se espera el
    # evento en vez de confiar en el reloj (la cola de Qt es asincrona).
    window._on_camera_frame(_frame(640, 480, value=30))
    assert pipeline.entered.wait(timeout=5.0), "el worker nunca entro a process"

    assert not window.video_view.pip_pixmap.isNull(), (
        "el primer cuadro de camara no pinto nada -- el video depende del pipeline"
    )
    seen = [_pip_color(window).red()]
    for value in (140, 250):
        window._on_camera_frame(_frame(640, 480, value=value))
        seen.append(_pip_color(window).red())

    assert len(set(seen)) == 3, (
        f"el video no siguio a la camara: los tres cuadros se vieron como {seen} "
        "mientras el pipeline estaba bloqueado"
    )
    # Y lo que se ve es el ULTIMO cuadro de camara, no el primero que el
    # pipeline alcanzo a tomar.
    assert abs(seen[-1] - 250) <= 2

    pipeline.release.set()
    window.close()


# ---------------------------------------------------------------------------
# C2 -- los avisos del pipeline tienen un canal donde verse
# ---------------------------------------------------------------------------


def test_the_adjacency_refusal_reaches_the_live_view(qtbot):
    """C2 (CRITICO): ``FrameResult.warnings`` no tenia NINGUN consumidor en
    todo el proyecto. El pipeline escribe ahi cinco mensajes de operador
    distintos y ninguno podia verse -- incluido el de dos frutos pegados, que
    la spec S7 etapa 4 no deja opcional: el sistema tiene que negarse a medir
    *y mostrar el aviso*, porque "una medida falsa es peor que ninguna".

    El fallo normal: dos bananos a 1.5 mm, el panel dice "No fruit detected"
    y la barra de estado se ve perfectamente sana -- sin ninguna via para
    enterarse de que el sistema esta rechazando, ni por que.

    La implementacion equivocada que esta prueba RECHAZA es la anterior, que
    leia ``plane`` y ``results`` y pasaba de largo por ``warnings``. Se
    afirma la FRASE DE ACCION completa ("Separate them on the tray."), no
    solo que el texto no este vacio: un canal que muestre un generico "a
    warning occurred" deja al operador sin saber que hacer, y pasaria una
    asercion mas laxa.

    La prueba que fija la redaccion exacta del productor contra este mismo
    canal vive junto a su fixture, en
    ``tests/test_pipeline.py::test_the_adjacency_warning_the_pipeline_writes_is_the_one_the_window_shows``.
    """
    window = MainWindow(intrinsics=_intrinsics())
    qtbot.addWidget(window)

    window.update_frame_result(
        _frame_result(
            rectified=_frame(900, 600),
            plane=_plane(),
            results=[],
            warnings=[
                "Fruits 1 and 2 are closer than 2.0 mm, so they cannot be measured "
                "separately. Separate them on the tray."
            ],
        )
    )

    assert "Separate them on the tray." in window.notice_text()
    assert "closer than 2.0 mm" in window.notice_text()


def test_several_warnings_are_all_shown_not_just_the_first(qtbot):
    """Un cuadro puede traer varios avisos a la vez (calibracion degradada
    MAS un fruto que toca el borde del area calibrada). Un canal que muestre
    solo ``warnings[0]`` esconde el resto -- y en ese par concreto el que se
    perderia es el que explica por que falta una medida."""
    window = MainWindow(intrinsics=_intrinsics())
    qtbot.addWidget(window)

    window.update_frame_result(
        _frame_result(
            rectified=_frame(900, 600),
            plane=_plane(state=CalibrationState.DEGRADED),
            warnings=[
                "Degraded calibration: 3/4 ArUco markers visible.",
                "Fruit 4: its outline touches the border of the calibrated area.",
            ],
        )
    )

    assert "Degraded calibration" in window.notice_text()
    assert "touches the border" in window.notice_text()


# ---------------------------------------------------------------------------
# I5 -- una excepcion del pipeline no es un traceback por cuadro
# ---------------------------------------------------------------------------


def test_a_degenerate_marker_layout_becomes_an_operator_message_not_a_traceback(qtbot):
    """I5: ``Pipeline.process`` DOCUMENTA que lanza ``DegenerateMarkerLayout``
    (``marker_layout`` colineal, un error de tipeo que pasa la validacion de
    arranque) y no habia ningun manejador. El resultado era un traceback
    crudo por CADA cuadro, el video congelado y la barra de estado en verde:
    el contrato "nunca un traceback" del punto de entrada valia solo hasta
    que existia una ventana.

    La implementacion equivocada que esta prueba RECHAZA: la anterior, sin
    ``try`` en el worker -- la excepcion viaja por el bucle de eventos de Qt
    y la ventana no se enterera de nada (``measurement_enabled`` seguiria en
    ``True`` y ``notice_text()`` vacio).

    Se afirma ademas que el pipeline se llama UNA SOLA VEZ: bloquear la
    medicion es lo que impide que el cuadro pendiente se reentregue y la
    ventana quede girando sobre el mismo error un cuadro a la vez.
    """
    pipeline = RaisingPipeline(DegenerateMarkerLayout("collinear marker positions"))
    window = MainWindow(intrinsics=_intrinsics(), pipeline=pipeline)
    qtbot.addWidget(window)

    window._on_camera_frame(_frame(64, 48))
    qtbot.waitUntil(lambda: window.notice_text() != "", timeout=2000)

    assert "marker_layout" in window.notice_text()
    assert window.measurement_enabled is False
    assert window.status_is_warning() is True

    # Un segundo cuadro de camara no vuelve a entrar al pipeline.
    window._on_camera_frame(_frame(64, 48, value=200))
    qtbot.wait(100)
    assert pipeline.calls == 1, "la medicion bloqueada no puede seguir reintentando"

    window.close()


def test_an_intrinsics_mismatch_raised_inside_the_pipeline_blocks_measurement(qtbot):
    """I5, la otra excepcion documentada. Se enruta al MISMO campo que la
    guardia de ``actual_size`` (misma causa, mismo mensaje), para que
    ``load_intrinsics`` la limpie igual que limpia la otra -- a diferencia
    del layout degenerado, que ninguna calibracion nueva arregla."""
    pipeline = RaisingPipeline(IntrinsicsMismatch("calibrated at (1920, 1080), got (1280, 720)"))
    window = MainWindow(intrinsics=_intrinsics(), pipeline=pipeline)
    qtbot.addWidget(window)

    window._on_camera_frame(_frame(64, 48))
    qtbot.waitUntil(lambda: window.notice_text() != "", timeout=2000)

    assert "1280" in window.notice_text()
    assert window.measurement_enabled is False

    window.load_intrinsics(_intrinsics(image_size=(1280, 720)))
    assert window.measurement_enabled is True, (
        "una calibracion nueva tiene que poder deshacer un bloqueo de resolucion"
    )

    window.close()


# ---------------------------------------------------------------------------
# I4 -- load_intrinsics tambien reemplaza los del pipeline
# ---------------------------------------------------------------------------


class _IntrinsicsHoldingPipeline:
    """Sustituto que solo guarda sus intrinsecos, como hace ``Pipeline``."""

    def __init__(self, intrinsics: Intrinsics) -> None:
        self.intrinsics = intrinsics

    def process(self, frame: np.ndarray) -> FrameResult:
        return _frame_result(rectified=frame, raw=frame, plane=_plane())


def test_load_intrinsics_also_replaces_the_pipelines_own_reference(qtbot):
    """I4: ``Pipeline`` guarda su PROPIA referencia a los intrinsecos y la usa
    para ``undistort`` en cada cuadro. ``load_intrinsics`` actualizaba solo la
    de la ventana, asi que la barra de estado habria mostrado el RMS de la
    calibracion NUEVA mientras cada medida se seguia corrigiendo con la
    VIEJA -- en verde, sin ningun aviso.

    ``require_compatible`` no cubre este camino: dos calibraciones de la MISMA
    resolucion y distinta distorsion pasan esa guardia sin objetar nada. Por
    eso la asercion compara el OBJETO, no la resolucion.

    La implementacion equivocada que esta prueba RECHAZA: la anterior, que
    asignaba ``self._intrinsics`` y nada mas. Hoy es inalcanzable (nadie
    llama a ``set_live_frame``, ver "Known limits" del README) y por eso
    mismo importa: el README apunta al lector hacia cablear esa pantalla, y
    hacerlo sin esta linea no deja una pantalla incompleta -- deja un
    sistema que mide mal en silencio.
    """
    old_intrinsics = _intrinsics(rms_px=0.21)
    pipeline = _IntrinsicsHoldingPipeline(old_intrinsics)
    window = MainWindow(intrinsics=old_intrinsics, pipeline=pipeline)
    qtbot.addWidget(window)

    # Misma resolucion, otra calibracion: ``require_compatible`` no puede
    # detectar la diferencia, solo la propagacion correcta puede.
    new_intrinsics = _intrinsics(rms_px=0.33)
    window.load_intrinsics(new_intrinsics)

    assert pipeline.intrinsics is new_intrinsics, (
        "el pipeline siguio corrigiendo distorsion con los intrinsecos viejos"
    )
    assert "0.33" in window.status_text()

    window.close()


# ---------------------------------------------------------------------------
# I8 -- el tercer campo de la barra de estado es una medida, no una constante
# ---------------------------------------------------------------------------


def test_status_bar_shows_the_measured_residual_not_the_configured_scale(qtbot):
    """I8: el tercer campo mostraba ``plane.mm_per_px``, que
    ``compute_plane_calibration`` copia literalmente de
    ``cfg.canvas_mm_per_px``: imprimia "0.200 mm/px" para siempre y no podia
    indicar salud, aunque el README lo llamara "la escala en vivo". Una
    constante disfrazada de lectura en vivo es, en un instrumento de
    medicion, peor que no mostrar nada.

    La implementacion equivocada que esta prueba RECHAZA: la anterior. Los
    dos planos de abajo comparten ``mm_per_px`` EXACTAMENTE (0.2, como en el
    rig real) y difieren solo en el residual, asi que la version vieja
    imprimia el mismo texto para los dos y no podria distinguirlos.
    """
    window = MainWindow(intrinsics=_intrinsics())
    qtbot.addWidget(window)

    window.update_frame_result(
        _frame_result(rectified=_frame(900, 600), plane=_plane(mm_per_px=0.2, residual_mm=0.183))
    )
    healthy_text = window.status_text()
    assert "0.183" in healthy_text
    assert window.status_is_warning() is False

    window.update_frame_result(
        _frame_result(rectified=_frame(900, 600), plane=_plane(mm_per_px=0.2, residual_mm=2.400))
    )
    degraded_text = window.status_text()

    assert "2.400" in degraded_text
    assert degraded_text != healthy_text, (
        "el campo tiene que cambiar con el residual medido, no con la escala configurada"
    )
    assert window.status_is_warning() is True, (
        f"un residual de 2.4 mm esta por encima de {RESIDUAL_WARN_MM} mm y tiene que avisar"
    )


def test_a_healthy_residual_just_below_the_threshold_does_not_warn(qtbot):
    """El otro lado del umbral: "avisa cuando corresponde" vs. "avisa
    siempre", la misma distincion que la Ruling 1 de la Task 18 ya exigio
    para el aviso de calibracion."""
    window = MainWindow(intrinsics=_intrinsics())
    qtbot.addWidget(window)

    window.update_frame_result(
        _frame_result(
            rectified=_frame(900, 600),
            plane=_plane(residual_mm=RESIDUAL_WARN_MM - 0.001),
        )
    )
    assert window.status_is_warning() is False

    window.update_frame_result(
        _frame_result(rectified=_frame(900, 600), plane=_plane(residual_mm=RESIDUAL_WARN_MM))
    )
    assert window.status_is_warning() is True


# ---------------------------------------------------------------------------
# Fix 1 (demo de LinkedIn) -- intrinsecos placeholder en la barra de estado.
# ---------------------------------------------------------------------------


def _placeholder_intrinsics() -> Intrinsics:
    """Reproduce ``config/intrinsics.json`` tal cual lo dejo el autor para
    poder arrancar la app sin el tablero de ajedrez: matriz identidad,
    distorsion cero, RMS 0.0 EXACTO."""
    return Intrinsics(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros(5),
        rms_px=0.0,
        image_size=(1920, 1080),
        zoom_note="PROVISIONAL - NOT a real calibration.",
    )


def test_status_bar_warns_about_placeholder_intrinsics_instead_of_showing_them_as_healthy(qtbot):
    """Fix de la demo de LinkedIn: intrinsecos placeholder (RMS 0.0 exacto)
    pasaban ``is_acceptable`` igual que la MEJOR calibracion real posible
    (``0.0 < INTRINSICS_MAX_RMS_PX``), asi que la barra de estado mostraba
    "Intrinsics RMS 0.00px" con el check VERDE -- el unico caso sin
    calibracion real mostrandose como el mejor posible.

    La implementacion equivocada que esta prueba RECHAZA: la anterior
    (solo ``is_acceptable``), que da ``status_is_warning() is False`` para
    el placeholder -- exactamente lo opuesto de lo que esta prueba exige.
    No alcanza con que aparezca ALGUNA palabra de aviso: se fija que el
    texto diga explicitamente que no es una calibracion real.
    """
    window = MainWindow(intrinsics=_placeholder_intrinsics())
    qtbot.addWidget(window)
    # Marcadores sanos (4/4), para aislar que el aviso viene de los
    # intrinsecos y no del estado por defecto "calibration blocked"
    # (``_plane is None``) con el que arranca cualquier ventana nueva.
    window.update_frame_result(_frame_result(rectified=_frame(900, 600), plane=_plane()))

    text = window.status_text()
    assert window.status_is_warning() is True
    assert "placeholder" in text.lower() or "not a real calibration" in text.lower(), text

    window.close()


def test_status_bar_does_not_warn_about_a_genuinely_good_calibration_with_a_low_rms(qtbot):
    """La otra mitad: una calibracion real con RMS 0.21 px (buena, bajo el
    umbral de 0.5) NO es un placeholder y tiene que seguir en el estilo
    SANO. La implementacion equivocada que esta prueba RECHAZA: una que
    marcara CUALQUIER RMS bajo (no solo 0.0 exacto) como placeholder --
    esa version tambien dispararia aqui, con 0.21, y esta prueba la
    atrapa.
    """
    good_intrinsics = _intrinsics(rms_px=0.21)
    window = MainWindow(intrinsics=good_intrinsics)
    qtbot.addWidget(window)
    window.update_frame_result(_frame_result(rectified=_frame(900, 600), plane=_plane()))

    text = window.status_text()
    assert window.status_is_warning() is False
    assert "0.21" in text
    assert "placeholder" not in text.lower()

    window.close()


def test_measurement_stays_enabled_with_placeholder_intrinsics(qtbot):
    """Decision explicita del encargo: el placeholder avisa, no bloquea.
    Bloquear la medicion pararia la grabacion en vivo del demo por una
    condicion que el sistema ya puede medir y etiquetar -- la respuesta
    honesta es medir con la salvedad visible, no negarse a medir. La
    implementacion equivocada que esta prueba RECHAZA: una que metiera
    ``is_placeholder`` dentro de ``measurement_enabled`` (o de
    ``blocked_reason``) -- esa version bloquearia la medicion justo
    cuando el autor esta grabando con intrinsecos placeholder a proposito.
    """
    window = MainWindow(intrinsics=_placeholder_intrinsics())
    qtbot.addWidget(window)

    assert window.measurement_enabled is True
    assert window.blocked_reason() == ""

    window.close()


def test_the_pip_sits_at_the_bottom_centre_so_it_covers_no_marker():
    """Los 4 ArUcos viven en las ESQUINAS del lienzo por construccion, asi
    que un PiP anclado a una esquina tapa siempre uno -- en la grabacion
    tapaba el #1. El centro inferior es la unica franja del borde sin
    marcador.

    Rechaza el anclaje anterior (abajo a la DERECHA): con un widget de
    1000 px y un PiP de 200, aquel daba x = 788 y este da x = 400. La
    coordenada vertical no cambia: pegado abajo es correcto, lo que estaba
    mal era el lado.
    """
    x, y = _pip_origin_px(widget_width_px=1000, widget_height_px=700, pip_width_px=200, pip_height_px=120)

    assert x == 400, f"el PiP no quedo centrado en horizontal: x={x}"
    assert y == 700 - 120 - _PIP_MARGIN_PX


def test_the_pip_origin_never_goes_negative_on_a_widget_narrower_than_itself():
    """Con el widget mas estrecho que el propio PiP, centrar daria una
    coordenada NEGATIVA y Qt recortaria por la izquierda.

    Rechaza una implementacion que centre con una resta pelada sin
    acotar en 0.
    """
    x, y = _pip_origin_px(widget_width_px=100, widget_height_px=80, pip_width_px=200, pip_height_px=120)

    assert x == 0
    assert y == 0

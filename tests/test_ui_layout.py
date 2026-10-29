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

from codex_caliper.aruco_tools import CalibrationState
from codex_caliper.camera import CameraThread
from codex_caliper.codex import CodexVerdict
from codex_caliper.homography import PlaneCalibration
from codex_caliper.intrinsics import Intrinsics
from codex_caliper.pipeline import MARKER_COUNT, BananaResult, FrameResult
from codex_caliper.aggregation import TrackAggregate
from codex_caliper.ui.main_window import MainWindow
from codex_caliper.ui.overlay import text_anchor_for
from codex_caliper.ui.side_panel import SIDE_PANEL_WIDTH_PX, SidePanel
from codex_caliper.ui.video_view import PIP_MIN_WIDTH_PX, PIP_SCALE, VideoView


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
    *, state: CalibrationState = CalibrationState.OK, mm_per_px: float = 0.2, residual_mm: float = 0.18
) -> PlaneCalibration:
    return PlaneCalibration(H=np.eye(3), residual_mm=residual_mm, state=state, mm_per_px=mm_per_px)


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
        volume_mm3=1.0e5,
    )


def _banana_result(*, track_id: int = 1, locked: bool = True, passes: bool = True) -> BananaResult:
    mask = np.zeros((200, 300), dtype=bool)
    mask[50:150, 50:250] = True
    return BananaResult(
        track_id=track_id,
        aggregate=_aggregate(track_id=track_id, locked=locked),
        verdict=CodexVerdict(passes=passes, length_ok=passes, caliper_ok=passes, commercial_grade=38),
        estimated_weight_g=145.0,
        mask=mask,
    )


def _frame_result(
    *,
    rectified: np.ndarray | None,
    raw: np.ndarray | None = None,
    plane: PlaneCalibration | None = None,
    results: list[BananaResult] | None = None,
    warnings: list[str] | None = None,
) -> FrameResult:
    if raw is None:
        raw = _frame(640, 480, value=200)
    return FrameResult(
        rectified=rectified,
        raw_undistorted=raw,
        plane=plane,
        results=results or [],
        warnings=warnings or [],
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
        return _frame_result(rectified=frame, raw=frame)


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


def test_pip_is_rendered_small_but_still_legible(qtbot):
    """Ruling 1: ``pip.width() < main.width() // 3`` por si solo lo satisface
    un PiP de un pixel. Se agrega un piso absoluto de legibilidad: el PiP
    tiene que ser chico Y seguir sirviendo como miniatura en un video real."""
    view = VideoView()
    qtbot.addWidget(view)

    raw = _frame(640, 480)
    rectified = _frame(900, 600)
    view.set_frame(rectified, raw)

    assert view.pip_pixmap.width() < view.main_pixmap.width() // 3
    assert view.pip_pixmap.width() >= 100, "el PiP no puede degenerar a algo ilegible en video"


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

    window.update_frame_result(_frame_result(rectified=_frame(900, 600), plane=_plane()))
    assert window.status_is_warning() is False


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
    panel.update_results([_banana_result(track_id=1)])
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
    window = MainWindow(intrinsics=_intrinsics(image_size=(1920, 1080)), pipeline=RecordingPipeline())
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
    assert pipeline.threads == [], "ningun cuadro debe llegar al pipeline mientras la medicion esta bloqueada"

    window.close()

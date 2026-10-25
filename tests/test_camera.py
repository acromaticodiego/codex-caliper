"""Tests del hilo de captura de camara: emision de frames via senal Qt, copia
defensiva del buffer, parada acotada, y la solicitud de MJPEG a 1080p.

Todo pasa por un ``cv2.VideoCapture`` falso inyectado -- estos tests no abren
ninguna camara real y deben pasar igual con o sin una conectada a la maquina.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import cv2
from PySide6.QtCore import Qt

from codex_caliper import camera
from codex_caliper.camera import CameraThread, open_capture


class FakeCapture:
    """Sustituto de ``cv2.VideoCapture``: registra las propiedades pedidas y
    devuelve frames de una lista fija, ciclando. ``granted_width/height``
    simulan que el driver concede una resolucion distinta de la solicitada,
    tal como hace un dispositivo real cuando no soporta el modo pedido."""

    def __init__(self, frames, granted_width=1920, granted_height=1080, opened=True):
        self._frames = frames
        self._next = 0
        self.properties: dict[int, float] = {}
        self.granted_width = granted_width
        self.granted_height = granted_height
        self.released = False
        self._opened = opened

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop_id, value) -> bool:
        self.properties[prop_id] = value
        return True

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.granted_width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.granted_height)
        return float(self.properties.get(prop_id, 0))

    def read(self):
        if not self._frames:
            return False, None
        # Un ritmo de lectura: una camara real bloquea en .read() hasta que
        # el siguiente frame esta listo (a 30 fps, ~33 ms). Sin esta pausa el
        # hilo falso emite en un bucle tan apretado que satura la entrega en
        # cola de Qt hacia el slot interno de qtbot.waitSignal -- un
        # artefacto de la prueba, no del comportamiento real de una camara.
        time.sleep(0.01)
        frame = self._frames[self._next % len(self._frames)]
        self._next += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class MutatingFakeCapture:
    """Reproduce el reuso de buffer interno de OpenCV: cada ``.read()``
    reescribe EL MISMO arreglo en vez de entregar uno nuevo, y devuelve la
    referencia a ese arreglo compartido. Si ``camera.py`` emitiera esa
    referencia cruda en vez de una copia, todos los frames capturados en el
    hilo receptor acabarian mostrando el ultimo valor escrito -- un frame
    rasgado, sin ninguna excepcion que lo delate."""

    def __init__(self, width=8, height=6):
        self._buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self._calls = 0
        self.properties: dict[int, float] = {}
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, prop_id, value) -> bool:
        self.properties[prop_id] = value
        return True

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._buffer.shape[1])
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._buffer.shape[0])
        return float(self.properties.get(prop_id, 0))

    def read(self):
        time.sleep(0.01)  # ver nota de ritmo en FakeCapture.read
        self._calls += 1
        self._buffer[:] = self._calls % 256  # reescribe el buffer compartido
        return True, self._buffer

    def release(self) -> None:
        self.released = True


def _make_frame(value: int, width=8, height=6) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_emits_frames_from_a_fake_capture(qtbot, monkeypatch):
    """La senal frame_ready debe llevar el frame que el capture falso
    produjo, no uno vacio o reconstruido -- una prueba que solo cuenta
    emisiones no veria un frame corrupto o sustituido."""
    expected_frame = _make_frame(42)
    fake_capture = FakeCapture(frames=[expected_frame])
    monkeypatch.setattr(camera, "open_capture", lambda *a, **kw: fake_capture)

    thread = CameraThread(0)
    try:
        with qtbot.waitSignal(thread.frame_ready, timeout=2000) as blocker:
            thread.start()
        (received,) = blocker.args
        np.testing.assert_array_equal(received, expected_frame)
    finally:
        thread.stop()


def test_stop_terminates_the_loop_and_releases_the_capture(qtbot, monkeypatch):
    """stop() debe: (1) dejar el QThread realmente terminado -- no solo
    liberar el capture mientras el bucle sigue vivo --, (2) liberar el
    capture, y (3) no permitir que llegue ningun frame despues de que stop()
    retorna."""
    fake_capture = FakeCapture(frames=[_make_frame(1), _make_frame(2)])
    monkeypatch.setattr(camera, "open_capture", lambda *a, **kw: fake_capture)

    thread = CameraThread(0)
    received = []
    # Conexion directa: el slot corre en el propio hilo de captura, en el
    # mismo orden en que se emite -- asi "recibido hasta ahora" es exacto.
    thread.frame_ready.connect(lambda f: received.append(f), Qt.ConnectionType.DirectConnection)

    with qtbot.waitSignal(thread.frame_ready, timeout=2000):
        thread.start()

    thread.stop()

    assert thread.isFinished() is True, "stop() debe esperar a que el bucle termine, no solo pedirlo"
    assert fake_capture.released is True

    frames_after_stop = len(received)
    # El hilo ya terminó (aserción anterior); si igual llegara un frame mas
    # tarde, seria la prueba de que el bucle seguia vivo tras stop().
    qtbot.wait(50)
    assert len(received) == frames_after_stop, "no debe emitirse ningun frame despues de stop()"


def test_requests_mjpeg_at_1080p(monkeypatch):
    """open_capture debe pedir MJPEG Y 1920x1080 -- a esta resolucion sobre
    USB 2.0, MJPEG es la unica opcion con ancho de banda para 30 fps."""
    fake_capture = FakeCapture(frames=[])
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **kw: fake_capture)

    result = open_capture(0, width=1920, height=1080, fourcc="MJPG", backend=cv2.CAP_DSHOW)

    mjpg_code = cv2.VideoWriter_fourcc(*"MJPG")
    assert result.properties[cv2.CAP_PROP_FOURCC] == mjpg_code
    assert result.properties[cv2.CAP_PROP_FRAME_WIDTH] == 1920
    assert result.properties[cv2.CAP_PROP_FRAME_HEIGHT] == 1080


def test_actual_size_reports_what_the_device_granted_not_what_was_asked(qtbot, monkeypatch):
    """El driver puede no conceder lo pedido; actual_size debe leer lo que la
    camara entrega de verdad, no repetir los parametros del constructor."""
    fake_capture = FakeCapture(frames=[_make_frame(7)], granted_width=1280, granted_height=720)
    monkeypatch.setattr(camera, "open_capture", lambda *a, **kw: fake_capture)

    thread = CameraThread(0, width=1920, height=1080)
    try:
        with qtbot.waitSignal(thread.frame_ready, timeout=2000):
            thread.start()
        assert thread.actual_size == (1280, 720)
    finally:
        thread.stop()


def test_frame_ready_emits_a_copy_not_the_shared_capture_buffer(qtbot, monkeypatch):
    """Ruling 1: cv2 puede reusar el mismo buffer entre lecturas. Si camera.py
    emitiera esa referencia cruda, todos los frames capturados acabarian
    mostrando el ultimo valor escrito por el capture, porque todos apuntan al
    mismo arreglo. Con copia, cada frame conserva el valor que tenia en su
    propio instante de emision, incluso despues de que el buffer siga
    mutando."""
    fake_capture = MutatingFakeCapture()
    monkeypatch.setattr(camera, "open_capture", lambda *a, **kw: fake_capture)

    thread = CameraThread(0)
    captured = []
    thread.frame_ready.connect(lambda f: captured.append(f), Qt.ConnectionType.DirectConnection)

    with qtbot.waitSignal(thread.frame_ready, timeout=2000):
        thread.start()
    qtbot.waitUntil(lambda: len(captured) >= 5, timeout=2000)
    thread.stop()

    # Inspeccionamos DESPUES de que el buffer compartido mutara muchas veces
    # mas. Si los frames capturados fueran la misma referencia, todos
    # mostrarian el ultimo valor escrito (un solo valor distinto en el set).
    values = {int(f[0, 0, 0]) for f in captured}
    assert len(values) >= 2, "los frames emitidos deben ser copias independientes del buffer del capture"


def test_stop_before_start_does_not_hang_or_raise():
    """stop() debe ser seguro de llamar incluso si el hilo nunca arranco."""
    thread = CameraThread(0)
    thread.stop()
    assert thread.isFinished() is False  # nunca corrio, no hay nada que terminar


def test_stop_returns_promptly_even_if_read_is_blocked(qtbot, monkeypatch):
    """Ruling 2: si el hilo esta parado dentro de un cap.read() que tarda en
    volver -- la camara se desenchufo a media lectura --, stop() debe
    retornar de todos modos, acotado, en vez de esperar sin limite a un
    hilo que no puede revisar la bandera de parada mientras esta bloqueado.

    Sincronizacion: ``isRunning()`` solo dice que el hilo del SO arranco, no
    que ``run()`` ya llego a la lectura bloqueante -- entre medio estan
    ``open_capture``, ``isOpened`` y la lectura de ``actual_size``. Si
    llamaramos a stop() basados solo en isRunning(), podria llegar antes de
    que el bucle revise la bandera siquiera una vez, y entonces read() jamas
    se ejecutaria: la prueba pasaria igual, pero sin haber probado nada.
    Por eso se sincroniza con un Event que ``read()`` marca justo al entrar,
    y solo se mide stop() una vez confirmado que la lectura bloqueante ya
    esta en curso."""

    class SlowFirstReadCapture(FakeCapture):
        """La primera lectura tarda mas que el timeout de stop() (simula el
        bloqueo); las siguientes fallan rapido, asi el hilo termina por su
        cuenta via la politica de desconexion y no queda nada corriendo de
        fondo al salir del test."""

        def __init__(self, *a, started_reading, **kw):
            super().__init__(*a, **kw)
            self._first_call = True
            self._started_reading = started_reading

        def read(self):
            if self._first_call:
                self._first_call = False
                self._started_reading.set()
                time.sleep(3.0)  # mas largo que _STOP_WAIT_TIMEOUT_MS (2s)
            return False, None

    started_reading = threading.Event()
    fake_capture = SlowFirstReadCapture(frames=[], started_reading=started_reading)
    monkeypatch.setattr(camera, "open_capture", lambda *a, **kw: fake_capture)

    thread = CameraThread(0)
    thread.start()
    assert started_reading.wait(timeout=2.0), "la lectura bloqueante nunca empezo"

    started = time.monotonic()
    thread.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 2.5, "stop() debe retornar acotado aunque read() siga bloqueado"

    # Limpieza: dejar que el hilo salga solo una vez pasa el bloqueo
    # simulado, para no dejar nada corriendo de fondo al cerrar el test.
    thread.wait(5000)


def test_capture_lost_is_emitted_when_reads_fail_repeatedly(qtbot, monkeypatch):
    """Politica de desconexion: si la lectura falla de forma persistente (la
    camara se desconecto), el hilo no debe quedarse reintentando para
    siempre en silencio ni explotar con una excepcion -- debe avisar via la
    senal capture_lost y terminar el bucle por su cuenta."""

    class AlwaysFailingCapture(FakeCapture):
        def read(self):
            return False, None

    fake_capture = AlwaysFailingCapture(frames=[])
    monkeypatch.setattr(camera, "open_capture", lambda *a, **kw: fake_capture)

    thread = CameraThread(0)
    with qtbot.waitSignal(thread.capture_lost, timeout=5000) as blocker:
        thread.start()

    qtbot.waitUntil(lambda: thread.isFinished(), timeout=2000)
    assert fake_capture.released is True

    # Finding 2 (ronda de revision 1): capture_lost es texto de runtime de
    # cara al usuario -- la razon misma por la que existe es para mostrarse
    # en la interfaz -- asi que debe estar en ingles, como el resto del
    # proyecto (ver IntrinsicsMismatch en intrinsics.py). Se verifica
    # contenido, no solo que algo se emitio.
    (message,) = blocker.args
    assert "stopped responding" in message
    assert "camara" not in message.lower()


def test_stop_called_before_the_thread_begins_running_never_opens_the_camera(qtbot, monkeypatch):
    """Finding 1 (ronda de revision 1): la carrera real era que stop()
    llegara antes de que run() ejecutara su primera linea, y run() hacia
    self._stop_requested.clear() ahi mismo, borrando la peticion en
    silencio -- el bucle arrancaba igual, con la camara abierta, y stop()
    ya habia retornado creyendo que funciono.

    No se puede forzar de forma determinista que stop() llegue DURANTE esa
    ventana especifica sin instrumentar run() (eso convertiria el test en
    una prueba del instrumento, no del codigo). En vez de eso, se fija el
    caso limite que la implementacion correcta debe tratar exactamente
    igual que esa ventana: pedir stop() ANTES de llamar a start() siquiera.
    Esto garantiza, sin ninguna carrera, que la bandera ya esta puesta
    cuando el hilo del SO exista y run() se ejecute -- el mismo estado que
    "stop() llego un microsegundo antes de la primera linea de run()". Si
    run() alguna vez volviera a limpiar la bandera ahi, este test lo
    atraparia exactamente igual que atraparia la carrera real."""
    fake_capture = FakeCapture(frames=[_make_frame(9)])
    monkeypatch.setattr(camera, "open_capture", lambda *a, **kw: fake_capture)

    thread = CameraThread(0)
    received = []
    thread.frame_ready.connect(lambda f: received.append(f), Qt.ConnectionType.DirectConnection)

    thread.stop()  # pedir parar ANTES de arrancar -- fuerza la peor ventana posible
    thread.start()

    qtbot.waitUntil(lambda: thread.isFinished(), timeout=2000)

    assert received == [], "no debe haber capturado ningun frame si stop() llego antes de start()"
    assert fake_capture.released is True


def test_capture_lost_is_emitted_when_the_capture_never_opens(qtbot, monkeypatch):
    """Finding 4: el unico otro sitio que emite capture_lost (ademas de la
    cascada de lecturas fallidas) es cuando isOpened() devuelve False justo
    despues de abrir -- y ningun test lo ejercia. FakeCapture ya tenia el
    parametro opened=False sin usar."""
    fake_capture = FakeCapture(frames=[_make_frame(1)], opened=False)
    monkeypatch.setattr(camera, "open_capture", lambda *a, **kw: fake_capture)

    thread = CameraThread(0)
    received = []
    thread.frame_ready.connect(lambda f: received.append(f), Qt.ConnectionType.DirectConnection)

    with qtbot.waitSignal(thread.capture_lost, timeout=2000) as blocker:
        thread.start()

    qtbot.waitUntil(lambda: thread.isFinished(), timeout=2000)
    assert received == [], "una camara que nunca abrio no debe producir ningun frame"
    assert fake_capture.released is True

    # Finding 2: mismo chequeo de idioma que en la cascada de reintentos.
    (message,) = blocker.args
    assert "could not be opened" in message
    assert "camara" not in message.lower()


def test_capture_lost_is_emitted_when_read_raises_a_driver_error(qtbot, monkeypatch):
    """Finding 3: si cv2.VideoCapture.read() LANZA (un error de driver, no
    simplemente un frame perdido que devuelve ok=False), el hilo no debe
    morir en silencio sin que la interfaz se entere -- la unica cosa que la
    Ruling 2 prohibe explicitamente. Debe convertirse en capture_lost, igual
    que la desconexion detectada por reintentos fallidos."""

    class RaisingCapture(FakeCapture):
        def read(self):
            raise cv2.error("simulated driver failure")

    fake_capture = RaisingCapture(frames=[_make_frame(1)])
    monkeypatch.setattr(camera, "open_capture", lambda *a, **kw: fake_capture)

    thread = CameraThread(0)
    with qtbot.waitSignal(thread.capture_lost, timeout=2000):
        thread.start()

    qtbot.waitUntil(lambda: thread.isFinished(), timeout=2000)
    assert fake_capture.released is True

"""Tests del punto de entrada (Task 20): los caminos de fallo de arranque.

Lo que esta bateria protege es una sola regla del brief: **fallar con un
mensaje claro y codigo 1, nunca con un stack trace**. Cada test de aqui es
un error de puesta en marcha que un operador real va a cometer -- config
sin escribir, layout sin medir, intrinsecos sin calibrar, camara en el
indice equivocado, ``ultralytics`` sin instalar -- y comprueba las DOS
mitades de esa regla: el codigo de salida Y el texto que el operador lee.

Dos decisiones de diseno de estas pruebas, ambas sobre determinismo:

**La camara se inyecta, no se prueba.** El test del brief
(``main(["--camera", "99"]) == 1``) depende del hardware de la maquina: en
esta, sin camara, pasa por casualidad (cualquier indice falla), y en una
maquina con una camara en el indice 99 podria no fallar en absoluto --
pasaria "verde" sin ejercitar nada. ``open_capture`` esta aislado en
``camera.py`` precisamente "para poder sustituirlo en tests" (Task 16), asi
que se inyecta un capture falso y el resultado deja de depender del
hardware presente. Por el mismo motivo se afirma el MENSAJE y no solo el
codigo de salida: una aserciona solo sobre ``== 1`` la satisface cualquier
implementacion que devuelva 1 por el motivo equivocado.

**La ausencia de ``ultralytics`` se fuerza, no se asume.** Aqui no esta
instalado (son ~2.5 GB con ``torch``), asi que el test pasaria igual sin
hacer nada; pero en la maquina del autor, despues de instalar los pesos
para la verificacion en hardware, dejaria de ejercitar ese camino en
silencio. Poner ``None`` en ``sys.modules["ultralytics"]`` hace que
``from ultralytics import YOLO`` lance ``ImportError`` haya o no un
``ultralytics`` instalado -- el mismo resultado en las dos maquinas.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

from codex_caliper.__main__ import main, probe_camera
from codex_caliper.config import AppConfig
from codex_caliper.intrinsics import Intrinsics


class FakeCapture:
    """Sustituto minimo de ``cv2.VideoCapture`` para el sondeo de camara.

    Solo necesita responder ``isOpened()`` y ``release()``: el punto de
    entrada no lee ni un cuadro de este capture -- solo comprueba que el
    dispositivo abre y lo libera para que ``CameraThread`` sea el unico
    dueno del dispositivo real (ver ``probe_camera``).
    """

    def __init__(self, opened: bool = True) -> None:
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self.released = True


def _write_valid_config(tmp_path, *, with_layout: bool = True):
    """Escribe una config valida del montaje real y devuelve su ruta.

    ``with_layout=False`` produce el error de configuracion mas probable de
    la primera puesta en marcha: una config escrita a mano sin haber medido
    todavia las posiciones de los marcadores.
    """
    layout = {1: (0.0, 0.0), 2: (215.0, 0.0), 3: (215.0, 118.0), 4: (0.0, 118.0)}
    cfg = AppConfig(
        marker_thickness_mm=2.5,
        marker_layout=layout if with_layout else {},
    )
    path = tmp_path / "app.json"
    cfg.save(path)
    return path


def _write_valid_intrinsics(tmp_path):
    """Escribe unos intrinsecos aceptables (RMS bajo el umbral) y devuelve su ruta."""
    intr = Intrinsics(
        camera_matrix=np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]]),
        dist_coeffs=np.zeros(5),
        rms_px=0.21,
        image_size=(1920, 1080),
    )
    path = tmp_path / "intrinsics.json"
    intr.save(path)
    return path


def test_main_reports_a_clear_error_when_config_is_missing(tmp_path, capsys):
    """El test del brief, mas dos aserciones que el brief no pedia: que la
    ruta que se intento aparezca en el mensaje (sin ella el operador no
    sabe que archivo escribir) y que stdout quede limpio -- los errores van
    a stderr."""
    missing = tmp_path / "nope.json"
    assert main(["--config", str(missing)]) == 1
    captured = capsys.readouterr()
    assert "config" in captured.err.lower()
    assert str(missing) in captured.err
    # Distingue este fallo del de "la config existe pero no se puede leer":
    # son acciones distintas para el operador (escribir el archivo vs.
    # arreglarlo), y un mensaje que vale para los dos no sirve para ninguno.
    assert "not found" in captured.err
    assert captured.out == ""


def test_main_reports_a_clear_error_when_the_config_cannot_be_parsed(tmp_path, capsys):
    """Una config con JSON roto (o sin ``marker_thickness_mm``, que no tiene
    default porque hay que MEDIRLO) no puede llegar como traceback.

    Las dos ultimas aserciones son las que hacen que este test DISCRIMINE
    (Finding 3 de la ronda 1 de revision). Con solo ``"config" in err`` y la
    ruta, una implementacion que emitiera el mensaje de ARCHIVO AUSENTE para
    un archivo que si existe pasaba las dos: el operador leeria "measure the
    rig and write it first" sobre un archivo que ya escribio, y se pondria a
    buscar un problema que no existe. Se afirma el fragmento propio de este
    camino Y la ausencia del fragmento del otro -- una sola de las dos deja
    pasar la mitad de la confusion.
    """
    broken = tmp_path / "app.json"
    broken.write_text("{not json", encoding="utf-8")
    assert main(["--config", str(broken)]) == 1
    err = capsys.readouterr().err
    assert "config" in err.lower()
    assert str(broken) in err
    assert "could not be read" in err
    assert "not found" not in err


def test_main_reports_a_clear_error_when_the_marker_layout_is_missing(tmp_path, capsys):
    """Sin ``marker_layout`` el lienzo no se puede dimensionar.

    ``canvas_size_for`` llama a ``min()`` sobre las posiciones declaradas, y
    con el diccionario vacio eso es un ``ValueError: min() arg is an empty
    sequence`` -- un traceback crudo sobre el error de configuracion mas
    probable de la primera puesta en marcha.
    """
    cfg_path = _write_valid_config(tmp_path, with_layout=False)
    intrinsics_path = _write_valid_intrinsics(tmp_path)
    exit_code = main(
        ["--config", str(cfg_path), "--intrinsics", str(intrinsics_path)],
        open_capture=lambda index, **kwargs: FakeCapture(),
    )
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "marker_layout" in err
    assert "Traceback" not in err


def test_main_reports_a_clear_error_when_the_intrinsics_file_is_missing(tmp_path, capsys):
    """``open_capture`` se inyecta aunque este camino no llegue a la camara.

    Finding 4 de la ronda 1 de revision: sin la inyeccion, este test era
    determinista SOLO por el orden de arranque actual (intrinsecos antes que
    camara). Si ese orden regresara, el test abriria el dispositivo real
    durante la bateria -- pasaria en una maquina con camara en el indice 0 y
    fallaria en una sin ella, que es exactamente lo que "los tests no
    requieren camara" existe para evitar. Con el capture falso, el resultado
    no depende del orden ni del hardware.
    """
    cfg_path = _write_valid_config(tmp_path)
    missing = tmp_path / "no-intrinsics.json"
    exit_code = main(
        ["--config", str(cfg_path), "--intrinsics", str(missing)],
        open_capture=lambda index, **kwargs: FakeCapture(),
    )
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "intrinsic" in err.lower()
    assert str(missing) in err
    assert "not found" in err


def test_main_reports_a_clear_error_when_the_intrinsics_file_is_incomplete(tmp_path, capsys):
    """Unos intrinsecos a los que les falta un campo tampoco son un traceback.

    ``Intrinsics.load`` INDEXA el JSON (``data["camera_matrix"]``), asi que
    un archivo incompleto lanza ``KeyError`` -- que no es ni ``ValueError``
    ni ``TypeError``, los tipos que basta enumerar para ``AppConfig.load``.
    Sin ``KeyError`` en la tupla de ``_load_intrinsics`` este camino llega
    al operador como traceback.
    """
    cfg_path = _write_valid_config(tmp_path)
    incomplete = tmp_path / "intrinsics.json"
    incomplete.write_text('{"rms_px": 0.21}', encoding="utf-8")
    exit_code = main(
        ["--config", str(cfg_path), "--intrinsics", str(incomplete)],
        open_capture=lambda index, **kwargs: FakeCapture(),
    )
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "intrinsic" in err.lower()
    assert "could not be read" in err
    assert "Traceback" not in err


def test_main_reports_a_clear_error_when_the_camera_cannot_be_opened(tmp_path, capsys):
    """La version determinista del test del brief (ver docstring del modulo).

    Afirma el MENSAJE, no solo el codigo de salida: tiene que nombrar el
    indice que se intento (es el dato con el que el operador corrige la
    invocacion) y tiene que salir por stderr. Y afirma que el capture
    fallido se libero: un ``VideoCapture`` que no se suelta deja el
    dispositivo tomado para el siguiente intento.
    """
    cfg_path = _write_valid_config(tmp_path)
    intrinsics_path = _write_valid_intrinsics(tmp_path)
    closed = FakeCapture(opened=False)

    exit_code = main(
        [
            "--config",
            str(cfg_path),
            "--intrinsics",
            str(intrinsics_path),
            "--camera",
            "99",
        ],
        open_capture=lambda index, **kwargs: closed,
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "camera 99" in captured.err.lower()
    assert captured.out == ""
    assert closed.released


def test_main_reports_a_clear_error_when_the_camera_driver_raises(tmp_path, capsys):
    """``cv2.VideoCapture`` puede lanzar ``cv2.error`` en vez de devolver un
    capture cerrado (backend ausente, indice invalido para el driver). El
    mismo mensaje claro, no un traceback de OpenCV."""
    cfg_path = _write_valid_config(tmp_path)
    intrinsics_path = _write_valid_intrinsics(tmp_path)

    def raising_open_capture(index, **kwargs):
        raise cv2.error("fake driver failure")

    exit_code = main(
        ["--config", str(cfg_path), "--intrinsics", str(intrinsics_path), "--camera", "3"],
        open_capture=raising_open_capture,
    )
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "camera 3" in err.lower()
    assert "Traceback" not in err


def test_main_reports_a_clear_error_when_ultralytics_is_not_installed(
    tmp_path, capsys, monkeypatch
):
    """El fallo mas probable de la primera ejecucion en una maquina limpia.

    El import de ``ultralytics`` es PEREZOSO (vive dentro de la rama de
    ``BananaDetector`` que construye el modelo real), asi que el
    ``ImportError`` no aparece al importar el paquete sino al arrancar el
    sistema -- y sin este camino llega al operador como un traceback. Se
    fuerza con ``sys.modules`` para que el resultado no dependa de si la
    maquina tiene o no ``ultralytics`` instalado.
    """
    cfg_path = _write_valid_config(tmp_path)
    intrinsics_path = _write_valid_intrinsics(tmp_path)
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    exit_code = main(
        ["--config", str(cfg_path), "--intrinsics", str(intrinsics_path)],
        open_capture=lambda index, **kwargs: FakeCapture(),
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "ultralytics" in err.lower()
    assert "pip install" in err.lower()
    assert "Traceback" not in err


def test_probe_camera_releases_the_capture_it_opened(tmp_path):
    """El sondeo libera el dispositivo tambien cuando abre BIEN.

    El punto de entrada abre la camara una vez solo para poder fallar con
    codigo 1 antes de levantar la ventana; quien lee cuadros despues es
    ``CameraThread``, que la abre por su cuenta. En Windows/DirectShow un
    segundo ``VideoCapture`` sobre un dispositivo ya tomado por este mismo
    proceso falla, asi que no liberar aqui romperia el arranque en el rig
    real -- y ningun test que solo mire el camino de error lo detectaria.
    """
    opened = FakeCapture(opened=True)
    error = probe_camera(0, open_capture=lambda index, **kwargs: opened)
    assert error is None
    assert opened.released


def test_probe_camera_reports_the_index_it_failed_to_open():
    closed = FakeCapture(opened=False)
    error = probe_camera(7, open_capture=lambda index, **kwargs: closed)
    assert error is not None
    assert "camera 7" in error.lower()
    assert closed.released

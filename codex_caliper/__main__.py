"""Punto de entrada del sistema (Task 20): ``python -m codex_caliper``.

Es la unica pieza del proyecto que habla con el operador antes de que haya
una ventana donde mostrar nada. Por eso su contrato no es "arrancar", es
**arrancar o explicar por que no**, con codigo de salida 1 y una sola linea
en stderr -- nunca un traceback. Un traceback en consola es, literalmente,
el anti-objetivo que el spec S4 nombra ("que la interfaz parezca un
script"), y en este sistema ademas llega en el peor momento posible: el
operador ya monto la camara, coloco los marcadores y puso la fruta.

Orden de arranque, y por que es ese
-----------------------------------
    config -> layout -> intrinsecos -> camara -> YOLO -> pipeline -> ventana

De lo barato y seguro a lo caro y fragil. Las dos ultimas comprobaciones
son las unicas que tardan: abrir un dispositivo USB y cargar los pesos de
YOLO (que ``ultralytics`` puede incluso descargar la primera vez). Si una
config sin medir o unos intrinsecos sin calibrar se detectaran DESPUES de
eso, el operador esperaria varios segundos para leer un error que se podia
dar al instante.

El sondeo de la camara esta ANTES de construir el detector por una razon
adicional: es la comprobacion que mas se equivoca en el rig real (indice de
dispositivo), y es la unica de las dos que no depende de ~2.5 GB de
dependencias. Que su mensaje no quede tapado por el de ``ultralytics`` es
tambien lo que hace deterministas las pruebas de esta tarea.

Lo que este modulo NO hace
--------------------------
**No crea la config si falta.** ``AppConfig.marker_thickness_mm`` no tiene
valor por defecto a proposito (Task 1): hay que MEDIRLO apilando los cuatro
marcadores (spec S5.4). Lo mismo vale para ``marker_layout``, que son las
posiciones reales de los marcadores sobre la cartulina. Una config generada
automaticamente rellenaria esos dos campos con numeros que nadie midio, y
el sistema mediria mal en silencio -- exactamente el fallo que este
proyecto persigue en todas sus tareas. Falta la config: se dice donde se
buscaba y se manda al README.

**No ajusta la densidad al arrancar, y nada la ajusta todavia (C3, ronda de
correcciones finales).** ``Pipeline`` acepta un ``density_g_per_cm3``
opcional (``_weight_g`` lo necesita para convertir volumen en peso), pero
``_build_detector``/``main`` construyen el ``Pipeline`` SIN pasarlo, asi
que queda en ``None`` para toda la sesion y ``estimated_weight_g`` nunca
deja de ser ``None`` -- la linea de peso de la spec S8 simplemente no
aparece. ``ValidationView`` SI llama a ``fit_density`` sobre las muestras
reales (``>= 10``, Task 15) para mostrar la densidad ajustada en su propio
resumen, pero eso es SOLO para mostrarla ahi: ningun camino de este
proyecto toma ese resultado y lo entrega de vuelta al ``Pipeline`` en
vivo. Cerrar esto es trabajo nuevo (que pantalla dispara el ajuste, con
que muestras, y cuando se refresca) fuera del alcance de esta ronda -- ver
"Known limits" del README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from codex_caliper import camera
from codex_caliper.config import AppConfig
from codex_caliper.detection import BananaDetector
from codex_caliper.intrinsics import Intrinsics
from codex_caliper.pipeline import Pipeline

#: Rutas por defecto. ``config/intrinsics.json`` es la que fija el spec
#: (S5.2 / S7 etapa 1) para la calibracion intrinseca persistente.
DEFAULT_CONFIG_PATH = Path("config/app.json")
DEFAULT_INTRINSICS_PATH = Path("config/intrinsics.json")

#: Nombre de la bitacora CSV de validacion, junto a la config.
#:
#: ``MainWindow`` sin este argumento cae en un archivo temporal con nombre
#: aleatorio (Task 19) -- correcto para una ventana de pruebas, inservible
#: para el rig: las muestras de validacion son el entregable publicable del
#: proyecto (spec S13, punto 4) y no pueden quedar en ``%TEMP%`` con un
#: nombre distinto en cada arranque. Se deriva de la ruta de la config en
#: vez de anadir una bandera mas: donde vive la config es donde vive el
#: estado de este montaje.
VALIDATION_LOG_NAME = "validation_log.csv"

#: Tamano inicial de la ventana. ``QMainWindow`` sin esto arranca con su
#: tamano minimo, que con el panel lateral de 240 px fijos (spec S8) deja
#: al video un hueco ridiculo en el primer arranque.
_INITIAL_WINDOW_SIZE_PX = (1280, 720)


class _StartupError(Exception):
    """Fallo de puesta en marcha, con el mensaje ya redactado para el operador.

    Existe para que haya UN SOLO punto en el modulo que escribe a stderr y
    devuelve 1 (el ``except`` de ``main``). Cada comprobacion se limita a
    describir el problema en ingles y en una linea; ninguna imprime ni
    decide codigos de salida por su cuenta, asi que no hay forma de anadir
    una comprobacion nueva que se olvide de una de las dos mitades del
    contrato.
    """


def _load_config(path: Path) -> AppConfig:
    """Carga la config del montaje, o lanza ``_StartupError`` explicando como crearla."""
    if not path.exists():
        raise _StartupError(
            f"config file not found: {path}. Measure the rig and write it first "
            "-- see the 'Setting up the rig' section of README.md."
        )
    try:
        return AppConfig.load(path)
    except (OSError, ValueError, TypeError) as exc:
        # Los tres tipos que una config mala puede producir, enumerados en
        # vez de un ``except Exception``: un JSON roto da ``ValueError``
        # (``JSONDecodeError`` hereda de el), una clave que falta o sobra da
        # ``TypeError`` desde el constructor del dataclass, y un archivo
        # ilegible da ``OSError``. Para el operador son el mismo problema
        # ("esta config no sirve, y esto es lo que esta mal") y ninguno
        # puede salir como traceback -- pero un ``AttributeError`` dentro de
        # ``AppConfig.load`` es un BUG, no una config mala, y tiene que
        # verse como bug. Este proyecto ya pago por un except ciego (Task 8:
        # ``np.trapezoid`` sin piso de numpy devolviendo ``None`` para
        # siempre, en silencio).
        raise _StartupError(f"config file {path} could not be read: {exc}") from exc


def _require_measured_layout(cfg: AppConfig, path: Path) -> None:
    """Exige que ``marker_layout`` cubra todos los ids declarados.

    Sin layout no hay lienzo: ``canvas_size_for`` toma ``min()`` sobre las
    posiciones declaradas y con el diccionario vacio eso es un
    ``ValueError: min() arg is an empty sequence`` dentro del constructor
    del ``Pipeline``. Es el error de configuracion mas probable de la
    primera puesta en marcha (la config se escribe antes de medir las
    posiciones) y el que peor se lee como traceback.
    """
    missing = [marker_id for marker_id in cfg.marker_ids if marker_id not in cfg.marker_layout]
    if not cfg.marker_layout or missing:
        raise _StartupError(
            f"config file {path} has no marker_layout entry for marker ids "
            f"{missing or cfg.marker_ids}. "
            "Measure each marker centre on the canvas, in millimetres, and add it "
            "-- see the 'Setting up the rig' section of README.md."
        )


def _load_intrinsics(path: Path) -> Intrinsics:
    """Carga los intrinsecos persistidos, o lanza ``_StartupError``.

    No se rechaza una calibracion con RMS por encima del umbral: cargarla
    es legitimo (puede ser lo unico que hay) y la interfaz ya la marca en
    rojo en la barra de estado con su RMS real, que es mas informativo que
    negarse a arrancar.
    """
    if not path.exists():
        raise _StartupError(
            f"intrinsics file not found: {path}. Run the intrinsic calibration first "
            "-- see the 'Setting up the rig' section of README.md."
        )
    try:
        return Intrinsics.load(path)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        # La tupla incluye ``KeyError``, que la de ``_load_config`` no
        # necesita: ``Intrinsics.load`` INDEXA el JSON
        # (``data["camera_matrix"]``, ``data["rms_px"]``, ...), asi que un
        # archivo de intrinsecos al que le falta un campo lanza
        # ``KeyError``, no ``TypeError``. ``AppConfig.load`` en cambio
        # expande con ``cls(**data)`` y ahi la clave ausente llega como
        # ``TypeError`` del constructor.
        raise _StartupError(f"intrinsics file {path} could not be read: {exc}") from exc


def probe_camera(index: int, *, open_capture=camera.open_capture) -> str | None:
    """Comprueba que la camara ``index`` abre, y la libera. ``None`` si abrio.

    Devuelve el mensaje de error en vez de lanzarlo para que sea
    comprobable sin tocar ``_StartupError``, y porque el unico lugar que
    decide codigos de salida es ``main``.

    **Siempre libera el capture, tambien cuando abre bien.** Este sondeo
    existe solo para poder fallar con codigo 1 ANTES de levantar la
    ventana; quien lee cuadros despues es ``CameraThread`` (Task 16), que
    abre el dispositivo por su cuenta. En Windows/DirectShow un segundo
    ``VideoCapture`` sobre un dispositivo ya tomado por este mismo proceso
    falla, asi que no liberar aqui romperia el arranque en el rig real
    mientras todas las pruebas del camino de error seguirian en verde.

    ``open_capture`` se inyecta porque esta aislado en ``camera.py``
    exactamente para eso (Task 16): sin inyeccion, probar este camino
    exigiria que la maquina de pruebas tuviera -- o no tuviera -- una
    camara en un indice concreto, y el resultado dependeria del hardware
    presente en vez del codigo.
    """
    try:
        capture = open_capture(index)
    except cv2.error as exc:
        # El backend puede lanzar en vez de devolver un capture cerrado
        # (DirectShow ausente, indice que el driver rechaza).
        return (
            f"camera {index} failed to open: {exc}. Check the connection and the "
            "device index (--camera N)."
        )
    try:
        if not capture.isOpened():
            return (
                f"camera {index} could not be opened. Check that it is connected, that "
                "--camera N is the right index, and that no other application is using it."
            )
    finally:
        capture.release()
    return None


def _build_detector() -> BananaDetector:
    """Construye el detector YOLO, traduciendo la falta de ``ultralytics``.

    El import de ``ultralytics`` es PEREZOSO (Task 12): vive dentro de la
    rama de ``BananaDetector`` que construye el modelo real, para que todo
    el resto del paquete y de las pruebas funcione sin esos ~2.5 GB
    instalados. El precio de esa decision es que la falta de la dependencia
    no se nota al importar nada, sino al arrancar el sistema -- en el rig,
    con todo ya montado. Sin esta traduccion llegaria como ``ImportError``
    crudo; con ella, como una linea que dice que instalar.
    """
    try:
        return BananaDetector()
    except ImportError as exc:
        raise _StartupError(
            f"the YOLO backend is unavailable: {exc}. Install it with "
            '\'pip install -e ".[live]"\' (the live extra; it pulls in torch, '
            "about 2.5 GB)."
        ) from exc


def _launch(
    cfg: AppConfig,
    intrinsics: Intrinsics,
    pipeline: Pipeline,
    camera_index: int,
    validation_log_path: Path,
):
    """Construye la ventana y la camara, y devuelve ``(app, window)`` sin
    arrancar el bucle de eventos todavia.

    Los imports de la interfaz son perezosos a proposito: ninguno de los
    caminos de fallo de arriba necesita el stack de widgets de Qt, y
    cargarlo igual haria que un error de config dependiera de que PySide6
    importe bien.

    Por que esto YA NO llama a ``app.exec()`` (I12, ronda de correcciones
    finales)
    --------------------------------------------------------------------------
    ``MainWindow.__init__`` construye un ``ValidationLog`` sobre el CSV real
    del rig (``pd.read_csv``): un archivo vacio (escritura interrumpida) o
    mal formado (editado a mano) lanzaba antes de este cambio como un
    traceback crudo, FUERA de cualquier guard -- el ``try/except
    _StartupError`` de ``main`` ya se habia cerrado para cuando esta
    funcion se llamaba, y en el peor momento posible: despues del sondeo de
    camara y la carga de YOLO, varios segundos de espera para leer un error
    que se podia dar al instante. Separar la CONSTRUCCION (que puede fallar
    por una causa de arranque) del bucle de eventos (que no es un fallo de
    arranque, es la aplicacion ya corriendo) deja a ``main`` envolver solo
    la primera bajo el mismo ``_StartupError`` que cualquier otro paso de
    esta cadena.
    """
    from PySide6.QtWidgets import QApplication

    from codex_caliper.camera import CameraThread
    from codex_caliper.ui.main_window import MainWindow

    # ``QApplication.instance()`` primero (I12, ronda de correcciones
    # finales, hallazgo de la suite completa): en el rig real nunca hay una
    # instancia previa y esto es un no-op, pero bajo pytest -- donde
    # ``pytest-qt`` ya crea y comparte UNA ``QApplication`` para toda la
    # sesion -- ``QApplication(...)`` a secas lanza
    # ``RuntimeError: ... destroy the QApplication singleton...`` en
    # cuanto cualquier otro archivo de pruebas con ``qtbot`` corrio antes
    # que ``test_entrypoint.py`` (el orden de recoleccion de pytest no es
    # garantia de lo contrario).
    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        window = MainWindow(
            intrinsics=intrinsics,
            pipeline=pipeline,
            cfg=cfg,
            validation_log_path=validation_log_path,
        )
    except (OSError, ValueError) as exc:
        # Misma tupla de excepciones "config mala, no bug" que
        # ``_load_config``/``_load_intrinsics`` -- ver sus comentarios. El
        # ``ValueError`` es el que ``ValidationLog.__init__`` lanza para un
        # CSV mal formado (ver su docstring); ``OSError`` cubre un archivo
        # ilegible por permisos. Cualquier otra excepcion (un ``TypeError``
        # de programacion, por ejemplo) se deja propagar: es un bug, no un
        # problema de puesta en marcha, y tiene que verse como tal.
        raise _StartupError(f"could not start the window: {exc}") from exc
    window.resize(*_INITIAL_WINDOW_SIZE_PX)

    camera_thread = CameraThread(camera_index)
    # Conectar ANTES de arrancar: el hilo puede emitir ``capture_lost`` en
    # su primera linea (camara desenchufada entre el sondeo y aqui), y una
    # senal emitida antes de conectar el slot no se entrega a nadie.
    window.connect_camera(camera_thread)
    camera_thread.start()

    return app, window


def main(argv: list[str] | None = None, *, open_capture=camera.open_capture) -> int:
    """Arranca el sistema. Devuelve 0 al cerrar la ventana, 1 si no pudo arrancar."""
    parser = argparse.ArgumentParser(
        prog="codex-caliper",
        description="Measure bananas against CODEX STAN 205-1997 from a live camera.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"rig configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=DEFAULT_INTRINSICS_PATH,
        help=f"intrinsic calibration file (default: {DEFAULT_INTRINSICS_PATH})",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="camera device index (default: 0)",
    )
    args = parser.parse_args(argv)

    try:
        cfg = _load_config(args.config)
        _require_measured_layout(cfg, args.config)
        intrinsics = _load_intrinsics(args.intrinsics)

        camera_error = probe_camera(args.camera, open_capture=open_capture)
        if camera_error is not None:
            raise _StartupError(camera_error)

        detector = _build_detector()
        pipeline = Pipeline(cfg, intrinsics, detector)

        # La CONSTRUCCION de la ventana entra aqui, dentro del mismo guard
        # (I12, ronda de correcciones finales) -- ver el docstring de
        # ``_launch`` para por que ya no incluye ``app.exec()``.
        app, window = _launch(
            cfg,
            intrinsics,
            pipeline,
            args.camera,
            args.config.parent / VALIDATION_LOG_NAME,
        )
    except _StartupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # El bucle de eventos queda FUERA del guard a proposito: una vez que la
    # ventana existe, cualquier fallo de aqui en adelante es un bug de la
    # aplicacion en marcha, no un problema de puesta en marcha que
    # ``_StartupError`` deba traducir.
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

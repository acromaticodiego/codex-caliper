"""Validacion, regresion de densidad y metricas de error (Task 15).

Este modulo es la unica parte del proyecto cuyo trabajo es contestar "funciona
el sistema" con un numero y su incertidumbre -- el objetivo 2 del spec pide
explicitamente "reportar cada medida con su incertidumbre" y una tabla de
validacion, y el anti-objetivo 4 declara que "reportar cifras sin error
asociado" hace fallar el proyecto aunque el codigo funcione. Ninguna funcion
de aqui devuelve un numero sin su companero de error.

Separar dos preguntas que no se deben mezclar (spec §7 etapa 10)
------------------------------------------------------------------
"¿Es correcta la metrologia?" y "¿es correcta la segmentacion del banano?"
son preguntas DISTINTAS, con fuentes de error distintas, y se validan con
instrumentos distintos a proposito:

- La metrologia (calibracion, homografia, correccion de paralaje) se valida
  con patrones PLANOS de dimension normalizada conocida -- una tarjeta
  ISO/IEC 7810 ID-1 (85.60 x 53.98 mm), un ArUco dejado fuera de la
  calibracion (debe medir 24.0 mm), monedas. Al ser planos aislan el error
  de calibracion del error de paralaje (que depende de la altura del
  objeto). ``dimensional_check`` es la funcion para esto.
- La segmentacion del banano se valida con la bascula de cocina:
  ``fit_density`` y ``error_metrics`` comparan el peso real contra el
  volumen calculado a partir de la mascara.

"Mezclar ambas impide saber de donde viene el error" (spec §7 etapa 10) --
por eso son funciones separadas y ninguna llama a la otra.

La longitud a mano cubre lo que la bascula no ve (spec §7.b)
---------------------------------------------------------------
El recorte del pedicelo (``measures.trim_pedicel``, Task 8) aterriza dentro
de unos +-3 mm de la base real del pedicelo -- en fruta con cabo, ese es el
error DOMINANTE del sistema, por encima del objetivo declarado de +-1 mm.
La bascula no lo puede detectar: el peso de un banano no cambia segun donde
el software decidio que termina el fruto y empieza el pedicelo, asi que un
banano mal recortado pero bien pesado pasa la validacion por peso sin
problema. La unica forma de medir ESTE error es compararlo contra una
longitud tomada a mano con regla sobre al fruto real (spec §7.b: "la
validacion de §7 etapa 10 debe incluir longitud medida a mano con regla
sobre fruta real con cabo, no solo peso en bascula"). Por eso
``ValidationLog.record`` acepta ``actual_length_mm`` (opcional -- los
patrones planos de ``dimensional_check`` no pasan por aqui y no tienen
longitud de fruta que medir), y ``error_metrics`` reporta
``mae_length_mm``/``n_length`` sobre las filas donde esa columna este
presente. No se le agrega un MAPE de longitud a proposito (YAGNI): el
estandar CODEX ya trabaja en milimetros absolutos (140 mm, 27 mm), y el
limite de +-3 mm del spec esta expresado en milimetros, no en porcentaje --
un MAE en mm es la metrica que de verdad se compara contra ese numero.

La densidad se ajusta, no se asume
------------------------------------
``fit_density`` hace una regresion lineal POR EL ORIGEN de peso real contra
volumen calculado -- no una recta con ordenada libre. La pendiente es la
densidad efectiva del sistema: mas honesta y mas precisa que tomar una
densidad de tabla (spec: "la densidad no se asume: se ajusta"). Por el
origen porque la cantidad fisica que se busca ES la densidad (peso =
densidad * volumen, sin termino constante); dejar un intercepto libre
responde una pregunta distinta ("¿cual es la mejor recta afin para estos
puntos?") y su pendiente ya no es una densidad.

La convencion de R² para una regresion por el origen (ver mas abajo) se
fija aparte porque es ambigua y el spec declara este numero "un resultado
publicable" (§7 etapa 10): hay que elegir una convencion a proposito, no
heredar la que caiga de la libreria que se use.

Convencion de R² elegida: NO centrada
----------------------------------------
En una regresion CON intercepto, R² = 1 - SS_res/SS_tot con
SS_tot = sum((y - y_barra)²) (centrada en la media) es la convencion
universal, y nunca es negativa porque el modelo siempre puede igualar la
media.

En una regresion POR EL ORIGEN eso deja de ser cierto: el modelo no tiene
libertad para ajustar un intercepto, asi que puede (y en la practica
suele) dar un ajuste PEOR que simplemente predecir la media -- lo que hace
SS_res > SS_tot y produce un R² centrado NEGATIVO, incluso para un ajuste
que a simple vista es excelente (ver
``tests/test_validation.py::test_r_squared_uses_the_uncentered_through_origin_convention``,
donde SS_res/SS_tot centrada da R² = -0.007 para un ajuste con error
relativo menor al 1%). Reportar ese numero como "el R² del sistema" seria
publicar una cifra tecnicamente correcta pero que no significa lo que
cualquier lector asumiria que significa -- exactamente el tipo de resultado
plausible-pero-enganoso que este proyecto evita en otras partes.

Este modulo usa la convencion NO CENTRADA (SS_tot = sum(y²), el origen como
referencia en vez de la media), que es la convencion estandar para
regresiones sin intercepto (statsmodels, por ejemplo, cambia a esta
definicion automaticamente cuando se ajusta sin constante) y que coincide
con lo que un lector espera de "que tan bien explica el volumen el peso"
cuando el modelo que se esta evaluando es, por construccion, uno que pasa
por el origen.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from codex_caliper.geometry.measures import BananaMeasures

#: Minimo de muestras para que la regresion de densidad sea confiable
#: (spec: "exigir al menos 10 muestras"). Por debajo de esto una pendiente
#: ajustada no es mas honesta que una densidad de tabla -- es ruido con
#: pretensiones.
MIN_DENSITY_SAMPLES = 10

#: Columnas del CSV de validacion, en el orden en que se escriben. Nombres
#: en ingles: son encabezados de cara al usuario.
#:
#: ``length_std_mm``/``caliper_std_mm`` (ronda de correcciones finales, I9):
#: la unica fila de este CSV que SI tiene una desviacion disponible es la
#: que viene de un ``TrackAggregate`` asentado (``ValidationView.record_sample``,
#: que trae ``length_std_mm``/``caliper_std_mm`` del agregado temporal,
#: Task 11); una muestra de patron plano (``dimensional_check``) no pasa
#: por ``ValidationLog.record`` en absoluto. Sin estas dos columnas, la
#: tabla de validacion publicable (spec S13, punto 4) mostraba longitud y
#: calibre SIN su incertidumbre -- el anti-objetivo exacto que
#: ``format_measurement`` existe para cerrar en todos los demas caminos de
#: la interfaz. Son OPCIONALES (``None`` -> NaN en el CSV) para que una
#: fila sin desviacion (``ValidationLog.record`` llamado directo, como
#: hacen las pruebas de este modulo) siga siendo un CSV valido.
_COLUMNS = [
    "sample_id",
    "codex_length_mm",
    "length_std_mm",
    "caliper_mm",
    "caliper_std_mm",
    "area_mm2",
    "volume_mm3",
    "actual_weight_g",
    "actual_length_mm",
]


@dataclasses.dataclass(frozen=True)
class DensityFit:
    """Resultado de ajustar peso real contra volumen calculado, por el origen.

    ``g_per_cm3`` es la pendiente ajustada (la densidad efectiva del
    sistema), ``r_squared`` usa la convencion NO CENTRADA (ver docstring
    del modulo) y ``n`` es la cantidad de muestras usadas.
    """

    g_per_cm3: float
    r_squared: float
    n: int


@dataclasses.dataclass(frozen=True)
class ErrorMetrics:
    """Metricas de error del peso predicho (y, si hay datos, de la longitud).

    ``mae_g``, ``mape_pct``, ``r_squared`` y ``n`` cubren el peso -- la
    validacion por bascula, que mide si la segmentacion del banano es
    correcta. Nunca se reporta una cifra de error sin decir sobre cuantas
    muestras se calculo (regla del modulo): ``n`` es ese conteo. Ninguna
    fila se excluye de las metricas de peso (a diferencia de las de
    longitud, mas abajo), asi que ``n == len(frame)`` siempre -- ver
    ``error_metrics`` para por que eso es asi y no un ``len(frame)`` suelto.

    ``mae_length_mm`` cubre la longitud medida a mano (spec §7.b): es
    ``None`` cuando ninguna fila del lote trae ``actual_length_mm``, para
    que validar solo con bascula (sin regla) no falle ni invente un numero.
    ``n_length`` cuenta cuantas filas SI tenian longitud a mano, que puede
    ser (y en la practica suele ser) menor que ``n``.
    """

    mae_g: float
    mape_pct: float
    r_squared: float
    n: int
    mae_length_mm: float | None
    n_length: int


class ValidationLog:
    """Bitacora CSV de muestras de validacion: identificador, medidas, peso real.

    Cada ``record`` escribe el CSV completo de inmediato (no hay un
    ``flush`` o ``close`` separado que se pueda olvidar) y el constructor lee
    el archivo existente si ya hay uno, para que retomar una sesion de
    validacion en curso acumule sobre las muestras anteriores en vez de
    perderlas.

    Guardia contra un CSV vacio o mal formado (I12, ronda de correcciones
    finales)
    --------------------------------------------------------------------------
    ``MainWindow.__init__`` construye un ``ValidationLog`` sobre el archivo
    REAL del rig -- este constructor corre despues del sondeo de camara y
    la carga de YOLO (varios segundos), el peor momento posible para un
    traceback crudo. Dos formas conocidas de que ese archivo este roto sin
    que ninguna muestra se haya perdido de verdad:

    - **Vacio (0 bytes).** Una escritura interrumpida (corte de luz, cierre
      forzado a mitad del primer ``record``) deja un archivo que EXISTE
      pero no tiene ni encabezado. ``pd.read_csv`` lanza
      ``pandas.errors.EmptyDataError`` -- se trata exactamente igual que
      "el archivo no existe": no hay ninguna fila que recuperar de un CSV
      vacio.
    - **Mal formado.** Editado a mano, truncado a mitad de una fila, o
      "arreglado" por una herramienta que no respeto las comillas.
      ``pd.read_csv`` lanza ``pandas.errors.ParserError``. A diferencia del
      caso vacio, aqui SI puede haber datos reales antes del punto de
      ruptura -- descartarlos en silencio seria la misma perdida silenciosa
      que este proyecto evita en otras partes -- asi que esto SI se deja
      propagar, pero como un ``ValueError`` con un mensaje para el
      operador, no como el ``ParserError`` crudo de pandas. El punto de
      entrada (``__main__._launch``, I12) construye la ventana bajo el
      mismo guard ``_StartupError`` que cualquier otro fallo de arranque,
      asi que este ``ValueError`` termina como una linea en stderr y
      codigo de salida 1 -- nunca un traceback.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            self._rows: list[dict[str, Any]] = []
            return
        try:
            self._rows = pd.read_csv(self.path).to_dict("records")
        except pd.errors.EmptyDataError:
            self._rows = []
        except pd.errors.ParserError as exc:
            raise ValueError(
                f"validation log {self.path} exists but could not be parsed as CSV: {exc}. "
                "Fix it by hand, or move it aside to start a fresh log, before running again."
            ) from exc

    def record(
        self,
        sample_id: str,
        measures: BananaMeasures,
        actual_weight_g: float,
        actual_length_mm: float | None = None,
        length_std_mm: float | None = None,
        caliper_std_mm: float | None = None,
    ) -> None:
        """Agrega una muestra y reescribe el CSV.

        ``actual_length_mm`` es OPCIONAL (spec §7.b): las muestras de
        patrones planos (tarjeta, ArUco, monedas) pasan por
        ``dimensional_check`` en vez de por aqui y no tienen una longitud de
        fruta que medir; las muestras de banano real con cabo si deberian
        traerla, porque es la unica forma de validar el limite de +-3 mm
        del recorte de pedicelo.

        ``length_std_mm``/``caliper_std_mm`` son OPCIONALES por la misma
        razon (I9, ronda de correcciones finales): quien SI las tiene es
        ``ValidationView.record_sample``, que las lee del ``TrackAggregate``
        asentado -- un llamador directo de este metodo (como hacen las
        pruebas de este modulo, con un ``BananaMeasures`` de un solo
        cuadro) no tiene una desviacion temporal que ofrecer, y ``None``
        sobrevive al CSV como NaN en vez de fabricar un 0.0 que afirmaria
        "incertidumbre cero" con tanta falsedad como no mostrar nada.
        """
        self._rows.append(
            {
                "sample_id": sample_id,
                "codex_length_mm": measures.codex_length_mm,
                "length_std_mm": length_std_mm,
                "caliper_mm": measures.caliper_mm,
                "caliper_std_mm": caliper_std_mm,
                "area_mm2": measures.area_mm2,
                "volume_mm3": measures.volume_mm3,
                "actual_weight_g": actual_weight_g,
                "actual_length_mm": actual_length_mm,
            }
        )
        self.to_frame().to_csv(self.path, index=False)

    def to_frame(self) -> pd.DataFrame:
        """Devuelve todas las muestras registradas hasta ahora como ``DataFrame``."""
        return pd.DataFrame(self._rows, columns=_COLUMNS)


def _r_squared_uncentered(actual: np.ndarray, predicted: np.ndarray) -> float:
    """R² con la suma de cuadrados total medida desde CERO, no desde la media.

    Convencion elegida para regresiones por el origen -- ver docstring del
    modulo para la justificacion completa y el caso donde difiere de la
    convencion centrada habitual.
    """
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum(actual**2))
    return 1.0 - ss_res / ss_tot


def fit_density(frame: pd.DataFrame) -> DensityFit:
    """Ajusta densidad = peso / volumen por regresion lineal POR EL ORIGEN.

    Exige al menos ``MIN_DENSITY_SAMPLES`` filas -- ver docstring del
    modulo. ``frame`` debe tener las columnas ``volume_mm3`` y
    ``actual_weight_g`` (las que produce ``ValidationLog.to_frame``).
    """
    n = len(frame)
    if n < MIN_DENSITY_SAMPLES:
        raise ValueError(
            f"fit_density needs at least {MIN_DENSITY_SAMPLES} samples, got {n}"
        )

    volume_cm3 = frame["volume_mm3"].to_numpy(dtype=float) / 1000.0
    weight_g = frame["actual_weight_g"].to_numpy(dtype=float)

    # Minimos cuadrados por el origen: minimizar sum((w - k*v)^2) sobre k
    # da k = sum(v*w) / sum(v*v) -- no hay termino de intercepto que ajustar.
    g_per_cm3 = float(np.sum(volume_cm3 * weight_g) / np.sum(volume_cm3 * volume_cm3))

    predicted = g_per_cm3 * volume_cm3
    r_squared = _r_squared_uncentered(weight_g, predicted)

    return DensityFit(g_per_cm3=g_per_cm3, r_squared=r_squared, n=n)


def error_metrics(frame: pd.DataFrame, density: DensityFit) -> ErrorMetrics:
    """Calcula MAE (g), MAPE (%) y R² del peso predicho por ``density``.

    Si ``frame`` trae ``actual_length_mm`` (spec §7.b), tambien calcula el
    MAE de la longitud (``codex_length_mm`` contra esa columna) sobre las
    filas donde el valor no es nulo -- ver docstring del modulo y de
    ``ErrorMetrics``.

    Exige al menos una fila: un lote vacio (el estado inicial de cualquier
    ``ValidationLog``, antes de registrar la primera muestra) no debe
    convertirse silenciosamente en ``nan`` -- ``np.mean`` de un arreglo
    vacio y la division del MAPE lo harian sin avisar, con un
    ``RuntimeWarning`` de regalo, y "MAE: nan g" es exactamente la cifra
    sin sentido que el objetivo de este modulo prohibe publicar. Por la
    misma razon se exige que ningun ``actual_weight_g`` sea cero: una
    bascula no pesa 0 g un banano real, asi que un cero ahi es un dato mal
    capturado, no un caso valido que haya que tolerar en silencio -- y
    dividir por el en el denominador del MAPE produce el mismo tipo de
    ``nan``/``inf`` que un lote vacio.
    """
    n = len(frame)
    if n == 0:
        raise ValueError("error_metrics requires at least one row, got an empty frame")

    volume_cm3 = frame["volume_mm3"].to_numpy(dtype=float) / 1000.0
    actual_weight_g = frame["actual_weight_g"].to_numpy(dtype=float)
    if np.any(actual_weight_g == 0.0):
        raise ValueError(
            "error_metrics got a row with actual_weight_g == 0.0 (a real banana cannot weigh 0 g)"
        )

    predicted_weight_g = density.g_per_cm3 * volume_cm3

    error = predicted_weight_g - actual_weight_g
    mae_g = float(np.mean(np.abs(error)))
    mape_pct = float(np.mean(np.abs(error) / np.abs(actual_weight_g)) * 100.0)
    r_squared = _r_squared_uncentered(actual_weight_g, predicted_weight_g)

    # Ninguna fila se excluye de las metricas de peso de arriba -- a
    # diferencia de las de longitud, abajo -- asi que n == len(frame), no
    # un conteo filtrado. Ver ErrorMetrics.n.
    mae_length_mm: float | None = None
    n_length = 0
    if "actual_length_mm" in frame.columns:
        has_length = frame["actual_length_mm"].notna().to_numpy()
        n_length = int(np.sum(has_length))
        if n_length > 0:
            measured_length_mm = frame["codex_length_mm"].to_numpy(dtype=float)[has_length]
            hand_length_mm = frame["actual_length_mm"].to_numpy(dtype=float)[has_length]
            mae_length_mm = float(np.mean(np.abs(measured_length_mm - hand_length_mm)))

    return ErrorMetrics(
        mae_g=mae_g,
        mape_pct=mape_pct,
        r_squared=r_squared,
        n=n,
        mae_length_mm=mae_length_mm,
        n_length=n_length,
    )


def dimensional_check(measured_mm: float, reference_mm: float) -> float:
    """Error con signo (mm) de una medida contra un patron plano de referencia.

    Para la tarjeta ISO/IEC 7810 ID-1, un ArUco dejado fuera de la
    calibracion, o una moneda -- objetos PLANOS cuya dimension real se
    conoce, usados para aislar el error de calibracion del de paralaje
    (spec §7 etapa 10). Con signo a proposito: un ``abs()`` aqui borraria
    si el sistema sobre o sub-estima, que es justo lo que un sesgo
    sistematico de calibracion necesita para diagnosticarse.
    """
    return float(measured_mm - reference_mm)

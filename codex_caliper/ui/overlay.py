"""Overlay de cotas sobre el frame (Task 17).

Este es el modulo donde la afirmacion central del proyecto se hace VISIBLE.
La spec (S8) es explicita al respecto: "si los numeros viven unicamente en
la tabla, el video pasa a ser decorativo y la mirada se va al panel -- justo
lo contrario del objetivo". Por eso ``draw_results`` dibuja las cotas
directamente sobre el lienzo rectificado, ancladas a cada banano por su
propia mascara, no solo las deja disponibles para un panel aparte.

``format_measurement`` no es un helper de formato: es el punto donde se
hace cumplir uno de los dos anti-objetivos que este modulo puede violar por
si solo (S4) -- no reportar una cifra sin su incertidumbre, y no mostrar mas
decimales de los que la calibracion sostiene.

Piso de incertidumbre (Ruling 2 del encargo)
---------------------------------------------
``aggregation.py`` calcula la desviacion con ``np.std`` sin ``ddof``, asi
que la desviacion de UNA sola muestra es exactamente 0.0 -- no un caso raro,
sino el primer frame de CADA track que el sistema mide. Sin piso,
``format_measurement`` imprimiria una precision infinita ahi (decimales sin
limite, la definicion misma de falsa precision). El piso fisico correcto no
es un caso especial para ``samples == 1``: es ``mm_per_px``. La spec (S5.1)
mide exactamente esto -- a 0.2 mm/px una ambiguedad de borde de 1 px son
0.2 mm, asi que el sistema no puede reclamar mas resolucion que la de su
propio lienzo, sin importar cuanto se asiente la dispersion temporal.

``format_measurement(value_mm, std_mm, floor_mm=_CANVAS_MM_PER_PX)`` aplica
el piso como PRIMER paso, incondicionalmente, antes de decidir decimales.
El parametro tiene un default (la escala fija del lienzo del rig, S5.1) para
que un caller que lo olvide siga protegido -- el piso no depende de que
alguien se acuerde de pasarlo. ``draw_results`` si pasa el ``mm_per_px``
real de cada frame explicitamente, porque esa es la escala verdadera de ESA
calibracion y puede ser mas fina que el default si el montaje cambia.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from codex_caliper.aggregation import TEMPORAL_WINDOW
from codex_caliper.aruco_tools import CalibrationState, MarkerDetection, calibration_state
from codex_caliper.pipeline import BananaResult

#: Escala del lienzo rectificado del rig (spec S5.1, S7). Es el default del
#: piso de incertidumbre de ``format_measurement``, no un valor operativo
#: que este modulo decida: si el montaje cambia de escala, ``draw_results``
#: ya pasa el ``mm_per_px`` real de cada frame y ese valor manda.
_CANVAS_MM_PER_PX = 0.2

#: Margen en pixeles entre el borde de la mascara y el texto que seria
#: ancla (Ruling 1: no hay bbox, se deriva de la mascara). Mismo numero para
#: X e Y: no hay ninguna razon fisica para que sean distintos.
_LABEL_MARGIN_PX = 10

#: Alto de linea del bloque de texto, en pixeles, a la escala de fuente
#: fija de este modulo (``_FONT_SCALE``).
_LINE_HEIGHT_PX = 18

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_TEXT_THICKNESS = 1
#: Grosor del trazo negro de contorno. Mas grueso que el relleno para que
#: sobreviva un fondo bananero: amarillo saturado contra blanco relleno es
#: el caso que mas cuesta leer, y el contorno negro es legible sobre
#: cualquiera de los dos. No depender solo del color para el contraste.
_OUTLINE_THICKNESS = 3

#: Colores en BGR (convencion de OpenCV). El texto nunca distingue estados
#: SOLO por color -- "OK"/"FAIL" y la presencia/ausencia de la barra de
#: progreso ya lo hacen por contenido -- el color es un refuerzo, no el
#: unico canal.
_COLOR_PASS = (60, 200, 60)
_COLOR_FAIL = (40, 40, 230)
_COLOR_MEASURING = (210, 210, 210)

_MARKER_COLORS = {
    CalibrationState.OK: (60, 200, 60),
    CalibrationState.DEGRADED: (0, 165, 255),
    CalibrationState.BLOCKED: (40, 40, 230),
}


def format_measurement(value_mm: float, std_mm: float, floor_mm: float = _CANVAS_MM_PER_PX) -> str:
    """Formatea ``value_mm ± std_mm`` con tantos decimales como la incertidumbre sostenga.

    Convencion de metrologia: la incertidumbre se redondea a 1 cifra
    significativa, y el VALOR se redondea al mismo lugar decimal que esa
    incertidumbre redondeada -- no al reves, y no por una cadena de
    umbrales ad hoc. A 0.82 mm de incertidumbre (redondea a 0.8, un
    decimal) el valor se imprime con un decimal; a 5.0 mm (redondea a 5,
    cero decimales) el valor se imprime entero. Por debajo de ``floor_mm``
    la incertidumbre se recorta al piso ANTES de este calculo (ver
    docstring del modulo): es la unica forma de que una desviacion de
    0.0 exacta -- el primer frame de cada track, Ruling 2 -- no produzca
    precision infinita.
    """
    std_effective = max(std_mm, floor_mm)
    std_rounded, decimals = _round_to_one_sig_fig(std_effective)
    return f"{value_mm:.{decimals}f} ± {std_rounded:.{decimals}f} mm"


def _round_to_one_sig_fig(std_mm: float) -> tuple[float, int]:
    """Redondea ``std_mm`` a 1 cifra significativa y devuelve (redondeado, decimales).

    ``decimals`` es cuantos decimales hacen falta para representar
    EXACTAMENTE el valor ya redondeado -- no el original, porque redondear
    puede subir un orden de magnitud (0.96 -> 1.0) y los decimales tienen
    que seguir al resultado, no al punto de partida.
    """
    if std_mm <= 0:
        return 0.0, 0
    exponent = math.floor(math.log10(std_mm))
    rounded = round(std_mm, -exponent)
    if rounded > 0:
        exponent = math.floor(math.log10(rounded))
    decimals = max(0, -exponent)
    return rounded, decimals


def text_anchor_for(mask: np.ndarray) -> tuple[int, int]:
    """Punto de anclaje (x, y) para el rotulo de un banano, derivado de su mascara.

    No existe bbox en ``BananaResult`` (Ruling 1: el pipeline no lo
    propaga a proposito, ver Task 14). Se deriva directamente de la
    extension real de la mascara booleana: la esquina superior derecha de
    su caja delimitadora, mas un margen fijo para no superponerse al
    contorno del fruto. Al moverse la mascara el ancla se traslada con
    ella exactamente lo mismo -- es la propiedad que hace que "los numeros
    sigan al objeto" (spec S8) sea cierto y no solo un efecto visual
    casual.

    Precondicion: ``mask`` tiene al menos un pixel en ``True``. El
    pipeline nunca produce un ``BananaResult`` con una mascara vacia
    (``refine_mask`` devuelve ``None`` antes de eso, Task 13); una mascara
    vacia aqui es un error de programacion aguas arriba, no un caso de
    frame en vivo, y se deja propagar la excepcion en vez de inventar un
    ancla sin sentido.
    """
    ys, xs = np.nonzero(mask)
    x_max = int(xs.max())
    y_min = int(ys.min())
    return x_max + _LABEL_MARGIN_PX, y_min + _LABEL_MARGIN_PX


def _progress_bar(samples: int, window: int = TEMPORAL_WINDOW, segments: int = 4) -> str:
    """Barra ASCII de progreso (p.ej. ``"###."``) -- ver nota de glifos en ``draw_results``."""
    fraction = min(samples / window, 1.0)
    filled = round(fraction * segments)
    return "#" * filled + "." * (segments - filled)


def compose_label_lines(result: BananaResult, mm_per_px: float) -> list[str]:
    """Lineas de texto del rotulo de un banano, en el orden del mockup (spec S8).

    Funcion PURA que devuelve cadenas (Ruling 3 del encargo): el brief
    pedia leer texto de vuelta de una imagen ya dibujada
    (``rendered_text_of``), que es OCR y no se puede escribir de forma
    fiable. Aqui se prueba la composicion sin pasar por pixeles, y
    ``draw_results`` solo coloca estas cadenas.

    Mientras el agregado NO esta ``locked`` (Task 11: ventana sin llenar o
    todavia ruidosa) se muestra unicamente el id y la barra de progreso --
    nunca L/CAL/peso. Mostrar esos numeros mientras la ventana sigue
    inestable seria exactamente el tipo de cifra plausible-pero-cambiante
    que el proyecto ya evito en otros cinco sitios (ver
    ``aggregation.py``): el veredicto Codex depende de dos umbrales y
    podria alternar pass/fail frame a frame con la interfaz mostrando
    numeros como si fueran definitivos. Una vez ``locked`` ocurre lo
    opuesto: la barra de progreso desaparece (ya no esta "progresando",
    esta fija) y aparecen el veredicto, L, calibre y peso, en ese orden --
    el orden del mockup de la spec.
    """
    aggregate = result.aggregate
    if not aggregate.locked:
        return [f"#{result.track_id}", f"{_progress_bar(aggregate.samples)} {aggregate.samples}/{TEMPORAL_WINDOW}"]

    verdict_word = "OK" if result.verdict.passes else "FAIL"
    lines = [
        f"#{result.track_id} {verdict_word}",
        f"L {format_measurement(aggregate.length_mm, aggregate.length_std_mm, mm_per_px)}",
        f"CAL {format_measurement(aggregate.caliper_mm, aggregate.caliper_std_mm, mm_per_px)}",
    ]
    if result.estimated_weight_g is not None:
        lines.append(f"{round(result.estimated_weight_g)} g")
    return lines


def _label_color(result: BananaResult) -> tuple[int, int, int]:
    if not result.aggregate.locked:
        return _COLOR_MEASURING
    return _COLOR_PASS if result.verdict.passes else _COLOR_FAIL


def _draw_outlined_text(canvas: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    """Dibuja ``text`` con un trazo negro de contorno antes del relleno.

    Es la defensa de legibilidad contra un fondo arbitrario (la
    restriccion concreta del encargo: un overlay illegible sobre un
    banano amarillo brillante no sirve). El contorno negro se ve sobre
    amarillo, blanco, o cualquier color de fondo del video en vivo; no se
    depende solo del color del texto para el contraste.
    """
    cv2.putText(canvas, text, origin, _FONT, _FONT_SCALE, (0, 0, 0), _OUTLINE_THICKNESS, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, _FONT, _FONT_SCALE, color, _TEXT_THICKNESS, cv2.LINE_AA)


def draw_results(canvas: np.ndarray, results: list[BananaResult], mm_per_px: float) -> np.ndarray:
    """Dibuja las cotas de cada banano SOBRE una copia del lienzo rectificado.

    Nunca sobre ``canvas`` directamente -- el caller sigue sosteniendo el
    lienzo original del frame para el resto del pipeline (p.ej. el PiP, o
    el frame siguiente), y mutarlo en el sitio seria una accion lateral
    que nada en la firma anuncia.

    Cada etiqueta se ancla a la mascara DE ESE resultado
    (``text_anchor_for``), nunca a su posicion en la lista: dos objetos
    pueden llegar en cualquier orden y cada rotulo tiene que seguir al
    banano que describe, no al indice con el que llego este frame (spec
    S8: "los numeros siguen a los bananos cuando se mueven con la mano").

    Nota sobre glifos: el mockup de la spec (S8) usa "✓" y "⌀" y una barra
    "▓▓▓░". ``cv2.putText`` usa las fuentes vectoriales Hershey de OpenCV,
    que solo cubren ASCII imprimible -- un caracter Unicode multibyte se
    codifica a UTF-8 y OpenCV dibuja cada BYTE como si fuera un indice de
    glifo independiente, produciendo marcas ilegibles, no el simbolo
    pedido (verificado: una sola "⌀" produce varias marcas pequenas
    inconexas, no un simbolo). Por eso el contenido se traduce a ASCII
    ("OK"/"FAIL", "CAL", y una barra con "#"/"."): el LAYOUT del mockup
    (que se muestra y en que orden) es vinculante, los glifos concretos no
    sobreviven al motor de dibujo elegido para esta tarea.
    """
    out = canvas.copy()
    for result in results:
        lines = compose_label_lines(result, mm_per_px)
        color = _label_color(result)
        x, y = text_anchor_for(result.mask)
        for i, line in enumerate(lines):
            _draw_outlined_text(out, line, (x, y + i * _LINE_HEIGHT_PX), color)
    return out


def draw_marker_overlay(frame: np.ndarray, detections: list[MarkerDetection]) -> np.ndarray:
    """Dibuja el contorno y el id de cada marcador ArUco sobre una copia de ``frame``.

    El color codifica el estado de calibracion que esas mismas detecciones
    sostienen (``aruco_tools.calibration_state``, derivado del CONTEO de
    ids distintos): verde con los 4 marcadores del rig, naranja con 3
    (``DEGRADED`` -- la spec S7 etapa 2 exige seguir midiendo y decirlo),
    rojo por debajo. No hace falta un parametro de estado aparte: la
    propia lista de detecciones ya lo determina, igual que en
    ``aruco_tools``, y pasar el estado por separado solo abriria la puerta
    a que quedara desincronizado de las detecciones que realmente se
    dibujan.
    """
    out = frame.copy()
    color = _MARKER_COLORS[calibration_state(detections)]
    for detection in detections:
        pts = detection.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
        label_x = int(detection.corners[:, 0].min())
        label_y = int(detection.corners[:, 1].min()) - 6
        _draw_outlined_text(out, f"#{detection.id}", (label_x, label_y), color)
    return out

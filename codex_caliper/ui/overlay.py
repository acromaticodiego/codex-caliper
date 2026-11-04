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

El metodo tambien se hace VISIBLE, no solo el numero (encargo posterior al
plan de 20 tareas)
-----------------------------------------------------------------------------
Hasta aqui ``draw_results`` solo dibujaba texto: el contorno, el eje medial
y la curva Codex se calculaban (Tasks 6-8) y se descartaban sin pintarse en
ningun lado, asi que todo el metodo -- deteccion, rectificacion,
segmentacion, esqueletizacion, medicion de arco -- quedaba invisible en el
video, exactamente lo que un demo en vivo no puede permitirse ("que la
persona que vea esto vea la implementacion, no que quede tras bambalinas").
``_draw_geometry`` dibuja, por cada ``BananaResult``, en orden de MENOS a
MAS protagonista: el contorno fino de la mascara, el eje medial completo,
la cuerda punta a punta PUNTEADA (el contraste directo), y por ultimo --
encima de las otras tres -- la curva Codex con halo, la cantidad que el
panel reporta como ``L``. Ningun paso recalcula geometria: ``BananaResult``
ya la carga desde ``measure_mask`` (Task 8) via el pipeline (Task 14, ver
sus docstrings), porque recalcular el eje medial en este hilo -- el de la
GUI, por cuadro -- cuesta ~11 ms/banano (ver ``geometry/medial.py``) y
pondria la fluidez del video en riesgo por el mismo overlay que existe
para demostrarlo.

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

El piso y la regla de decimales viven ahora en ``readout.py``, junto a
``format_measurement``: este modulo dibuja, no decide cifras.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from codex_caliper.aruco_tools import CalibrationState, MarkerDetection, calibration_state
from codex_caliper.pipeline import BananaResult
from codex_caliper.ui.readout import compose_canvas_tag

#: Escala del lienzo rectificado del rig (spec S5.1, S7). Es el default del
#: piso de incertidumbre de ``format_measurement``, no un valor operativo
#: que este modulo decida: si el montaje cambia de escala, ``draw_results``
#: ya pasa el ``mm_per_px`` real de cada frame y ese valor manda.

#: Margen en pixeles entre el borde de la mascara y el texto que seria
#: ancla (Ruling 1: no hay bbox, se deriva de la mascara). Mismo numero para
#: X e Y: no hay ninguna razon fisica para que sean distintos.
_LABEL_MARGIN_PX = 10

#: Alto de linea del bloque de texto, en pixeles, a la escala de fuente
#: fija de este modulo (``_FONT_SCALE``).
_LINE_HEIGHT_PX = 34

#: Ancho presupuestado, en pixeles, para cualquier linea que este modulo
#: dibuje a su fuente fija (ronda 1 de revision, Finding 2). No se mide
#: con ``cv2.getTextSize`` por cada render -- eso acoplaria el ancla al
#: contenido exacto del texto -- se reserva un presupuesto generoso una
#: sola vez, igual que ``_LINE_HEIGHT_PX`` ya hace con el alto. La linea
#: mas larga real ("CAL 999.9 +/- 99.9 mm") cabe con margen de sobra.
_LABEL_BLOCK_WIDTH_PX = 190

#: Lineas maximas que ``compose_label_lines`` puede devolver (veredicto,
#: L, CAL, peso -- el mockup completo de S8). Solo se usa para reservar
#: espacio VERTICAL al anclar (Finding 2); no decide que se muestra.
_LABEL_MAX_LINES = 1

_FONT = cv2.FONT_HERSHEY_SIMPLEX
#: Escala y grosores dimensionados para que el texto se LEA EN VIDEO, que
#: es a lo que esta cadena de cotas se destina (spec S13: el video en vivo
#: es lo primero que se publica, y se ve en un movil). Los valores
#: anteriores -- escala 0.5, relleno 1, contorno 3 -- tenian dos defectos
#: que solo aparecen al renderizar una linea real a resolucion nativa:
#:
#: 1. El contorno era TRES VECES el grosor del relleno, asi que en una
#:    linea densa como "CAL 38.9 +/- 0.2 mm" los halos de letras contiguas
#:    se tocaban y rellenaban los huecos: el texto se leia como una mancha
#:    en negrita, con el espaciado aparentemente roto.
#: 2. A escala 0.5 el bloque ocupaba ~180 px de un lienzo de ~1240, ilegible
#:    en la pantalla de un telefono.
#:
#: El contorno sigue siendo mas grueso que el relleno -- es la defensa de
#: legibilidad contra el amarillo saturado del banano, y no se depende solo
#: del color para el contraste -- pero ahora en proporcion al trazo, no
#: ahogandolo.
_FONT_SCALE = 1.0
_TEXT_THICKNESS = 2
_OUTLINE_THICKNESS = 6

#: Colores en BGR (convencion de OpenCV). El texto nunca distingue estados
#: SOLO por color -- "OK"/"FAIL" y la presencia/ausencia de la barra de
#: progreso ya lo hacen por contenido -- el color es un refuerzo, no el
#: unico canal.
_COLOR_PASS = (60, 200, 60)
_COLOR_FAIL = (40, 40, 230)
#: Gris oscuro, no gris casi-blanco (fix de la demo de LinkedIn): el valor
#: anterior, (210, 210, 210), es casi indistinguible de la tarjeta blanca
#: sobre la que se grabo el demo real -- el halo negro de
#: ``_draw_outlined_text`` protege el BORDE del glifo, pero el interior del
#: trazo seguia siendo ese mismo gris casi-blanco sobre blanco. (90, 90, 90)
#: sostiene un contraste real contra blanco (igual que cualquier texto
#: "midiendo" sobre la tarjeta) sin tomar ningun tono de ``_COLOR_PASS``/
#: ``_COLOR_FAIL`` -- sigue siendo gris puro (R=G=B), asi que no compite
#: con el verde/rojo del veredicto. El color no es lo unico que distingue
#: "midiendo" de "fijo": el contenido ya lo hace (la barra de progreso y la
#: ausencia de OK/FAIL), este cambio solo corrige que el refuerzo de color
#: sea LEGIBLE, no que sea el unico canal.
#: Gris OSCURO, no claro. Se intento aclararlo al rediseñar la lectura y
#: ``test_unlocked_label_fill_has_real_contrast_on_a_white_card`` lo
#: rechazo con razon: el rotulo se ancla al lado de la mascara, o sea
#: sobre el CARTON BLANCO casi siempre, y un relleno claro ahi desaparece
#: -- el defecto que este mismo valor arreglo en su dia. El contraste
#: contra el amarillo del banano lo da el halo negro, no el relleno.
_COLOR_MEASURING = (90, 90, 90)

_MARKER_COLORS = {
    CalibrationState.OK: (60, 200, 60),
    CalibrationState.DEGRADED: (0, 165, 255),
    CalibrationState.BLOCKED: (40, 40, 230),
}

# ---------------------------------------------------------------------------
# Geometria del metodo (encargo posterior al plan de 20 tareas): contorno,
# eje medial, cuerda y curva Codex -- ver docstring de ``_draw_geometry``
# para la jerarquia visual completa.
#
# Jerarquia de PESO, no solo de color (restriccion del encargo: esto se
# dibuja sobre un banano amarillo saturado contra una tarjeta blanca, asi
# que el color SOLO no puede cargar la distincion):
#
#   contorno (1px, gris neutro)  <  eje medial (1px, acero)
#   <  cuerda (1px, acero, PUNTEADA)  <  curva Codex (3px + halo negro, celeste)
#
# La curva Codex es la PROTAGONISTA (asi lo pide el encargo explicitamente):
# es la unica con halo y la mas gruesa, y se dibuja ULTIMA (encima de las
# otras tres) en ``_draw_geometry``. El contorno y el eje comparten la
# region tonal "acero/gris" porque son evidencia de apoyo -- la FORMA ya los
# distingue (uno bordea la mascara, el otro la cruza por el centro), asi que
# no hace falta gastar un segundo tono saturado en diferenciarlos. La cuerda
# reusa el mismo acero que el eje (sus puntas SON las puntas del eje, Ruling
# de pruebas de esta tarea) pero PUNTEADA: el patron de guiones, no el
# color, es lo que la separa visualmente de la curva Codex solida.
_COLOR_CONTOUR = (130, 130, 130)
_CONTOUR_THICKNESS_PX = 1

_COLOR_AXIS = (150, 95, 60)
_AXIS_THICKNESS_PX = 1

_COLOR_CHORD = (150, 95, 60)
_CHORD_THICKNESS_PX = 1
_CHORD_DASH_PX = 14
_CHORD_GAP_PX = 9

#: Celeste vivo: no colisiona con el verde/rojo/naranja ya reservados para
#: veredicto/calibracion (``_COLOR_PASS``, ``_COLOR_FAIL``, ``_MARKER_COLORS``)
#: y contrasta con fuerza contra el amarillo de la fruta (complementario) y
#: contra la tarjeta blanca.
_COLOR_CODEX_CURVE = (225, 155, 15)
_CODEX_CURVE_THICKNESS_PX = 3
_CODEX_CURVE_HALO_THICKNESS_PX = 6


def text_anchor_for(mask: np.ndarray) -> tuple[int, int]:
    """Punto de anclaje (x, y) para el rotulo de un banano, derivado de su mascara.

    No existe bbox en ``BananaResult`` (Ruling 1: el pipeline no lo
    propaga a proposito, ver Task 14). Se deriva directamente de la
    extension real de la mascara booleana: por defecto, la esquina
    superior derecha de su caja delimitadora, mas un margen fijo para no
    superponerse al contorno del fruto.

    Recorte a los limites del lienzo (Finding 2, ronda 1 de revision)
    --------------------------------------------------------------------
    ``mask.shape`` ES la forma del lienzo rectificado (misma forma que
    ``canvas`` -- ver docstring de ``BananaResult.mask`` en
    ``pipeline.py``), asi que esta funcion ya tiene todo lo que necesita
    para no anclar fuera de el; no hace falta un parametro de lienzo
    aparte. Sin este recorte, un banano que una mano mueve hasta el borde
    del frame -- exactamente el momento que spec S13 vende como la
    publicacion del proyecto -- produce un ancla fuera de
    ``canvas.shape``; ``cv2.putText`` no lanza ahi, solo recorta en
    silencio, y la etiqueta entera desaparece justo cuando mas se esta
    mirando el video.

    Si el bloque no cabe a la derecha de la mascara, se VOLTEA al otro
    lado (se ancla a la izquierda del borde izquierdo de la mascara) en
    vez de limitarse a recortar contra el borde del lienzo -- asi el
    ancla sigue derivada de la extension de ESTA mascara (Ruling 4),
    nunca de un punto fijo del lienzo que no tenga relacion con donde
    esta el banano. El mismo recorte, sin volteo, se aplica en Y contra
    el borde inferior (no hay "arriba" y "abajo" del banano que tenga
    sentido voltear ahi; subir el bloque es suficiente).

    Cuando NI el volteo alcanza (fix de la demo de LinkedIn)
    ---------------------------------------------------------
    El volteo a la izquierda (``x_min - margen - bloque``) puede salir
    NEGATIVO sin que el lienzo sea mas angosto que el bloque: basta con
    que la MASCARA sea ancha -- un banano que cruza casi todo el ancho del
    lienzo, exactamente la postura que spec S13 vende en la demo, con
    ``x_min`` chico y ``x_max`` grande a la vez. La version anterior, en
    ese caso, SOBREESCRIBIA ``x`` con ese volteo negativo y dejaba que el
    clamp final de abajo lo aplastara a ``0`` -- el borde ABSOLUTO del
    lienzo, sin ninguna relacion con ``x_min`` real de la mascara (que
    podia estar a cientos de pixeles de ese borde). Visible en la
    grabacion de la demo: el rotulo "#1 / #### 30/30" flotando en la
    esquina superior izquierda, sobre tarjeta blanca vacia, lejos del
    banano.

    El fix: si el volteo a la izquierda tambien es negativo, ``x`` se deja
    en su valor SIN voltear (el intento original a la derecha de
    ``x_max``) en vez de adoptar el volteo roto -- el clamp final lo
    recorta entonces contra ``canvas_w - block_w``, que lo deja pegado al
    borde del lienzo mas CERCA de ``x_max`` (el lado derecho de la propia
    mascara), typicamente superpuesto a la parte derecha del banano en vez
    de flotando en una esquina sin relacion con el. Solo en el caso
    patologico de verdad -- el lienzo entero mas angosto que el bloque,
    "no esperado en el rig real, solo en fixtures de prueba sinteticos" --
    ese clamp final sigue cayendo en ``[0, canvas_w - block_w]`` como
    ultimo recurso genuino.

    Al moverse la mascara el ancla se traslada con ella exactamente lo
    mismo, salvo que el recorte de borde intervenga -- es la propiedad
    que hace que "los numeros sigan al objeto" (spec S8) sea cierto y no
    solo un efecto visual casual.

    Precondicion: ``mask`` tiene al menos un pixel en ``True``. El
    pipeline nunca produce un ``BananaResult`` con una mascara vacia
    (``refine_mask`` devuelve ``None`` antes de eso, Task 13); una mascara
    vacia aqui es un error de programacion aguas arriba, no un caso de
    frame en vivo, y se deja propagar la excepcion en vez de inventar un
    ancla sin sentido.
    """
    ys, xs = np.nonzero(mask)
    x_max = int(xs.max())
    x_min = int(xs.min())
    y_min = int(ys.min())
    canvas_h, canvas_w = mask.shape[:2]

    x = x_max + _LABEL_MARGIN_PX
    if x + _LABEL_BLOCK_WIDTH_PX > canvas_w:
        flipped_x = x_min - _LABEL_MARGIN_PX - _LABEL_BLOCK_WIDTH_PX
        if flipped_x >= 0:
            x = flipped_x
        # Si ``flipped_x`` tambien es negativo, ``x`` se queda en el
        # intento original (lado derecho, sin voltear) -- ver docstring
        # mas arriba (fix de la demo de LinkedIn): el clamp de abajo lo
        # deja pegado al borde del lienzo mas cerca de ``x_max``, no al
        # volteo roto.
    x = max(0, min(x, max(0, canvas_w - _LABEL_BLOCK_WIDTH_PX)))

    block_h = _LABEL_MAX_LINES * _LINE_HEIGHT_PX
    y = y_min + _LABEL_MARGIN_PX
    y = max(0, min(y, max(0, canvas_h - block_h)))

    return x, y


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


def _as_point_list(path_px: np.ndarray) -> np.ndarray | None:
    """Convierte ``path_px`` al formato ``(N, 1, 2)`` entero que piden
    ``cv2.polylines``/``cv2.drawContours``, o ``None`` si no hay al menos 2
    puntos (nada que trazar -- la geometria vacia por defecto de un
    ``BananaResult`` de prueba que no la necesita, Ruling de pruebas de esta
    tarea, cae exactamente aqui).
    """
    if path_px is None or len(path_px) < 2:
        return None
    return np.round(np.asarray(path_px, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)


def _draw_polyline(canvas: np.ndarray, path_px: np.ndarray, color: tuple[int, int, int], thickness: int, *, closed: bool) -> None:
    """Traza ``path_px`` (pixeles del lienzo, mismo formato que ``MedialAxis.path_px``)
    como una polilinea simple, sin halo. No hace nada si ``path_px`` no trae
    al menos 2 puntos.

    ``cv2.LINE_8`` (sin antialiasing), no ``cv2.LINE_AA``: a 1px de grosor
    -- el de las tres lineas de APOYO que usan este ayudante -- el
    antialiasing reparte la intensidad entre dos filas/columnas vecinas y
    ninguna de las dos llega al color pedido (medido: ~90% de intensidad
    en el canal mas fuerte sobre una linea horizontal perfecta). Eso
    ademas de verse mas tenue de lo pensado sobre el fondo bananero, hace
    que ninguna prueba pueda anclar un color exacto. ``cv2.LINE_8`` pinta
    el pixel exacto al color exacto, sin perder nitidez a 1px -- la misma
    convencion que ``draw_marker_overlay`` ya usaba.
    """
    pts = _as_point_list(path_px)
    if pts is None:
        return
    cv2.polylines(canvas, [pts], closed, color, thickness, cv2.LINE_8)


def _draw_haloed_polyline(canvas: np.ndarray, path_px: np.ndarray, color: tuple[int, int, int], thickness: int, halo_thickness: int, *, closed: bool) -> None:
    """Traza ``path_px`` con un halo negro por debajo -- misma tecnica que
    ``_draw_outlined_text``, para la curva PROTAGONISTA (ver docstring de
    ``_draw_geometry``). No hace nada si ``path_px`` no trae al menos 2
    puntos.

    Esta si usa ``cv2.LINE_AA`` (al reves que ``_draw_polyline``): al
    grosor de esta curva (``_CODEX_CURVE_THICKNESS_PX = 3``, halo 6) la fila
    central SI llega al color exacto pedido -- medido -- porque el trazo es
    lo bastante ancho para que el antialiasing quede confinado a sus dos
    bordes, no al centro. Vale la suavidad del trazo AA en la curva que mas
    se nota, sin pagar el costo de intensidad parcial que si paga un trazo
    de 1px.
    """
    pts = _as_point_list(path_px)
    if pts is None:
        return
    cv2.polylines(canvas, [pts], closed, (0, 0, 0), halo_thickness, cv2.LINE_AA)
    cv2.polylines(canvas, [pts], closed, color, thickness, cv2.LINE_AA)


def _draw_dashed_line(canvas: np.ndarray, point_a: np.ndarray, point_b: np.ndarray, color: tuple[int, int, int], thickness: int, dash_px: float, gap_px: float) -> None:
    """Traza un segmento PUNTEADO entre ``point_a`` y ``point_b``.

    ``cv2`` no trae una primitiva de linea discontinua; se recorre el
    segmento a pasos de ``dash_px``/``gap_px`` alternando trazo y hueco. El
    numero de pasos es ``longitud / dash_px`` -- unas pocas decenas para la
    cuerda tipica de un banano -- asi que el costo es insignificante frente
    al resto del dibujo del cuadro (ver reporte de la tarea para la medicion
    real). No hace nada si los dos puntos practicamente coinciden (cuerda
    degenerada: el sentinela de pruebas ``(np.zeros(2), np.zeros(2))`` cae
    aqui, a proposito, para que un ``BananaResult`` sin geometria no dibuje
    una cuerda fantasma en el origen del lienzo).
    """
    start = np.asarray(point_a, dtype=np.float64)
    end = np.asarray(point_b, dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        return
    direction = delta / length

    distance = 0.0
    drawing = True
    while distance < length:
        segment_len = dash_px if drawing else gap_px
        next_distance = min(distance + segment_len, length)
        if drawing:
            seg_start = start + direction * distance
            seg_end = start + direction * next_distance
            cv2.line(
                canvas,
                tuple(np.round(seg_start).astype(int)),
                tuple(np.round(seg_end).astype(int)),
                color,
                thickness,
                cv2.LINE_8,  # mismo motivo que ``_draw_polyline``: color exacto a 1px
            )
        distance = next_distance
        drawing = not drawing


def _chord_endpoints(result: BananaResult) -> tuple[np.ndarray, np.ndarray] | None:
    """Los dos extremos de la cuerda: los de la CURVA CODEX, no los del eje.

    Comparten extremos con el arco a proposito. Asi el hueco entre la recta
    a trazos y la curva es exactamente la curvatura, que es lo que el video
    quiere mostrar y lo que ``readout`` publica como porcentaje -- los dos
    leen la misma geometria. Tomarlos de ``axis_tips_px`` (la version
    anterior) daba una recta que arrancaba en el pedicelo mientras el arco
    arrancaba despues de el: las dos lineas no se podian comparar a ojo.

    ``None`` si no hay curva que acotar -- un ``BananaResult`` de prueba
    sin geometria.
    """
    curve = result.outer_curve_px
    if curve is None or len(curve) < 2:
        return None
    return np.asarray(curve[0], dtype=float), np.asarray(curve[-1], dtype=float)


def _draw_geometry(canvas: np.ndarray, result: BananaResult) -> None:
    """Dibuja la geometria del METODO para ``result``, sobre ``canvas`` (que
    ``draw_results`` ya sabe que es una copia).

    Encargo posterior al plan de 20 tareas: "que la persona que vea esto
    vea la implementacion, no que quede tras bambalinas" -- antes de esta
    funcion ``draw_results`` solo dibujaba texto, y el metodo entero
    (segmentacion, eje medial, curva Codex) era invisible en el video.

    Jerarquia visual, de MENOS a MAS protagonista (ver constantes de
    modulo para el porque de cada peso/color) -- y por eso se dibuja en
    ESE orden: cada capa queda por debajo de la siguiente, asi que la curva
    Codex, la mas importante, nunca queda tapada por una de apoyo:

    1. ``contour_px`` -- el contorno fino de la mascara (demuestra que la
       segmentacion es un borde preciso, no una caja).
    2. ``axis_path_px`` -- el eje medial completo (la linea de
       "adelgazamiento").
    3. ``axis_tips_px`` -- la cuerda punta a punta, PUNTEADA: el contraste
       directo contra la curva Codex, lo que vuelve visible en un segundo
       que el CODEX mide mas que la cuerda recta.
    4. ``outer_curve_px`` -- la curva Codex recortada al fruto, con halo:
       la PROTAGONISTA, literalmente la geometria detras de ``L`` en la
       etiqueta de este mismo resultado.

    Cualquiera de los cuatro puede venir vacio (un ``BananaResult`` de
    prueba sin geometria, o -- en produccion -- nunca, porque
    ``Pipeline.process`` siempre los puebla desde ``measure_mask``): cada
    ayudante de dibujo no hace nada ante una entrada vacia, asi que esta
    funcion no necesita chequear longitud ella misma.
    """
    _draw_polyline(canvas, result.contour_px, _COLOR_CONTOUR, _CONTOUR_THICKNESS_PX, closed=True)
    _draw_polyline(canvas, result.axis_path_px, _COLOR_AXIS, _AXIS_THICKNESS_PX, closed=False)
    chord = _chord_endpoints(result)
    if chord is not None:
        _draw_dashed_line(
            canvas, chord[0], chord[1], _COLOR_CHORD,
            _CHORD_THICKNESS_PX, _CHORD_DASH_PX, _CHORD_GAP_PX,
        )
    _draw_haloed_polyline(
        canvas,
        result.outer_curve_px,
        _COLOR_CODEX_CURVE,
        _CODEX_CURVE_THICKNESS_PX,
        _CODEX_CURVE_HALO_THICKNESS_PX,
        closed=False,
    )


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
    La geometria del metodo (``_draw_geometry``, encargo posterior al plan
    de 20 tareas) sigue la MISMA regla por la MISMA razon: cada contorno,
    eje y curva se dibuja con la mascara/geometria PROPIA de ``result``,
    nunca con la de otro elemento de la lista.

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
        _draw_geometry(out, result)
        _draw_outlined_text(
            out,
            compose_canvas_tag(result),
            text_anchor_for(result.mask),
            _tag_color(result),
        )
    return out


def _tag_color(result: BananaResult) -> tuple[int, int, int]:
    """Color del rotulo del lienzo. Es un REFUERZO, nunca el unico canal:
    el propio texto ya dice "PASS"/"FAIL" o el conteo de la ventana, asi
    que perder el color (escala de grises, daltonismo) no pierde nada."""
    if not result.aggregate.locked:
        return _COLOR_MEASURING
    return _COLOR_PASS if result.verdict.passes else _COLOR_FAIL


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
        _draw_marker_id(out, detection, color)
    return out


def _draw_marker_id(canvas: np.ndarray, detection: MarkerDetection, color: tuple[int, int, int]) -> None:
    """Escribe ``#id`` CENTRADO dentro del cuadrilatero del marcador.

    Antes se anclaba arriba a la izquierda y por FUERA del marcador
    (``min y - 6``). Los cuatro marcadores del rig viven en las esquinas
    del lienzo por construccion, asi que ese "por fuera" cae fuera del
    lienzo: en la grabacion de la demo dos de los cuatro ids no se veian.
    Dentro del cuadrilatero la etiqueta esta atada al marcador -- si el
    marcador se ve, el id se ve -- y ademas deja de poder confundirse con
    la del marcador de al lado.

    El marcador mide ~120 px de lado en el lienzo del rig y el texto a la
    escala de este modulo ronda los 30 px de alto, asi que cabe con
    holgura. Si no cupiera, el texto se saldria del cuadrilatero pero
    seguiria dentro del lienzo, que es el fallo benigno de los dos.
    """
    centre = detection.corners.mean(axis=0)
    text = f"#{detection.id}"
    (text_w, text_h), _baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _TEXT_THICKNESS)
    origin = (int(round(centre[0] - text_w / 2.0)), int(round(centre[1] + text_h / 2.0)))
    _draw_outlined_text(canvas, text, origin, color)

"""Tests del overlay de cotas sobre el frame (Task 17).

El brief de esta tarea tenia dos defectos de datos y una prueba imposible de
escribir; las correcciones (rulings del encargo) se verifican aqui, no solo
se mencionan:

1. **No hay bbox.** ``BananaResult`` (Task 14) solo trae ``mask`` -- el bbox
   vive en ``Instance`` (Task 12) y el pipeline no lo propaga a proposito.
   ``text_anchor_for`` deriva el ancla de la mascara booleana.

2. **Piso de incertidumbre.** ``aggregation.py`` calcula la desviacion con
   ``np.std`` sin ``ddof``, asi que la desviacion de UNA sola muestra es
   EXACTAMENTE 0.0. En el primer frame de cada track (el caso MAS comun,
   no un borde raro), ``length_std_mm``/``caliper_std_mm`` llegan en 0.0.
   Sin un piso, ``format_measurement`` imprimiria precision infinita justo
   donde el proyecto menos puede permitirselo (spec S4, anti-objetivo de
   falsa precision). El piso es ``mm_per_px``: spec S5.1 mide que una
   ambiguedad de borde de 1 px, a 0.2 mm/px, son 0.2 mm -- el sistema no
   puede reclamar mas resolucion que la de su propio lienzo.

3. **Sin OCR.** El brief pedia ``rendered_text_of(draw_results(...))``; leer
   texto de vuelta de una imagen rasterizada es OCR, no existe y no se
   puede escribir de forma fiable. ``compose_label_lines`` expone la
   composicion de texto como funcion PURA y estas pruebas verifican las
   cadenas, no pixeles interpretados.

4. **El test de posicion del brief no discrimina.** ``text_anchor_for(a) !=
   text_anchor_for(b)`` lo pasa cualquier funcion que varie con algo --
   incluida una que ancle a la esquina equivocada o al indice de la lista
   en vez de a la mascara. Las pruebas de aqui comprueban una propiedad
   mas fuerte: el ancla queda genuinamente DENTRO/cerca de la extension
   real de la mascara y se TRASLADA exactamente lo que se traslada la
   mascara.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from codex_caliper.aggregation import TEMPORAL_WINDOW, TrackAggregate
from codex_caliper.aruco_tools import CalibrationState, MarkerDetection
from codex_caliper.codex import CodexVerdict
from codex_caliper.pipeline import BananaResult
#: ``_LABEL_MARGIN_PX`` y ``_LABEL_BLOCK_WIDTH_PX`` se IMPORTAN del modulo
#: en vez de espejarse aqui. Espejarlas creaba una segunda fuente de verdad
#: que se desincronizaba en silencio: al subir la escala de fuente para que
#: el texto se leyera en video, el modulo paso a reservar 330 px de bloque y
#: las copias de este archivo seguian en 220, de modo que las pruebas de
#: anclaje comprobaban una geometria que ya no existia. Son privadas, si,
#: pero son LA geometria que estas pruebas afirman; afirmarla contra una
#: copia es no afirmarla.
from codex_caliper.ui.overlay import (
    _LABEL_BLOCK_WIDTH_PX,
    _LABEL_MARGIN_PX,
    draw_marker_overlay,
    draw_results,
    text_anchor_for,
)

#: Lienzo de pruebas, del orden del REAL (el rig da ~1241x745 px a 0.2
#: mm/px sobre la disposicion de marcadores). Antes era 500x300: mas
#: estrecho que el bloque de texto que el modulo reserva, asi que el
#: recorte contra el borde saltaba en casi todas las mascaras y tapaba la
#: propiedad que estas pruebas afirman -- el anclaje queda escondido
#: detras del clamp y el test deja de medir lo que dice medir. El clamp
#: tiene su propia prueba dedicada, con su propio lienzo estrecho.
_CANVAS_SHAPE = (400, 1200, 3)


def _rectangle_mask(shape: tuple[int, int], x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Mascara booleana rectangular, para no depender de datos sinteticos de otra tarea."""
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _aggregate(
    *,
    track_id: int = 1,
    samples: int = 30,
    locked: bool = True,
    length_mm: float = 187.34,
    length_std_mm: float = 0.82,
    caliper_mm: float = 38.4,
    caliper_std_mm: float = 0.6,
    chord_mm: float = 152.0,
    volume_mm3: float = 1.0e5,
    window: int = TEMPORAL_WINDOW,
) -> TrackAggregate:
    return TrackAggregate(
        track_id=track_id,
        samples=samples,
        locked=locked,
        length_mm=length_mm,
        length_std_mm=length_std_mm,
        caliper_mm=caliper_mm,
        caliper_std_mm=caliper_std_mm,
        chord_mm=chord_mm,
        volume_mm3=volume_mm3,
        window=window,
    )


def _verdict(passes: bool = True) -> CodexVerdict:
    return CodexVerdict(
        passes=passes, length_ok=passes, caliper_ok=passes, commercial_grade=38 if passes else None
    )


def _result(
    mask: np.ndarray,
    *,
    track_id: int = 1,
    aggregate: TrackAggregate | None = None,
    verdict: CodexVerdict | None = None,
    weight_g: float | None = 124.0,
    contour_px: np.ndarray | None = None,
    axis_path_px: np.ndarray | None = None,
    axis_tips_px: tuple[np.ndarray, np.ndarray] | None = None,
    outer_curve_px: np.ndarray | None = None,
) -> BananaResult:
    """``contour_px``/``axis_path_px``/``axis_tips_px``/``outer_curve_px``
    (encargo posterior al plan de 20 tareas) tienen default ``None`` aqui y
    se traducen a sentinelas VACIOS -- no al pipeline real: estos tests
    construyen un resultado para probar el TEXTO del rotulo, no la
    geometria del overlay, asi que la mayoria no les pasa nada. Los que si
    les importa la geometria (ver mas abajo) los pasan explicitos."""
    empty_path = np.empty((0, 2), dtype=np.float64)
    return BananaResult(
        track_id=track_id,
        aggregate=aggregate if aggregate is not None else _aggregate(track_id=track_id),
        verdict=verdict if verdict is not None else _verdict(),
        estimated_weight_g=weight_g,
        mask=mask,
        contour_px=contour_px if contour_px is not None else empty_path,
        axis_path_px=axis_path_px if axis_path_px is not None else empty_path,
        axis_tips_px=axis_tips_px if axis_tips_px is not None else (np.zeros(2), np.zeros(2)),
        outer_curve_px=outer_curve_px if outer_curve_px is not None else empty_path,
    )


# ---------------------------------------------------------------------------
# text_anchor_for -- Ruling 1 (deriva de la mascara, no de un bbox que no
# existe) y Ruling 4 (el test del brief no discrimina).
# ---------------------------------------------------------------------------


def test_anchor_sits_outside_the_masks_right_edge_and_near_its_top():
    mask = _rectangle_mask(_CANVAS_SHAPE[:2], 30, 40, 70, 120)
    ax, ay = text_anchor_for(mask)
    ys, xs = np.nonzero(mask)
    assert ax > int(xs.max())
    assert int(ys.min()) <= ay <= int(ys.min()) + 40


def test_anchor_translates_exactly_with_the_mask():
    # la propiedad fuerte que el test del brief no verificaba: un ancla que
    # dependiera del indice en la lista, o de la esquina equivocada, no
    # pasaria esto para un desplazamiento arbitrario.
    base = _rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 90, 150)
    shifted = _rectangle_mask(_CANVAS_SHAPE[:2], 130, 70, 170, 170)
    ax, ay = text_anchor_for(base)
    bx, by = text_anchor_for(shifted)
    assert (bx - ax, by - ay) == (80, 20)


def test_labels_follow_the_banana_position():
    # version corregida del nombre del brief: la propiedad real es que dos
    # posiciones distintas producen anclas distintas Y consistentes con la
    # extension de cada mascara (no solo "distintas", que cualquier cosa
    # que varie de cualquier forma ya cumpliria).
    at_left = _rectangle_mask(_CANVAS_SHAPE[:2], 20, 50, 60, 150)
    at_right = _rectangle_mask(_CANVAS_SHAPE[:2], 380, 50, 420, 150)
    left_anchor = text_anchor_for(at_left)
    right_anchor = text_anchor_for(at_right)
    assert left_anchor != right_anchor
    assert right_anchor[0] > left_anchor[0]


def test_anchor_stays_within_canvas_bounds_near_the_right_edge():
    # Finding 2, ronda 1 de revision: una mascara que toca la ULTIMA
    # columna del lienzo (spec S13 -- la mano entra por el borde del
    # frame). Sin recorte, x_max + margen cae en o mas alla de
    # canvas_w, y cv2.putText no lanza ahi -- solo recorta en silencio,
    # asi que la etiqueta entera desaparece. El ancla tiene que quedar
    # DENTRO del lienzo sin importar donde este el banano.
    canvas_shape = _CANVAS_SHAPE[:2]
    mask = _rectangle_mask(canvas_shape, 450, 120, 500, 180)
    ax, ay = text_anchor_for(mask)
    assert 0 <= ax < canvas_shape[1]
    assert 0 <= ay < canvas_shape[0]


def test_anchor_stays_within_canvas_bounds_near_the_bottom_edge():
    # La misma proteccion, eje Y: una mascara que toca la ULTIMA fila.
    canvas_shape = _CANVAS_SHAPE[:2]
    mask = _rectangle_mask(canvas_shape, 200, 290, 240, 300)
    ax, ay = text_anchor_for(mask)
    assert 0 <= ax < canvas_shape[1]
    assert 0 <= ay < canvas_shape[0]


def test_anchor_sticks_to_the_masks_right_edge_when_the_mask_spans_nearly_the_full_canvas():
    """Fix 3 (demo de LinkedIn): con una mascara que cubre casi todo el
    ancho del lienzo, NI el lado derecho (``x_max + margen + bloque`` se
    sale del lienzo) NI el volteo a la izquierda (``x_min - margen -
    bloque`` es NEGATIVO) caben enteros dentro de ``canvas_w``.

    La implementacion ANTERIOR, en ese caso, adoptaba el volteo negativo y
    dejaba que el clamp final lo aplastara a ``0`` -- el borde ABSOLUTO
    del lienzo, sin ninguna relacion con ``x_min`` real (``50`` en este
    fixture, lejos de ``0``). Es exactamente lo que se vio en la grabacion
    de la demo: el rotulo flotando en la esquina superior izquierda, sobre
    tarjeta blanca vacia, lejos del banano -- porque el banano de esa
    toma cruzaba casi todo el ancho del cuadro.

    Con el fix, cuando el volteo tambien falla, el intento ORIGINAL (lado
    derecho de ``x_max``) sobrevive y el clamp final lo pega al borde del
    lienzo mas CERCA de ``x_max`` -- en este fixture, exactamente
    ``canvas_w - block_w = 1500 - 220 = 1280``, pegado al extremo derecho
    de la propia mascara (``x_max = 1449``), no al origen del lienzo.

    La implementacion vieja que esta prueba RECHAZA: cualquiera que
    devuelva ``ax == 0`` (o cualquier valor cercano a 0) para este
    fixture en vez de ``1280``.
    """
    canvas_shape = (300, 1500)
    mask = _rectangle_mask(canvas_shape, 50, 120, 1450, 180)
    ax, ay = text_anchor_for(mask)

    expected_ax = canvas_shape[1] - _LABEL_BLOCK_WIDTH_PX  # 1280
    assert ax == expected_ax, f"el ancla colapso a {ax} en vez de pegarse al borde derecho de la mascara"
    assert ax != 0, "el ancla cayo en el borde absoluto del lienzo -- el bug viejo exacto"


# ---------------------------------------------------------------------------
# draw_results -- coloca las etiquetas sobre una COPIA del lienzo.
# ---------------------------------------------------------------------------


def test_draw_results_does_not_mutate_the_input_canvas():
    canvas = np.zeros(_CANVAS_SHAPE, np.uint8)
    before = canvas.copy()
    mask = _rectangle_mask(_CANVAS_SHAPE[:2], 50, 100, 150, 200)
    draw_results(canvas, [_result(mask)], 0.2)
    assert np.array_equal(canvas, before)


def test_draw_results_paints_near_each_results_own_anchor_regardless_of_list_order():
    # el banano de la derecha va PRIMERO en la lista y el de la izquierda
    # SEGUNDO -- a proposito, para que un bug que ancle por indice de lista
    # (en vez de por la mascara de cada resultado) no pueda esconderse
    # detras de un orden casual que coincida con la posicion espacial.
    #
    # Lienzo LOCAL mas ancho que el del resto del archivo (no el
    # _CANVAS_SHAPE compartido): con el recorte de borde del Finding 2
    # (ronda 1 de revision), un mask_right demasiado cerca del borde
    # derecho de un lienzo angosto hace que su ancla se VOLTEE al otro
    # lado y termine pegada a la del mask_left -- una colision de fixture
    # con el recorte de borde, no un bug de indice de lista, que es lo
    # que este test en particular quiere aislar. Un lienzo mas ancho deja
    # a ambas mascaras lejos de cualquier borde y evita esa colision.
    # Se deriva del presupuesto del propio modulo en vez de fijar 900 a
    # mano: al subir la escala de fuente para que el texto se lea en video
    # el bloque paso a 330 px, y con 900 la mascara derecha (acaba en
    # x=640) ya no cabia -- 640 + margen + bloque se sale del lienzo --
    # asi que su ancla se volteaba a la izquierda y aterrizaba sobre la
    # del mask_left: exactamente la colision de fixture que el parrafo de
    # arriba describe. Derivarlo evita que vuelva a quedarse corto en
    # silencio la proxima vez que cambie la tipografia.
    wide_shape = (300, 640 + _LABEL_MARGIN_PX + _LABEL_BLOCK_WIDTH_PX + 60, 3)
    canvas = np.zeros(wide_shape, np.uint8)
    mask_left = _rectangle_mask(wide_shape[:2], 20, 50, 60, 150)
    mask_right = _rectangle_mask(wide_shape[:2], 600, 50, 640, 150)
    left_result = _result(mask_left, track_id=1)
    right_result = _result(mask_right, track_id=2)

    out = draw_results(canvas, [right_result, left_result], 0.2)

    for mask in (mask_left, mask_right):
        ax, ay = text_anchor_for(mask)
        region = out[max(ay - 5, 0) : ay + 70, max(ax - 5, 0) : ax + 220]
        assert region.max() > 0, "no se dibujo nada cerca del ancla propia de esta mascara"

    # La comprobacion de arriba sola NO discrimina: un bug que baraje que
    # ancla le toca a cada resultado (p.ej. ``anchors[idx - 1]`` en vez de
    # ``anchors[idx]``) sigue pintando ALGO cerca de cada ancla -- solo que
    # es la etiqueta del OTRO banano. Dos track_id distintos (1 vs 2)
    # producen texto distinto ("#1 ..." vs "#2 ..."); se renderiza cada
    # resultado SOLO (sin el otro) para saber como se ve SU etiqueta en SU
    # propio ancla, y se compara pixel a pixel contra la region
    # correspondiente del render conjunto. No es OCR: no se interpreta
    # ningun pixel, solo se compara un recorte de imagen contra otro.
    solo_left = draw_results(canvas, [left_result], 0.2)
    solo_right = draw_results(canvas, [right_result], 0.2)
    ax, ay = text_anchor_for(mask_left)
    bx, by = text_anchor_for(mask_right)
    region_left_together = out[max(ay - 5, 0) : ay + 70, max(ax - 5, 0) : ax + 220]
    region_left_solo = solo_left[max(ay - 5, 0) : ay + 70, max(ax - 5, 0) : ax + 220]
    region_right_together = out[max(by - 5, 0) : by + 70, max(bx - 5, 0) : bx + 220]
    region_right_solo = solo_right[max(by - 5, 0) : by + 70, max(bx - 5, 0) : bx + 220]
    assert np.array_equal(region_left_together, region_left_solo), (
        "la etiqueta junto al ancla izquierdo no coincide con la etiqueta propia de ese banano"
    )
    assert np.array_equal(region_right_together, region_right_solo), (
        "la etiqueta junto al ancla derecho no coincide con la etiqueta propia de ese banano"
    )


def test_draw_results_with_no_results_returns_an_unchanged_copy():
    canvas = np.zeros(_CANVAS_SHAPE, np.uint8)
    out = draw_results(canvas, [], 0.2)
    assert np.array_equal(out, canvas)
    assert out is not canvas


def test_draw_results_keeps_the_label_visible_when_the_mask_touches_the_right_edge():
    # Finding 2, ronda 1 de revision. La prueba fuerte que se pidio: no
    # solo que una aserción falle, sino que la etiqueta DESAPAREZCA de
    # verdad sin el recorte. Esta mascara toca la ultima columna del
    # lienzo (450..500 de un canvas de 500 de ancho); sin el recorte de
    # text_anchor_for, el ancla x (x_max + margen = 509) cae fuera de
    # canvas.shape[1] = 500 para las CUATRO lineas del rotulo por igual
    # (misma x en todas), asi que cv2.putText no dibuja ni un pixel.
    canvas = np.zeros(_CANVAS_SHAPE, np.uint8)
    mask = _rectangle_mask(_CANVAS_SHAPE[:2], 450, 120, 500, 180)
    out = draw_results(canvas, [_result(mask)], 0.2)
    assert out.sum() > 0, "la etiqueta desaparecio por completo: el ancla cayo fuera del lienzo"


def test_draw_results_keeps_the_label_visible_when_the_mask_touches_the_bottom_edge():
    # Misma proteccion, eje Y: mascara que toca la ultima fila
    # (290..300 de un canvas de 300 de alto). Sin el recorte, el ancla y
    # (y_min + margen = 300) cae exactamente en canvas.shape[0] = 300 --
    # fuera de rango para un array de 300 filas.
    #
    # ``out.sum() > 0`` NO discrimina aqui: incluso sin recorte, la
    # PRIMERA linea todavia deja un borrón de tinta en las ultimas filas
    # del lienzo (el glifo se dibuja hacia ARRIBA desde su linea base, y
    # esa linea base cae apenas 10 px mas alla del borde -- suficiente
    # para que una esquina del glifo siga cayendo dentro). Lo que
    # realmente desaparece son las OTRAS TRES lineas (CAL, L y el peso),
    # cuyo origen cae aun mas abajo, totalmente fuera de rango. Medido
    # sin el recorte: la tinta
    # ocupa unicamente las filas [289, 299] (un borrón de 11 filas de la
    # primera linea); CON el recorte ocupa [217, 285] (51 filas, las
    # cuatro lineas completas, ver reporte para los numeros exactos). La
    # aserción comprueba que hay tinta bien por encima de donde cae ese
    # borrón aislado -- una zona que el bug nunca alcanza.
    canvas = np.zeros(_CANVAS_SHAPE, np.uint8)
    mask = _rectangle_mask(_CANVAS_SHAPE[:2], 200, 290, 240, 300)
    out = draw_results(canvas, [_result(mask)], 0.2)
    assert out[:280].sum() > 0, (
        "solo sobrevive un borron aislado de la primera linea cerca del borde; "
        "el resto del rotulo (CAL / L / peso) desaparecio"
    )


# ---------------------------------------------------------------------------
# Geometria del metodo (encargo posterior al plan de 20 tareas): contorno,
# eje medial, cuerda y curva Codex. Las coordenadas de abajo son todas
# HORIZONTALES/VERTICALES a proposito: ``_draw_geometry`` dibuja las tres
# lineas de apoyo (contorno/eje/cuerda) con ``cv2.LINE_8`` -- sin
# antialiasing -- precisamente para que un pixel sobre la linea tenga el
# color EXACTO pedido, verificable sin tolerancia (ver docstring de
# ``_draw_polyline`` en ``overlay.py``); una linea diagonal seguiria siendo
# un color solido con ``LINE_8``, pero elegir horizontal/vertical hace que
# la aritmetica de "que pixel cae sobre la linea" sea trivial de leer en
# esta prueba.
# ---------------------------------------------------------------------------

_COLOR_CONTOUR = (130, 130, 130)
_COLOR_AXIS_AND_CHORD = (150, 95, 60)
_COLOR_CODEX_CURVE = (225, 155, 15)


def test_contour_is_a_thin_border_not_a_filled_box():
    """Ruling 1 del encargo original ("no hay bbox") aplicado a la geometria
    nueva: ``contour_px`` tiene que trazarse como un borde FINO, no como una
    caja rellena. Una implementacion que confundiera "dibujar el contorno"
    con "pintar su interior" (p.ej. ``cv2.fillPoly`` en vez de
    ``cv2.polylines``) pintaria el centro del rectangulo igual que el
    borde; esta prueba lo rechaza comprobando AMBOS: borde pintado, centro
    sin pintar.
    """
    canvas = np.zeros(_CANVAS_SHAPE, np.uint8)
    rectangle = np.array([[50.0, 50.0], [250.0, 50.0], [250.0, 150.0], [50.0, 150.0]])
    mask = _rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 250, 150)
    result = _result(mask, contour_px=rectangle)

    out = draw_results(canvas, [result], 0.2)

    assert tuple(out[50, 150]) == _COLOR_CONTOUR, "el borde superior del contorno no se dibujo"
    assert tuple(out[150, 50]) == _COLOR_CONTOUR, "el borde izquierdo del contorno no se dibujo"
    assert tuple(out[100, 150]) == (0, 0, 0), "el interior del contorno quedo pintado -- es una caja, no un borde"


def test_the_chord_connects_the_codex_curve_endpoints_not_the_medial_axis_tips():
    """La cuerda a trazos y la curva Codex tienen que COMPARTIR extremos.

    Es lo que hace legible el contraste que el video vende: con los mismos
    dos extremos, el hueco entre la recta y el arco ES la curvatura, y se
    lee de un vistazo. Antes la cuerda unia las puntas del EJE MEDIAL
    (pedicelo incluido) mientras el arco estaba recortado al fruto: las
    dos lineas arrancaban en sitios distintos y la comparacion no se
    entendia -- visible en la grabacion de la demo.

    Rechaza la implementacion anterior, que leia ``axis_tips_px``: el
    fixture pone esas puntas en una FILA DISTINTA de los extremos de la
    curva, asi que dibujar desde ellas pinta donde este test exige fondo.
    """
    canvas_shape = (300, 400, 3)
    canvas = np.zeros(canvas_shape, np.uint8)
    mask = _rectangle_mask(canvas_shape[:2], 40, 120, 360, 180)

    # Curva Codex horizontal en la fila 150, de x=60 a x=340.
    outer_curve = np.array([[x, 150] for x in range(60, 341, 10)], dtype=float)
    # Puntas del eje medial a OTRA altura (fila 200) y mas abiertas: si la
    # cuerda se dibujara desde aqui, caeria en esa fila.
    tips = (np.array([20.0, 200.0]), np.array([380.0, 200.0]))

    out = draw_results(
        canvas,
        [_result(mask, axis_tips_px=tips, outer_curve_px=outer_curve)],
        mm_per_px=0.2,
    )

    # A mitad de camino entre los extremos de la curva (x=200) la cuerda
    # cae sobre la fila 150, encima de la propia curva. Se mira un punto
    # donde la curva NO esta: la cuerda entre los extremos del arco es el
    # propio segmento, asi que se comprueba por la NEGATIVA -- la fila 200,
    # donde la implementacion anterior la habria puesto, tiene que estar
    # limpia.
    assert not out[196:205, 150:250].any(), (
        "se dibujo una cuerda en la fila de las puntas del eje medial"
    )
    # Y el segmento SI esta en la fila del arco: hay tinta de cuerda o de
    # arco a lo largo de toda su extension.
    assert out[146:155, 60:341].any()

def test_codex_curve_is_drawn_on_top_of_the_contour_at_a_shared_pixel():
    """Jerarquia visual (encargo: "la curva Codex es la protagonista, no
    cuatro lineas de igual peso"): en un pixel donde el contorno y la
    curva Codex se CRUZAN a proposito, el pixel final tiene que mostrar el
    color de la curva Codex, no el del contorno -- la protagonista se
    dibuja AL FINAL, encima de las lineas de apoyo. Si el orden de dibujo
    se invirtiera (contorno despues de la curva), este pixel mostraria
    ``_COLOR_CONTOUR`` en vez de ``_COLOR_CODEX_CURVE``.
    """
    canvas = np.zeros(_CANVAS_SHAPE, np.uint8)
    rectangle = np.array([[50.0, 50.0], [250.0, 50.0], [250.0, 150.0], [50.0, 150.0]])
    # Cruza el borde superior del rectangulo (y=50) en x=150.
    crossing_curve = np.array([[150.0, 20.0], [150.0, 80.0]])
    mask = _rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 250, 150)
    result = _result(mask, contour_px=rectangle, outer_curve_px=crossing_curve)

    out = draw_results(canvas, [result], 0.2)

    assert tuple(out[50, 150]) == _COLOR_CODEX_CURVE, (
        "en el pixel donde contorno y curva Codex se cruzan debe ganar la curva Codex (protagonista)"
    )


def test_each_result_draws_only_its_own_geometry_not_its_neighbors():
    # Mismo espiritu que test_draw_results_paints_near_each_results_own_anchor_regardless_of_list_order
    # (Ruling 4 del encargo original), aplicado a la geometria nueva en vez
    # del texto: dos frutos, con geometria bien separada y EN ORDEN
    # INVERTIDO a su posicion espacial, para que un bug que mezclara la
    # geometria de un resultado con la del otro (p.ej. por indice de lista
    # en vez de por el propio ``BananaResult``) no pueda esconderse detras
    # de un orden casual.
    # Se deriva del presupuesto del propio modulo en vez de fijar 900 a
    # mano: al subir la escala de fuente para que el texto se lea en video
    # el bloque paso a 330 px, y con 900 la mascara derecha (acaba en
    # x=640) ya no cabia -- 640 + margen + bloque se sale del lienzo --
    # asi que su ancla se volteaba a la izquierda y aterrizaba sobre la
    # del mask_left: exactamente la colision de fixture que el parrafo de
    # arriba describe. Derivarlo evita que vuelva a quedarse corto en
    # silencio la proxima vez que cambie la tipografia.
    wide_shape = (300, 640 + _LABEL_MARGIN_PX + _LABEL_BLOCK_WIDTH_PX + 60, 3)
    canvas = np.zeros(wide_shape, np.uint8)

    mask_left = _rectangle_mask(wide_shape[:2], 20, 50, 60, 150)
    curve_left = np.array([[20.0, 200.0], [60.0, 200.0]])

    mask_right = _rectangle_mask(wide_shape[:2], 600, 50, 640, 150)
    curve_right = np.array([[600.0, 220.0], [640.0, 220.0]])

    left_result = _result(mask_left, track_id=1, outer_curve_px=curve_left)
    right_result = _result(mask_right, track_id=2, outer_curve_px=curve_right)

    out = draw_results(canvas, [right_result, left_result], 0.2)
    solo_left = draw_results(canvas, [left_result], 0.2)
    solo_right = draw_results(canvas, [right_result], 0.2)

    # Cada curva aparece en SU propia fila, y NO en la fila del otro fruto.
    assert tuple(out[200, 40]) == _COLOR_CODEX_CURVE
    assert tuple(out[220, 620]) == _COLOR_CODEX_CURVE
    assert tuple(out[220, 40]) == (0, 0, 0), "la curva del fruto derecho se filtro a la fila del fruto izquierdo"
    assert tuple(out[200, 620]) == (0, 0, 0), "la curva del fruto izquierdo se filtro a la fila del fruto derecho"

    # Y el render conjunto coincide, pixel a pixel, con el render SOLO de
    # cada fruto en su propia region -- ninguno de los dos quedo pintado
    # con la curva ajena.
    assert np.array_equal(out[180:240, 0:100], solo_left[180:240, 0:100])
    assert np.array_equal(out[180:240, 580:680], solo_right[180:240, 580:680])


def test_draw_results_does_not_mutate_the_geometry_arrays_either():
    # La proteccion de "nunca mutar" del encargo original (Ruling del
    # lienzo) se extiende a la geometria nueva: ni ``_draw_geometry`` ni
    # ninguno de sus ayudantes puede modificar los arreglos que
    # ``BananaResult`` trae, porque son los MISMOS arreglos que
    # ``measure_mask`` calculo y que ``correct_measures``/el pipeline
    # siguen sosteniendo para el resto del cuadro.
    canvas = np.zeros(_CANVAS_SHAPE, np.uint8)
    contour = np.array([[50.0, 50.0], [250.0, 50.0], [250.0, 150.0], [50.0, 150.0]])
    axis_path = np.array([[50.0, 100.0], [150.0, 60.0], [250.0, 100.0]])
    tips = (np.array([10.0, 150.0]), np.array([290.0, 150.0]))
    outer_curve = np.array([[150.0, 20.0], [150.0, 80.0]])
    contour_before, axis_before, outer_before = contour.copy(), axis_path.copy(), outer_curve.copy()
    tips_before = (tips[0].copy(), tips[1].copy())

    mask = _rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 250, 150)
    result = _result(
        mask, contour_px=contour, axis_path_px=axis_path, axis_tips_px=tips, outer_curve_px=outer_curve
    )
    draw_results(canvas, [result], 0.2)

    assert np.array_equal(contour, contour_before)
    assert np.array_equal(axis_path, axis_before)
    assert np.array_equal(tips[0], tips_before[0]) and np.array_equal(tips[1], tips_before[1])
    assert np.array_equal(outer_curve, outer_before)


# ---------------------------------------------------------------------------
# draw_marker_overlay -- el estado de calibracion se deriva del CONTEO de
# detecciones, igual que ``aruco_tools.calibration_state``; no hace falta
# un parametro extra para saber si esta OK/DEGRADED/BLOCKED.
# ---------------------------------------------------------------------------


def _marker(id_: int, cx: float, cy: float, half: float = 15.0) -> MarkerDetection:
    corners = np.array(
        [[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]],
        dtype=np.float32,
    )
    return MarkerDetection(id=id_, corners=corners)


def test_draw_marker_overlay_does_not_mutate_the_input_frame():
    frame = np.zeros((200, 200, 3), np.uint8)
    before = frame.copy()
    draw_marker_overlay(frame, [_marker(1, 50, 50)])
    assert np.array_equal(frame, before)


def test_draw_marker_overlay_draws_near_every_detection():
    frame = np.zeros((200, 200, 3), np.uint8)
    detections = [_marker(1, 40, 40), _marker(2, 160, 160)]
    out = draw_marker_overlay(frame, detections)
    for detection in detections:
        x0, y0 = detection.corners.min(axis=0).astype(int)
        x1, y1 = detection.corners.max(axis=0).astype(int)
        region = out[max(y0 - 5, 0) : y1 + 25, max(x0 - 5, 0) : x1 + 25]
        assert region.max() > 0


def test_draw_marker_overlay_handles_zero_detections_without_raising():
    frame = np.zeros((200, 200, 3), np.uint8)
    out = draw_marker_overlay(frame, [])
    assert np.array_equal(out, frame)


# ---------------------------------------------------------------------------
# Finding 3 (promovido, ronda 1 de revision): el mapeo de color de
# draw_marker_overlay es la UNICA expresion en la interfaz del estado
# DEGRADED que spec S7 etapa 2 exige mostrar, y no tenia ninguna
# aseveracion sobre que color se elige para 4 / 3 / menos marcadores.
# Se comprueba el color de verdad presente en los pixeles dibujados (los
# contornos de cv2.polylines son de color solido, sin anti-aliasing) --
# no es OCR, es una comparacion de color exacta.
# ---------------------------------------------------------------------------

_COLOR_GREEN_OK = (60, 200, 60)
_COLOR_ORANGE_DEGRADED = (0, 165, 255)
_COLOR_RED_BLOCKED = (40, 40, 230)


def _has_pixel_of_color(image: np.ndarray, color: tuple[int, int, int]) -> bool:
    return bool(np.any(np.all(image == np.array(color, dtype=image.dtype), axis=-1)))


def test_draw_marker_overlay_colors_green_with_all_four_markers():
    frame = np.zeros((200, 200, 3), np.uint8)
    detections = [_marker(i, 30 + 40 * i, 30 + 40 * i) for i in range(1, 5)]
    out = draw_marker_overlay(frame, detections)
    assert _has_pixel_of_color(out, _COLOR_GREEN_OK)
    assert not _has_pixel_of_color(out, _COLOR_ORANGE_DEGRADED)
    assert not _has_pixel_of_color(out, _COLOR_RED_BLOCKED)


def test_draw_marker_overlay_colors_orange_with_exactly_three_markers():
    frame = np.zeros((200, 200, 3), np.uint8)
    detections = [_marker(i, 30 + 40 * i, 30 + 40 * i) for i in range(1, 4)]
    out = draw_marker_overlay(frame, detections)
    assert _has_pixel_of_color(out, _COLOR_ORANGE_DEGRADED)
    assert not _has_pixel_of_color(out, _COLOR_GREEN_OK)
    assert not _has_pixel_of_color(out, _COLOR_RED_BLOCKED)


def test_draw_marker_overlay_colors_red_with_fewer_than_three_markers():
    frame = np.zeros((200, 200, 3), np.uint8)
    detections = [_marker(1, 50, 50), _marker(2, 100, 100)]
    out = draw_marker_overlay(frame, detections)
    assert _has_pixel_of_color(out, _COLOR_RED_BLOCKED)
    assert not _has_pixel_of_color(out, _COLOR_GREEN_OK)
    assert not _has_pixel_of_color(out, _COLOR_ORANGE_DEGRADED)


# ---------------------------------------------------------------------------
# Fix 2 (demo de LinkedIn): el relleno del rotulo "midiendo" (``locked``
# ``False``) tiene que sostener contraste real sobre una tarjeta blanca --
# no basta con "algo de color", el halo negro de ``_draw_outlined_text``
# protege el BORDE del glifo pero el interior del trazo sigue siendo el
# color de relleno a secas. El valor viejo, (210, 210, 210), es casi
# indistinguible de blanco; se fija aqui el valor nuevo real, no solo "es
# mas oscuro que antes".
# ---------------------------------------------------------------------------

_COLOR_MEASURING_FILL = (90, 90, 90)
_COLOR_LOCKED_PASS = (60, 200, 60)
_COLOR_LOCKED_FAIL = (40, 40, 230)

#: Piso de conteo, no "al menos un pixel" (ver docstring de la prueba de
#: abajo: con el relleno VIEJO, (210, 210, 210), el halo negro
#: anti-aliased sobre blanco deja por pura coincidencia aritmetica
#: exactamente 1 pixel en (90, 90, 90) -- medido al escribir esta prueba.
#: El relleno nuevo, en el mismo render, deja 229. El piso de 20 separa
#: con margen amplio "es el relleno real" de "una coincidencia de
#: anti-aliasing en el borde del halo".
_MEASURING_FILL_PIXEL_FLOOR = 20


def _count_pixels_of_color(image: np.ndarray, color: tuple[int, int, int]) -> int:
    return int(np.sum(np.all(image == np.array(color, dtype=image.dtype), axis=-1)))


def test_unlocked_label_fill_has_real_contrast_on_a_white_card():
    """La implementacion equivocada que esta prueba RECHAZA: el
    ``_COLOR_MEASURING`` ANTERIOR, (210, 210, 210) -- casi blanco sobre
    blanco, exactamente el defecto que la captura de pantalla del demo
    mostro.

    Se cuenta, no solo se comprueba presencia: con el relleno viejo el
    render de este mismo fixture deja exactamente 1 pixel en (90, 90, 90)
    -- una coincidencia aritmetica del halo negro anti-aliased sobre el
    lienzo blanco, SIN relacion con ``_COLOR_MEASURING`` -- mientras que
    con el relleno nuevo deja 229. Una aserción de "algun pixel" (``any``)
    la pasaria igual con el valor viejo por esa UNICA coincidencia; el
    piso (``_MEASURING_FILL_PIXEL_FLOOR``, bien por debajo de 229 y bien
    por encima de 1) es lo que de verdad discrimina. Se dibuja sobre un
    lienzo BLANCO (255, 255, 255), no negro, porque el defecto es
    especificamente "gris claro sobre tarjeta clara" -- un lienzo negro no
    lo habria mostrado nunca.
    """
    white_card = np.full(_CANVAS_SHAPE, 255, np.uint8)
    unlocked = _result(
        _rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 90, 150),
        aggregate=_aggregate(locked=False, samples=19),
    )

    out = draw_results(white_card, [unlocked], 0.2)

    count = _count_pixels_of_color(out, _COLOR_MEASURING_FILL)
    assert count >= _MEASURING_FILL_PIXEL_FLOOR, (
        f"solo {count} pixeles del relleno oscuro nuevo -- "
        "demasiado pocos para ser el relleno real del rotulo, no una coincidencia de borde"
    )


def test_locked_label_colors_are_unchanged_by_the_unlocked_fill_fix():
    """Red de seguridad: el fix de contraste de arriba toca SOLO el
    relleno de ``locked is False``. Una implementacion que, al oscurecer
    ese color, hubiera tocado por error ``_COLOR_PASS``/``_COLOR_FAIL``
    (p.ej. compartiendo una sola constante para los tres estados) se
    atrapa aqui: el veredicto OK sigue en verde y FAIL sigue en rojo,
    exactamente como antes del fix.
    """
    white_card = np.full(_CANVAS_SHAPE, 255, np.uint8)
    passing = _result(_rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 90, 150), verdict=_verdict(passes=True))
    failing = _result(_rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 90, 150), verdict=_verdict(passes=False))

    out_pass = draw_results(white_card, [passing], 0.2)
    out_fail = draw_results(white_card, [failing], 0.2)

    assert _has_pixel_of_color(out_pass, _COLOR_LOCKED_PASS)
    assert _has_pixel_of_color(out_fail, _COLOR_LOCKED_FAIL)


def test_marker_id_label_is_drawn_inside_the_marker_so_the_canvas_edge_cannot_clip_it():
    """El id de un marcador pegado al borde del lienzo tiene que seguir
    siendo legible.

    Los 4 marcadores del rig viven en las ESQUINAS del lienzo por
    construccion, asi que una etiqueta colocada por fuera del marcador cae
    fuera del lienzo y se recorta: en la grabacion de la demo, dos de los
    cuatro ids no se veian. Dibujarla DENTRO del cuadrilatero la ata a algo
    que, si el marcador se ve, tambien se ve.

    Rechaza la implementacion anterior, que anclaba el texto en
    ``(min x, min y - 6)`` -- arriba y a la izquierda del marcador, fuera
    de el.
    """
    canvas = np.zeros((400, 400, 3), np.uint8)
    # Marcador pegado al borde superior: su esquina de arriba esta en y=1,
    # asi que cualquier texto colocado por encima cae fuera del lienzo.
    corners = np.array([[150.0, 1.0], [250.0, 1.0], [250.0, 101.0], [150.0, 101.0]])
    detections = [MarkerDetection(id=7, corners=corners)]

    out = draw_marker_overlay(canvas, detections)

    # Region ESTRICTAMENTE interior: deja fuera los 2 px del contorno del
    # propio marcador, que se dibuja en ambas implementaciones y por tanto
    # no distingue nada. Con el anclaje anterior el texto caia en y = -5,
    # fuera del lienzo, y aqui no habria un solo pixel encendido.
    interior = out[12:92, 162:238]
    assert interior.any(), "no se dibujo nada dentro del marcador: el id se perdio"

    ys, xs = np.nonzero(interior.any(axis=2))
    centre_x = 162 + xs.mean()
    centre_y = 12 + ys.mean()
    assert abs(centre_x - 200.0) < 12.0, f"el id no quedo centrado en x: {centre_x:.1f}"
    assert abs(centre_y - 51.0) < 12.0, f"el id no quedo centrado en y: {centre_y:.1f}"

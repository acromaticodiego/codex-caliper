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
from codex_caliper.ui.overlay import (
    compose_label_lines,
    draw_marker_overlay,
    draw_results,
    format_measurement,
    text_anchor_for,
)

_CANVAS_SHAPE = (300, 500, 3)


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
) -> BananaResult:
    return BananaResult(
        track_id=track_id,
        aggregate=aggregate if aggregate is not None else _aggregate(track_id=track_id),
        verdict=verdict if verdict is not None else _verdict(),
        estimated_weight_g=weight_g,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# format_measurement -- el punto de aplicacion del anti-objetivo de falsa
# precision (spec S4).
# ---------------------------------------------------------------------------


def test_measurement_text_is_english_and_carries_the_uncertainty():
    assert format_measurement(187.34, 0.82) == "187.3 ± 0.8 mm"


def test_decimals_are_capped_by_the_uncertainty():
    # anti-objetivo del spec: nada de falsa precision
    assert format_measurement(187.34, 5.0) == "187 ± 5 mm"


def test_decimal_rule_at_the_floor_boundary():
    # 0.2 mm/px es el piso (Ruling 2): a esa incertidumbre exacta, 1 decimal.
    assert format_measurement(187.34, 0.2) == "187.3 ± 0.2 mm"


def test_decimal_rule_rounds_a_two_digit_uncertainty_to_one_significant_figure():
    # 12.0 mm redondeado a 1 cifra significativa es 10; con esa incertidumbre
    # no se sostienen decimales en el valor tampoco.
    assert format_measurement(187.34, 12.0) == "187 ± 10 mm"


def test_zero_std_is_floored_by_default_to_the_canvas_scale():
    # El caso degenerado real: primer frame de CUALQUIER track, std == 0.0
    # exacto (np.std sin ddof sobre una sola muestra). Sin el piso esto
    # imprimiria una precision que la calibracion no sostiene -- el defecto
    # que Ruling 2 señala como alcanzable en el primer frame de cada track.
    assert format_measurement(187.34567, 0.0) == "187.3 ± 0.2 mm"


def test_floor_can_be_tightened_explicitly_when_the_canvas_scale_differs():
    # draw_results conoce el mm_per_px real del frame y lo pasa explicito;
    # el piso no queda pegado al default cuando la calibracion es otra.
    assert format_measurement(187.34567, 0.0, floor_mm=0.1) == "187.3 ± 0.1 mm"


def test_non_positive_floor_is_rejected_instead_of_silently_widened():
    # Finding 1, ronda 1 de revision: un floor_mm <= 0 solo puede llegar
    # por un defecto aguas arriba (p.ej. un mm_per_px de calibracion en
    # 0.0). Ensancharlo en silencio al default del rig (0.2) mentiria
    # sobre un piso que nunca se aplico realmente -- este caso debe
    # explotar, no maquillarse.
    #
    # std_mm = 0.03 (NO 0.0) a proposito: con std_mm == 0.0,
    # math.log10(0.0) ya lanza ValueError por su cuenta (dominio
    # matematico), lo que haria que esta prueba "pasara" aunque la
    # validacion explicita de floor_mm no existiera -- un accidente de
    # implementacion que enmascararia el hueco real. Con 0.03 (positivo,
    # no-cero) no hay ningun log10(0) de por medio: si la validacion de
    # floor_mm no estuviera, la llamada devolveria silenciosamente
    # "187.35 ± 0.03 mm" en vez de lanzar -- precision que el piso roto
    # nunca verifico. Verificado con el bug deliberado, ver reporte.
    with pytest.raises(ValueError):
        format_measurement(187.34567, 0.03, floor_mm=0.0)
    with pytest.raises(ValueError):
        format_measurement(187.34567, 0.03, floor_mm=-0.5)


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


# ---------------------------------------------------------------------------
# compose_label_lines -- Ruling 3: funcion pura, probada sobre cadenas.
# ---------------------------------------------------------------------------


def test_locked_result_label_shows_verdict_length_caliper_and_weight_in_order():
    result = _result(_rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 90, 150))
    lines = compose_label_lines(result, mm_per_px=0.2)
    assert lines == [
        "#1 OK",
        "L 187.3 ± 0.8 mm",
        "CAL 38.4 ± 0.6 mm",
        "124 g",
    ]


def test_locked_result_that_fails_codex_says_so():
    result = _result(
        _rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 90, 150),
        verdict=_verdict(passes=False),
    )
    assert compose_label_lines(result, mm_per_px=0.2)[0] == "#1 FAIL"


def test_missing_weight_omits_the_weight_line():
    result = _result(_rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 90, 150), weight_g=None)
    lines = compose_label_lines(result, mm_per_px=0.2)
    assert len(lines) == 3
    assert not any("g" in line.split() for line in lines)


def test_unlocked_results_are_drawn_with_a_progress_indicator():
    unlocked = _result(
        _rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 90, 150),
        aggregate=_aggregate(samples=19, locked=False),
    )
    lines = compose_label_lines(unlocked, mm_per_px=0.2)
    assert any("19/30" in line for line in lines)


def test_unlocked_result_label_omits_measurement_lines():
    # mientras no esta "locked" la ventana sigue ruidosa (Task 11); mostrar
    # L/CAL/peso en ese momento seria exactamente el tipo de numero
    # plausible-pero-inestable que el proyecto evita en otros cinco sitios.
    unlocked = _result(
        _rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 90, 150),
        aggregate=_aggregate(samples=19, locked=False),
    )
    lines = compose_label_lines(unlocked, mm_per_px=0.2)
    assert not any(line.startswith("L ") for line in lines)
    assert not any(line.startswith("CAL ") for line in lines)
    assert not any(line.endswith(" g") for line in lines)


def test_locked_result_label_has_no_progress_indicator():
    # la otra mitad del mismo defecto potencial: la barra de progreso no
    # debe sobrevivir una vez que la medida quedo fija.
    locked = _result(_rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 90, 150))
    lines = compose_label_lines(locked, mm_per_px=0.2)
    assert not any(f"/{TEMPORAL_WINDOW}" in line for line in lines)


def test_progress_denominator_follows_the_aggregates_own_window_not_the_module_constant():
    """I7 (ronda de correcciones finales): ``compose_label_lines``/``_progress_bar``
    leian ``TEMPORAL_WINDOW`` (el default de modulo de ``aggregation.py``, 30)
    directamente, mientras ``MeasurementAggregator`` recibe ``cfg.temporal_window``
    -- un campo de config documentado que puede ser distinto. Con
    ``temporal_window: 60`` en la config, un agregado con 45 muestras sin
    asentar mostraba "45/30": una barra YA LLENA (``min(45/30, 1.0) == 1.0``)
    sobre una medida a la que todavia le faltan 15 cuadros.

    La implementacion equivocada que esta prueba RECHAZA: volver a leer
    ``TEMPORAL_WINDOW`` (30) en vez de ``aggregate.window`` (60) -- esa
    version muestra "45/30" y una barra llena; esta prueba exige "45/60" y
    una barra a 3/4.
    """
    window = 60
    assert window != TEMPORAL_WINDOW, "el caso solo discrimina si window difiere del default de modulo"
    unlocked = _result(
        _rectangle_mask(_CANVAS_SHAPE[:2], 50, 50, 90, 150),
        aggregate=_aggregate(samples=45, locked=False, window=window),
    )
    lines = compose_label_lines(unlocked, mm_per_px=0.2)
    assert any("45/60" in line for line in lines), lines
    assert not any("45/30" in line for line in lines), lines
    # 45/60 = 0.75 de 4 segmentos = 3 llenos, 1 vacio -- no los 4 llenos que
    # saldrian de dividir por 30 en vez de por 60.
    assert any("###." in line for line in lines), lines


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
    wide_shape = (300, 900, 3)
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

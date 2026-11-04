"""Tests de la composicion de la lectura de medidas.

``readout.py`` es la UNICA fuente de lo que se muestra: el rotulo sobre el
video y la ficha del panel salen los dos de aqui, asi que no pueden
discrepar sobre el mismo fruto. Es una funcion pura sobre cadenas y
numeros -- nada de pixeles ni de widgets -- por la misma razon que la
antigua ``compose_label_lines``: leer texto de vuelta de una imagen ya
dibujada es OCR y no se puede probar de forma fiable.
"""

from __future__ import annotations

import numpy as np
import pytest

from codex_caliper.aggregation import TEMPORAL_WINDOW, TrackAggregate
from codex_caliper.codex import CODEX_MIN_CALIPER_MM, CODEX_MIN_LENGTH_MM, CodexVerdict
from codex_caliper.pipeline import BananaResult
from codex_caliper.ui.readout import compose_canvas_tag, compose_readout, format_measurement


def _aggregate(**overrides) -> TrackAggregate:
    base = dict(
        track_id=1,
        samples=TEMPORAL_WINDOW,
        locked=True,
        length_mm=257.3,
        length_std_mm=0.4,
        caliper_mm=39.2,
        caliper_std_mm=0.3,
        chord_mm=208.9,
        volume_mm3=1.0e5,
        window=TEMPORAL_WINDOW,
    )
    base.update(overrides)
    return TrackAggregate(**base)


def _result(aggregate: TrackAggregate | None = None, *, weight_g: float | None = None) -> BananaResult:
    aggregate = aggregate if aggregate is not None else _aggregate()
    verdict = CodexVerdict(
        passes=aggregate.length_mm >= CODEX_MIN_LENGTH_MM
        and aggregate.caliper_mm >= CODEX_MIN_CALIPER_MM,
        length_ok=aggregate.length_mm >= CODEX_MIN_LENGTH_MM,
        caliper_ok=aggregate.caliper_mm >= CODEX_MIN_CALIPER_MM,
        commercial_grade=None,
    )
    return BananaResult(
        track_id=aggregate.track_id,
        aggregate=aggregate,
        verdict=verdict,
        estimated_weight_g=weight_g,
        mask=np.zeros((4, 4), bool),
        contour_px=np.empty((0, 2)),
        axis_path_px=np.empty((0, 2)),
        axis_tips_px=(np.zeros(2), np.zeros(2)),
        outer_curve_px=np.empty((0, 2)),
    )


def test_every_measurement_row_carries_the_codex_minimum_it_is_judged_against():
    """Un numero suelto no dice si cumple: "271 mm" solo significa algo al
    lado de "minimo 140 mm". Es el contexto que convierte la lectura en
    algo entendible por alguien que no escribio el programa, que es el
    encargo.

    Rechaza una lectura que muestre las medidas sin su umbral -- la version
    anterior ("L 271 +- 6 mm") pasaba cualquier test de formato y no decia
    contra que se compara.
    """
    readout = compose_readout(_result(), mm_per_px=0.2)

    by_name = {row.name: row for row in readout.rows}
    assert "140" in by_name["CODEX LENGTH"].requirement
    assert "27" in by_name["CALIPER"].requirement


def test_measurements_always_show_their_uncertainty():
    """Ninguna cifra se publica sin su ``±``: es la regla que el README
    afirma del proyecto entero.

    Rechaza una lectura que redondee a un valor limpio y se coma la sigma
    al rediseñar el formato -- el riesgo real de este cambio, porque un
    numero sin ``±`` queda mas bonito en la ficha.
    """
    readout = compose_readout(_result(), mm_per_px=0.2)

    for row in readout.rows:
        assert "±" in row.value, f"{row.name} se publico sin incertidumbre: {row.value}"


def test_the_verdict_stays_hidden_until_the_measurement_is_locked():
    """El veredicto se imprime SIN incertidumbre -- es una comparacion
    contra dos umbrales, no una magnitud que un ``±`` salve de parecer
    definitiva -- asi que no puede mostrarse mientras la ventana todavia se
    esta asentando.

    Las medidas SI se muestran desde el primer cuadro (con su ``±``): la
    regla distingue entre "una cifra honesta bajo incertidumbre" y "un
    fallo presentado como definitivo". Rechaza tanto ocultarlo todo como
    mostrarlo todo.
    """
    unlocked = compose_readout(_result(_aggregate(locked=False, samples=12)), mm_per_px=0.2)
    locked = compose_readout(_result(), mm_per_px=0.2)

    assert unlocked.verdict is None
    assert all(row.meets is None for row in unlocked.rows)
    assert any("±" in row.value for row in unlocked.rows), "las medidas si se muestran sin lock"

    assert locked.verdict == "PASS"
    assert all(row.meets is True for row in locked.rows)


def test_the_weight_stays_hidden_until_the_measurement_is_locked():
    """El peso se imprime como un entero sin ninguna incertidumbre, asi que
    cae bajo la misma regla que el veredicto.

    Rechaza una lectura que lo muestre en cuanto exista una densidad.
    """
    unlocked = compose_readout(_result(_aggregate(locked=False, samples=5), weight_g=118.0), mm_per_px=0.2)
    locked = compose_readout(_result(weight_g=118.0), mm_per_px=0.2)

    assert unlocked.weight_g is None
    assert locked.weight_g == 118


def test_reports_how_much_longer_the_codex_arc_is_than_the_straight_chord():
    """El contraste cuerda/arco es el argumento del proyecto: una caja
    contenedora mide la cuerda, la norma exige el arco.

    Se fija el PORCENTAJE calculado, no solo su presencia: con 208.9 mm de
    cuerda y 257.3 de arco son +23%. Rechaza una implementacion que
    invierta el cociente (arco/cuerda contra cuerda/arco da -19% en vez de
    +23%) o que lo calcule sobre la base equivocada, errores que una
    asercion de "aparece un porcentaje" no distinguiria.
    """
    readout = compose_readout(_result(), mm_per_px=0.2)

    assert readout.chord_mm == pytest.approx(208.9)
    assert readout.arc_mm == pytest.approx(257.3)
    assert readout.curvature_gain_pct == pytest.approx(100.0 * (257.3 / 208.9 - 1.0), abs=0.05)
    assert readout.curvature_gain_pct == pytest.approx(23.2, abs=0.1)


def test_a_failing_fruit_names_which_requirement_it_missed():
    """Con un fruto que no cumple, la ficha tiene que decir CUAL de los dos
    minimos fallo, no solo "FAIL".

    Rechaza una lectura que propague el veredicto global a las dos filas:
    un banano de 120 mm con calibre de 39 incumple el largo y cumple el
    calibre, y marcar las dos en rojo seria falso.
    """
    short = _aggregate(length_mm=120.0, caliper_mm=39.2)

    readout = compose_readout(_result(short), mm_per_px=0.2)

    by_name = {row.name: row for row in readout.rows}
    assert readout.verdict == "FAIL"
    assert by_name["CODEX LENGTH"].meets is False
    assert by_name["CALIPER"].meets is True


def test_the_canvas_tag_is_one_short_line_without_the_numbers():
    """Sobre el fruto solo va el identificador y el estado: los numeros
    viven en el panel.

    La version anterior apilaba cuatro lineas de texto gris ENCIMA del
    banano, tapando la curva Codex que es la protagonista del video y
    quedando ilegible sobre amarillo. Rechaza una etiqueta que vuelva a
    arrastrar las medidas al lienzo.
    """
    locked = compose_canvas_tag(_result())
    measuring = compose_canvas_tag(_result(_aggregate(locked=False, samples=12)))

    assert locked == "#1 PASS"
    assert measuring == "#1 12/30"
    for tag in (locked, measuring):
        assert "\n" not in tag
        assert "mm" not in tag
        assert "±" not in tag


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


def test_a_full_window_that_has_not_settled_does_not_say_it_is_still_measuring():
    """Con la ventana LLENA pero la medida sin fijar, "MEASURING 30/30" se
    lee como un proceso atascado: no queda nada por acumular, lo que pasa
    es que la dispersion no baja del umbral de ``lock``.

    Es el caso NORMAL de este rig -- el calibre se asienta en sigma 0.33 mm
    pero la longitud se queda cerca de 5 mm, contra un umbral de 1.0 mm en
    ambas -- asi que es lo que el espectador va a ver casi todo el video.
    Rechaza la version que decia "MEASURING" en los dos casos.
    """
    settling = compose_readout(_result(_aggregate(locked=False, samples=12)), mm_per_px=0.2)
    full_but_loose = compose_readout(
        _result(_aggregate(locked=False, samples=TEMPORAL_WINDOW)), mm_per_px=0.2
    )

    assert settling.status == f"MEASURING 12/{TEMPORAL_WINDOW}"
    assert full_but_loose.status == f"UNSTABLE {TEMPORAL_WINDOW}/{TEMPORAL_WINDOW}"

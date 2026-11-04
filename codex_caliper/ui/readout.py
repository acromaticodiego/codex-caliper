"""Que se publica de un fruto medido, y en que palabras.

Fuente UNICA de la lectura: el rotulo que ``overlay.py`` pone sobre el
video y la ficha que ``side_panel.py`` dibuja a la derecha salen los dos
de aqui, asi que no pueden mostrar cosas distintas del mismo fruto. Es la
misma propiedad que sostenia la antigua ``compose_label_lines``; lo que
cambia es que ahora los dos consumidores muestran COSAS DISTINTAS a
proposito (un rotulo corto sobre el fruto, la ficha completa en el panel)
y por eso la fuente comun devuelve datos estructurados en vez de lineas de
texto ya maquetadas.

Por que el rotulo del lienzo se quedo en una linea
---------------------------------------------------
Antes se apilaban cuatro lineas sobre el fruto. En la grabacion de la demo
eso tapaba la curva Codex -- la protagonista, y literalmente la geometria
detras del numero que el texto mostraba -- y encima quedaba gris sobre
amarillo saturado, ilegible. Los numeros no se pierden: se mudan al panel,
donde hay sitio para decir contra que minimo se comparan.

Que se oculta mientras la medida no esta ``locked``, y que no
--------------------------------------------------------------
Se muestran SIEMPRE las magnitudes con su ``±``: "271 ± 6 mm" es verdad
desde el primer cuadro y se va estrechando a la vista, que es mejor
material de demo que una barra sin cifras y mas honesto (spec S4 corre en
las dos direcciones: ni falsa precision, ni esconder lo que si se sabe).

Se ocultan el VEREDICTO y el PESO, que se imprimen sin ninguna
incertidumbre -- un "PASS" o un "118 g" no tienen un ``±`` que los salve
de parecer definitivos, y cambiarian de cuadro a cuadro mientras la
ventana se asienta.
"""

from __future__ import annotations

import dataclasses
import math

from codex_caliper.codex import CODEX_MIN_CALIPER_MM, CODEX_MIN_LENGTH_MM
from codex_caliper.pipeline import BananaResult


#: Escala del lienzo del rig, en mm/px. Es el piso por defecto de la
#: incertidumbre: ninguna medida puede reportarse mas fina que el pixel
#: con el que se tomo.
_CANVAS_MM_PER_PX = 0.2


def format_measurement(value_mm: float, std_mm: float, floor_mm: float = _CANVAS_MM_PER_PX) -> str:
    """Formatea ``value_mm ± std_mm`` con tantos decimales como la incertidumbre sostenga.

    Convencion de metrologia: la incertidumbre se redondea a 1 cifra
    significativa, y el VALOR se redondea al mismo lugar decimal que esa
    incertidumbre redondeada -- no al reves, y no por una cadena de
    umbrales ad hoc. A 0.82 mm de incertidumbre (redondea a 0.8, un
    decimal) el valor se imprime con un decimal; a 5.0 mm (redondea a 5,
    cero decimales) el valor se imprime entero.

    El piso de incertidumbre (Ruling 2) se aplica dentro de
    ``_round_to_one_sig_fig``, no aqui -- ver esa funcion para el porque
    (ronda 1 de revision: el piso vivia en esta funcion y cualquier otro
    llamador interno futuro de la primitiva quedaba sin proteccion).
    """
    std_rounded, decimals = _round_to_one_sig_fig(std_mm, floor_mm)
    return f"{value_mm:.{decimals}f} ± {std_rounded:.{decimals}f} mm"


def _round_to_one_sig_fig(std_mm: float, floor_mm: float) -> tuple[float, int]:
    """Aplica el piso de incertidumbre y redondea a 1 cifra significativa.

    Devuelve ``(redondeado, decimales)``. ``decimales`` es cuantos
    decimales hacen falta para representar EXACTAMENTE el valor ya
    redondeado -- no el original, porque redondear puede subir un orden
    de magnitud (0.96 -> 1.0) y los decimales tienen que seguir al
    resultado, no al punto de partida.

    El piso se aplica AQUI, como primer paso, DENTRO de la primitiva --
    no en una capa por encima (ronda 1 de revision). ``format_measurement``
    es hoy el unico llamador, pero la proteccion no depende de que siga
    siendolo: cualquier llamador interno futuro de esta funcion la hereda
    sin tener que acordarse de aplicarla el mismo.

    ``floor_mm`` tiene que ser una escala fisica POSITIVA -- el mm/px de
    la calibracion de ESE frame (Ruling 2), no un limite absoluto del
    proyecto. La revision de la ronda 1 propuso clavarlo a
    ``max(floor_mm, _CANVAS_MM_PER_PX)``; se rechaza a proposito: un
    montaje con un lienzo mas fino (S5.1 llega a ~0.07 mm/px con otra
    focal) tiene derecho a reportar esa resolucion mas fina, y forzar un
    piso de 0.2 mm por encima de lo que pide el caller mentiria HACIA
    ARRIBA -- el anti-objetivo de falsa precision (S4) corre en ambas
    direcciones, no solo "menos decimales de los que hacen falta".

    Lo que SI hay que rechazar no es un piso fino, es un piso que no es
    un piso en absoluto: ``floor_mm <= 0`` solo puede llegar por un
    defecto aguas arriba (p.ej. un ``mm_per_px`` de calibracion en 0.0 por
    un fallo de ``compute_plane_calibration``), y este proyecto ya tiene
    la convencion de dejar propagar ese tipo de error en vez de
    enmascararlo con un valor por defecto silencioso (``pipeline.py``
    deja propagar ``DegenerateMarkerLayout`` por la misma razon: un error
    de programacion aguas arriba tiene que verse). Por eso se lanza
    ``ValueError`` en vez de reinterpretar un piso roto como "usa 0.2
    entonces" -- una excepcion ruidosa aqui es preferible a una cifra con
    precision falsa que se ve bien en pantalla.
    """
    if floor_mm <= 0:
        raise ValueError(f"floor_mm debe ser una escala fisica positiva (mm/px); llego {floor_mm!r}")
    std_effective = max(std_mm, floor_mm)
    exponent = math.floor(math.log10(std_effective))
    rounded = round(std_effective, -exponent)
    if rounded > 0:
        exponent = math.floor(math.log10(rounded))
    decimals = max(0, -exponent)
    return rounded, decimals


@dataclasses.dataclass(frozen=True)
class ReadoutRow:
    """Una magnitud publicada, con el minimo contra el que se juzga.

    ``meets`` es ``None`` -- y no ``False`` -- mientras la medida no esta
    fijada: "todavia no se sabe" y "no cumple" son estados distintos, y
    colapsarlos pintaria de rojo un fruto que acaba de entrar en escena.
    """

    name: str
    value: str
    requirement: str
    meets: bool | None


@dataclasses.dataclass(frozen=True)
class MeasurementReadout:
    """Todo lo publicable de un fruto, sin decidir como se dibuja."""

    title: str
    status: str
    locked: bool
    rows: tuple[ReadoutRow, ...]
    chord_mm: float
    arc_mm: float
    curvature_gain_pct: float
    weight_g: int | None
    verdict: str | None


def _curvature_gain_pct(chord_mm: float, arc_mm: float) -> float:
    """Cuanto MAS largo es el arco Codex que la cuerda recta, en por ciento.

    La base es la CUERDA, no el arco: la frase que acompaña al numero es
    "medir la cuerda se queda corto en un X%", y esa X es el exceso
    relativo a lo que la cuerda daria. Con 208.9 y 257.3 son +23.2%;
    invertir la base daria -18.8%, un numero distinto que describe otra
    cosa.

    Devuelve 0.0 con una cuerda no positiva: no es un caso que
    ``measure_mask`` pueda producir (una curva de dos puntos distintos
    siempre tiene cuerda > 0), pero dividir por ella aqui convertiria una
    degeneracion aguas arriba en una excepcion en el hilo de la interfaz.
    """
    if chord_mm <= 0.0:
        return 0.0
    return 100.0 * (arc_mm / chord_mm - 1.0)


def _status_word(aggregate) -> str:
    """En que estado esta la medida: ``STABLE``, ``MEASURING`` o ``UNSTABLE``.

    Los tres estados son distintos y colapsar los dos ultimos engaña.
    ``MEASURING`` es "todavia acumulo muestras"; ``UNSTABLE`` es "la
    ventana ya esta llena y aun asi la dispersion no baja del umbral de
    ``lock``" -- no queda nada que esperar, la medida tiembla. En este rig
    es el estado NORMAL (calibre con sigma 0.33 mm pero longitud cerca de
    5 mm, contra un umbral de 1.0 mm en ambas), asi que decir "MEASURING
    30/30" para siempre es justo la pantalla atascada que el encargo
    señalaba como no entendible.
    """
    if aggregate.locked:
        return "STABLE"
    if aggregate.samples >= aggregate.window:
        return "UNSTABLE"
    return "MEASURING"


def compose_readout(result: BananaResult, mm_per_px: float) -> MeasurementReadout:
    """Compone la lectura completa de un fruto. Funcion pura."""
    aggregate = result.aggregate
    locked = aggregate.locked

    rows = (
        ReadoutRow(
            name="CODEX LENGTH",
            value=format_measurement(aggregate.length_mm, aggregate.length_std_mm, mm_per_px),
            requirement=f"min {CODEX_MIN_LENGTH_MM:.0f} mm",
            meets=result.verdict.length_ok if locked else None,
        ),
        ReadoutRow(
            name="CALIPER",
            value=format_measurement(aggregate.caliper_mm, aggregate.caliper_std_mm, mm_per_px),
            requirement=f"min {CODEX_MIN_CALIPER_MM:.0f} mm",
            meets=result.verdict.caliper_ok if locked else None,
        ),
    )

    weight_g = (
        int(round(result.estimated_weight_g))
        if locked and result.estimated_weight_g is not None
        else None
    )
    verdict = ("PASS" if result.verdict.passes else "FAIL") if locked else None

    return MeasurementReadout(
        title=f"BANANA #{result.track_id}",
        status=f"{_status_word(aggregate)} {aggregate.samples}/{aggregate.window}",
        locked=locked,
        rows=rows,
        chord_mm=aggregate.chord_mm,
        arc_mm=aggregate.length_mm,
        curvature_gain_pct=_curvature_gain_pct(aggregate.chord_mm, aggregate.length_mm),
        weight_g=weight_g,
        verdict=verdict,
    )


def compose_canvas_tag(result: BananaResult) -> str:
    """El rotulo corto que va SOBRE el fruto en el video.

    Una sola linea y sin cifras: identifica al fruto y dice si la medida
    ya esta fija. Todo lo demas vive en el panel -- ver docstring del
    modulo para por que el rotulo adelgazo.
    """
    aggregate = result.aggregate
    if aggregate.locked:
        return f"#{result.track_id} {'PASS' if result.verdict.passes else 'FAIL'}"
    return f"#{result.track_id} {aggregate.samples}/{aggregate.window}"

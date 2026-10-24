"""Veredicto CODEX STAN 205-1997 y grado comercial Cavendish (Task 10).

Este modulo es deliberadamente autonomo: toma dos ``float`` ya calculados
en otra parte (``codex_length_mm`` es el arco de la curva exterior con el
pedicelo recortado; ``caliper_mm`` es el diametro inscripto en el punto
medio del eje medial podado, ambos ya corregidos de paralaje) y no importa
ninguno de los modulos de geometria. Es pura aritmetica de umbrales: el
unico riesgo real es un operador de comparacion fuera de lugar, por eso los
dos minimos del estandar se verifican con ``>=`` -- ambos son inclusivos
(140.0 mm y 27.0 mm pasan exactamente).

Grado comercial vs. CODEX
--------------------------
El grado comercial Cavendish (minimo 34 mm, premium desde 42+ mm) es una
escala de PRECIO, distinta de los minimos de CODEX (27 mm de calibre). Un
banano puede cumplir CODEX a 27 mm y no tener grado comercial alguno; el
codigo no debe confundir ambas escalas ni derivar una de la otra. El grado
se trunca al milimetro entero (no se redondea): un banano de 38.7 mm es
grado 38, no 39, porque calificar nunca debe promover una fruta a un
bracket que no alcanzo.
"""

from __future__ import annotations

import dataclasses
import math

# Minimos del estandar CODEX STAN 205-1997 (enmendado 2005 y 2022).
CODEX_MIN_LENGTH_MM = 140.0
CODEX_MIN_CALIPER_MM = 27.0

# Minimo comercial Cavendish (escala de grado/precio, no de CODEX).
COMMERCIAL_MIN_CALIPER_MM = 34.0


@dataclasses.dataclass(frozen=True)
class CodexVerdict:
    """Resultado de evaluar una medida contra el estandar CODEX.

    ``passes`` es la conjuncion de ``length_ok`` y ``caliper_ok``.
    ``commercial_grade`` es independiente de ``passes``: puede ser ``None``
    incluso cuando la fruta cumple CODEX.
    """

    passes: bool
    length_ok: bool
    caliper_ok: bool
    commercial_grade: int | None


def evaluate(length_mm: float, caliper_mm: float) -> CodexVerdict:
    """Evalua una longitud y un calibre contra CODEX STAN 205-1997.

    Ambos minimos son inclusivos: exactamente ``CODEX_MIN_LENGTH_MM`` o
    ``CODEX_MIN_CALIPER_MM`` cumple.
    """
    length_ok = length_mm >= CODEX_MIN_LENGTH_MM
    caliper_ok = caliper_mm >= CODEX_MIN_CALIPER_MM
    commercial_grade = _commercial_grade(caliper_mm)
    return CodexVerdict(
        passes=length_ok and caliper_ok,
        length_ok=length_ok,
        caliper_ok=caliper_ok,
        commercial_grade=commercial_grade,
    )


def _commercial_grade(caliper_mm: float) -> int | None:
    """Calibre truncado a milimetro entero, o ``None`` bajo el minimo comercial."""
    if caliper_mm < COMMERCIAL_MIN_CALIPER_MM:
        return None
    return math.floor(caliper_mm)

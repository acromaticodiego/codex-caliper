"""Tests de ``AppConfig`` (Task 1) y de la validacion de forma de su JSON.

La validacion de ENTRADAS (no solo de contenedores) se prueba aqui, en el
nivel donde vive; su camino hasta el mensaje del operador se prueba en
``tests/test_entrypoint.py``.
"""

from __future__ import annotations

import json

import pytest

from codex_caliper.config import AppConfig


def test_effective_height_subtracts_marker_thickness():
    cfg = AppConfig(marker_thickness_mm=3.0, working_distance_mm=270.0)
    assert cfg.effective_height_mm == 267.0


def test_roundtrip_preserves_marker_layout(tmp_path):
    cfg = AppConfig(marker_thickness_mm=2.5, marker_layout={0: (0.0, 0.0), 1: (300.0, 0.0)})
    cfg.save(tmp_path / "c.json")
    assert AppConfig.load(tmp_path / "c.json").marker_layout == cfg.marker_layout


# ---------------------------------------------------------------------------
# I6 -- la validacion de forma mira las ENTRADAS, no solo los contenedores
# ---------------------------------------------------------------------------


def _write(tmp_path, layout):
    path = tmp_path / "app.json"
    path.write_text(
        json.dumps({"marker_thickness_mm": 2.0, "marker_layout": layout}), encoding="utf-8"
    )
    return path


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param({"x": 0, "y": 0}, id="object-instead-of-pair"),
        pytest.param("215,118", id="string-instead-of-pair"),
        pytest.param([215.0, 118.0, 0.0], id="three-coordinates"),
        pytest.param([215.0], id="one-coordinate"),
        pytest.param(215.0, id="bare-number"),
        pytest.param([215.0, None], id="null-coordinate"),
        pytest.param([215.0, "118"], id="stringy-coordinate"),
        pytest.param([True, False], id="booleans"),
    ],
)
def test_a_marker_layout_value_of_the_wrong_shape_raises_value_error(tmp_path, bad_value):
    """I6 (ronda de correcciones finales): la ronda anterior validaba que la
    raiz fuera un objeto y que ``marker_layout`` fuera un objeto -- los dos
    CONTENEDORES -- y nada sobre sus entradas.

    La implementacion equivocada que esta prueba RECHAZA es exactamente esa:
    con ella, ``{"1": {"x": 0, "y": 0}}`` carga limpio como
    ``{1: ('x', 'y')}`` y el ``TypeError`` aparece mas tarde, dentro de
    ``canvas_size_for`` en el constructor del ``Pipeline`` -- que vive en el
    ``try`` de ``main()`` cuyo unico manejador es ``except _StartupError``, o
    sea un traceback crudo en la cara del operador.

    Se parametriza sobre la FORMA DEL VALOR, no sobre otro contenedor: el
    punto del hallazgo es fijar el mecanismo (cada valor es una pareja de
    numeros) en vez de las dos entradas concretas que se encontraron. Las
    cadenas y los booleanos estan aqui a proposito -- ``"19"`` tiene
    longitud 2 y ``isinstance(True, int)`` es ``True``, asi que una
    comprobacion de longitud a secas, o una de "numero" ingenua, los
    aceptaria.
    """
    path = _write(tmp_path, {"1": bad_value})
    with pytest.raises(ValueError) as excinfo:
        AppConfig.load(path)
    # El mensaje tiene que nombrar la clave: es el dato con el que el
    # operador encuentra la linea que corregir.
    assert "marker_layout" in str(excinfo.value)
    assert "'1'" in str(excinfo.value)


def test_a_well_shaped_marker_layout_still_loads(tmp_path):
    """El otro lado: la validacion no puede rechazar lo valido. Enteros y
    flotantes, los dos, porque un JSON escrito a mano trae ``0`` tanto como
    ``0.0``."""
    path = _write(tmp_path, {"1": [0, 0], "2": [215.0, 0.5], "3": (215, 118)})
    cfg = AppConfig.load(path)
    assert cfg.marker_layout == {1: (0.0, 0.0), 2: (215.0, 0.5), 3: (215.0, 118.0)}

"""Tests de la vista de validacion (Task 19).

Dos huecos del encargo se cubren aqui, no solo se mencionan:

1. **La interfaz entre ``TrackAggregate`` y ``BananaMeasures`` (Ruling 1).**
   ``TrackAggregate`` (Task 11) no agrega ``area_mm2`` ni ``width_profile`` --
   Task 11 los llamo "auxiliares" a proposito. El adaptador de esta vista
   NO puede fabricar un area que ninguna medida respalda (eso es exactamente
   el anti-objetivo de la spec S4): marca ``area_mm2`` como NaN y
   ``width_profile`` como arreglos vacios. Se prueba que ese NaN sobrevive
   hasta el CSV, no solo en el objeto en memoria.

2. **``test_validation_view_shows_mae_and_r_squared`` del brief es debil**
   (ruling 2): ``"MAE" in texto and "R²" in texto`` lo satisface una
   etiqueta sin ningun numero detras. Se fijan las CIFRAS reales,
   calculadas de forma independiente con ``fit_density``/``error_metrics``
   sobre el mismo log, no solo las etiquetas.

Tambien se decide deliberadamente que muestra la vista en sus dos estados
sin ajuste posible: cero muestras (no puede mostrar "MAE: nan g", eso es
precisamente lo que la spec S4 prohibe publicar) y menos de
``MIN_DENSITY_SAMPLES`` (``fit_density`` lanza por debajo de 10 -- la vista
tiene que absorber eso con una pantalla de progreso, no un traceback).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from codex_caliper.aggregation import TEMPORAL_WINDOW, TrackAggregate
from codex_caliper.geometry.measures import BananaMeasures
from codex_caliper.validation import MIN_DENSITY_SAMPLES, ValidationLog, error_metrics, fit_density
from codex_caliper.ui.validation_view import ValidationView


def _aggregate(
    *,
    track_id: int = 1,
    locked: bool = True,
    length_mm: float = 187.3,
    caliper_mm: float = 38.4,
    volume_mm3: float = 1.0e5,
    samples: int = 30,
    length_std_mm: float = 0.8,
    caliper_std_mm: float = 0.5,
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
        window=TEMPORAL_WINDOW,
    )


# ---------------------------------------------------------------------------
# Ruling 1 -- el adaptador TrackAggregate -> BananaMeasures
# ---------------------------------------------------------------------------


def test_measures_from_aggregate_marks_area_and_profile_unavailable_not_fabricated(qtbot, tmp_path):
    """``area_mm2`` es NaN (no un area reinventada) y ``width_profile`` son
    arreglos vacios -- Task 11 nunca agrego esas dos cosas, asi que no hay
    ninguna medida real que respalde un numero ahi."""
    aggregate = _aggregate()

    measures = ValidationView.measures_from_aggregate(aggregate)

    assert isinstance(measures, BananaMeasures)
    assert measures.codex_length_mm == aggregate.length_mm
    assert measures.caliper_mm == aggregate.caliper_mm
    assert measures.volume_mm3 == aggregate.volume_mm3
    assert math.isnan(measures.area_mm2)
    s_mm, w_mm = measures.width_profile
    assert len(s_mm) == 0 and len(w_mm) == 0


def test_recording_from_a_locked_aggregate_writes_nan_area_to_the_csv(qtbot, tmp_path):
    log = ValidationLog(tmp_path / "v.csv")
    view = ValidationView(log)
    qtbot.addWidget(view)

    view.record_sample("sample-1", _aggregate(locked=True), actual_weight_g=140.0)

    frame = log.to_frame()
    assert len(frame) == 1
    assert frame.loc[0, "codex_length_mm"] == pytest.approx(187.3)
    assert frame.loc[0, "caliper_mm"] == pytest.approx(38.4)
    assert frame.loc[0, "volume_mm3"] == pytest.approx(1.0e5)
    assert math.isnan(frame.loc[0, "area_mm2"]), "el CSV tiene que decir 'no medido', no inventar un area"

    # El mismo CSV, releido desde disco -- el NaN sobrevive el roundtrip.
    reloaded = ValidationLog(tmp_path / "v.csv").to_frame()
    assert math.isnan(reloaded.loc[0, "area_mm2"])


def test_recording_from_an_unlocked_aggregate_is_rejected(qtbot, tmp_path):
    """Una medida que todavia no se asento (ventana sin llenar, o ruidosa)
    no debe entrar PERMANENTEMENTE a la bitacora de validacion -- es
    exactamente el tipo de cifra plausible-pero-inestable que el proyecto
    ya evito en otros cinco sitios."""
    log = ValidationLog(tmp_path / "v.csv")
    view = ValidationView(log)
    qtbot.addWidget(view)

    with pytest.raises(ValueError):
        view.record_sample("sample-1", _aggregate(locked=False), actual_weight_g=140.0)

    assert len(log.to_frame()) == 0


# ---------------------------------------------------------------------------
# Estado vacio (cero muestras) -- ninguna cifra sin sentido
# ---------------------------------------------------------------------------


def test_empty_log_shows_a_deliberate_empty_state_without_nan_or_crashing(qtbot, tmp_path):
    log = ValidationLog(tmp_path / "v.csv")
    view = ValidationView(log)
    qtbot.addWidget(view)

    text = view.summary_text()

    assert "nan" not in text.lower(), "cero muestras no puede imprimir 'MAE: nan g'"
    assert view.sample_count() == 0
    # La pantalla vacia es una invitacion a actuar (skill frontend-design):
    # tiene que decir CUANTAS muestras hacen falta, no solo "sin datos".
    assert str(MIN_DENSITY_SAMPLES) in text


def test_fewer_than_minimum_samples_shows_progress_not_a_fitted_density(qtbot, tmp_path):
    log = ValidationLog(tmp_path / "v.csv")
    view = ValidationView(view_log := log)
    qtbot.addWidget(view)

    for i in range(MIN_DENSITY_SAMPLES - 1):
        view.record_sample(f"sample-{i}", _aggregate(volume_mm3=90_000.0 + i * 1000), actual_weight_g=100.0 + i)

    text = view.summary_text()
    assert "nan" not in text.lower()
    assert view.sample_count() == MIN_DENSITY_SAMPLES - 1
    assert str(view.sample_count()) in text
    assert str(MIN_DENSITY_SAMPLES) in text
    assert "MAE" not in text, "no hay suficientes muestras para ajustar una densidad -- no se inventa un MAE"


# ---------------------------------------------------------------------------
# Ruling 2 -- MAE y R² reales, no solo las etiquetas
# ---------------------------------------------------------------------------


def _fill_log_with_real_samples(log: ValidationLog, n: int = 12) -> None:
    """Pesos que son casi, pero no exactamente, proporcionales al volumen --
    para que la densidad ajustada NO sea perfecta y el R²/MAE resultante
    sea un numero no trivial, no 1.0/0.0 por casualidad."""
    rng = np.random.default_rng(3)
    density = 0.95
    for i in range(n):
        volume_mm3 = 80_000.0 + i * 4_000.0
        true_weight = density * volume_mm3 / 1000.0
        noisy_weight = float(true_weight + rng.normal(scale=1.5))
        log.record(
            f"s{i}",
            BananaMeasures(
                codex_length_mm=180.0 + i,
                caliper_mm=36.0,
                area_mm2=float("nan"),
                volume_mm3=volume_mm3,
                width_profile=(np.array([]), np.array([])),
            ),
            actual_weight_g=noisy_weight,
        )


def test_summary_reports_the_real_mae_and_r_squared_once_fitted(qtbot, tmp_path):
    log = ValidationLog(tmp_path / "v.csv")
    _fill_log_with_real_samples(log, n=12)
    view = ValidationView(log)
    qtbot.addWidget(view)

    frame = log.to_frame()
    expected_density = fit_density(frame)
    expected_metrics = error_metrics(frame, expected_density)

    text = view.summary_text()

    assert "MAE" in text and "R²" in text
    # No alcanza con las ETIQUETAS: se fijan las cifras reales, calculadas
    # de forma independiente sobre el mismo log.
    assert f"{expected_metrics.mae_g:.1f}" in text
    assert f"{expected_metrics.r_squared:.3f}" in text
    assert f"n={expected_metrics.n}" in text or f"n = {expected_metrics.n}" in text


def test_summary_includes_hand_measured_length_mae_when_present(qtbot, tmp_path):
    log = ValidationLog(tmp_path / "v.csv")
    rng = np.random.default_rng(5)
    for i in range(MIN_DENSITY_SAMPLES):
        volume_mm3 = 80_000.0 + i * 3_000.0
        weight = 0.95 * volume_mm3 / 1000.0 + rng.normal(scale=1.0)
        log.record(
            f"s{i}",
            BananaMeasures(
                codex_length_mm=185.0,
                caliper_mm=36.0,
                area_mm2=float("nan"),
                volume_mm3=volume_mm3,
                width_profile=(np.array([]), np.array([])),
            ),
            actual_weight_g=float(weight),
            actual_length_mm=182.0,
        )
    view = ValidationView(log)
    qtbot.addWidget(view)

    frame = log.to_frame()
    expected_metrics = error_metrics(frame, fit_density(frame))

    text = view.summary_text()
    assert expected_metrics.mae_length_mm is not None
    assert f"{expected_metrics.mae_length_mm:.1f}" in text


# ---------------------------------------------------------------------------
# Tabla
# ---------------------------------------------------------------------------


def test_table_reflects_recorded_rows_including_unavailable_area(qtbot, tmp_path):
    log = ValidationLog(tmp_path / "v.csv")
    view = ValidationView(log)
    qtbot.addWidget(view)

    view.record_sample("sample-1", _aggregate(), actual_weight_g=140.0)

    assert view._table.rowCount() == 1
    assert view._table.item(0, 0).text() == "sample-1"
    # El area no medida se ve como "no disponible", no como "0" ni "nan"
    # crudo (ambos engañarian a un lector rapido de la tabla).
    area_cell = view._table.item(0, view._column_index("area_mm2")).text()
    assert area_cell in ("—", "n/a", "N/A")


def test_length_and_caliper_cells_carry_their_sigma(qtbot, tmp_path):
    """I9 (ronda de correcciones finales): ``_format_cell`` imprimia
    ``codex_length_mm``/``caliper_mm`` con ``.1f`` a secas -- la unica
    columna de esta tabla publicable (spec S13, punto 4) sin ``±``, aunque
    ``record_sample`` SI tiene la desviacion a la mano (``aggregate.length_std_mm``/
    ``caliper_std_mm``). Se fija la cadena exacta que ``format_measurement``
    produce, no solo "contiene un numero": es la misma funcion que usa el
    resto de la interfaz, asi que la tabla y el video no pueden mostrar un
    formato distinto para la misma medida.

    La implementacion equivocada que esta prueba RECHAZA: la version
    anterior, que ignoraba ``length_std_mm``/``caliper_std_mm`` y devolvia
    ``f"{value:.1f}"`` -- "187.3", sin ningun "±".
    """
    log = ValidationLog(tmp_path / "v.csv")
    view = ValidationView(log)
    qtbot.addWidget(view)

    view.record_sample(
        "sample-1",
        _aggregate(length_mm=187.34, length_std_mm=0.82, caliper_mm=38.4, caliper_std_mm=0.6),
        actual_weight_g=140.0,
    )

    from codex_caliper.ui.overlay import format_measurement

    length_cell = view._table.item(0, view._column_index("codex_length_mm")).text()
    caliper_cell = view._table.item(0, view._column_index("caliper_mm")).text()
    assert length_cell == format_measurement(187.34, 0.82)
    assert "±" in length_cell
    assert caliper_cell == format_measurement(38.4, 0.6)
    assert "±" in caliper_cell


def test_volume_cell_is_shown_in_cm3_at_reduced_precision(qtbot, tmp_path):
    """I9 (ronda de correcciones finales): ``volume_mm3`` se imprimia con
    ``.1f`` sobre el mm³ crudo -- "100000.0" para un volumen de 100 cm³,
    siete cifras significativas sobre un volumen integrado de medidas con
    +-1 mm de incertidumbre (falsa precision, spec S4). Ahora se muestra en
    cm³ y a una precision reducida.

    La implementacion equivocada que esta prueba RECHAZA: seguir mostrando
    el mm³ crudo -- esta prueba falla si la celda contiene "100000", el
    valor exacto en mm³, y exige que contenga "cm" en vez de "mm".
    """
    log = ValidationLog(tmp_path / "v.csv")
    view = ValidationView(log)
    qtbot.addWidget(view)

    view.record_sample("sample-1", _aggregate(volume_mm3=1.0e5), actual_weight_g=140.0)

    volume_cell = view._table.item(0, view._column_index("volume_mm3")).text()
    assert "100000" not in volume_cell, "sigue mostrando el mm3 crudo sin convertir"
    assert "100" in volume_cell
    assert "cm" in volume_cell

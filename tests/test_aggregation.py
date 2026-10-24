"""Tests de agregacion temporal e incertidumbre (Task 11).

Tres ideas verificadas, no de pasada:

1. Que el agregado reporta la MEDIANA, no la media -- un solo frame con la
   mascara rota no debe arrastrar el resultado.
   ``test_reports_the_median_not_the_mean`` usa un atipico extremo y fuera
   de centro (900.0 mm contra una serie estable de 190.0 mm, mas de 4x el
   valor real) a proposito: con una serie simetrica alrededor de la media
   una implementacion con ``np.mean`` tambien pasaria por casualidad. Se
   verifico a mano que cambiar ``median`` por ``mean`` en la implementacion
   hace fallar este test (ver informe de la tarea).

2. Que ``locked`` exige las DOS condiciones -- ventana llena Y desviacion
   baja -- no cualquiera de las dos por separado.
   ``test_does_not_lock_until_the_window_is_full`` fija una serie SIN
   ruido (desviacion ~0) pero con la ventana todavia sin llenar: si el
   candado dependiera solo de la desviacion, este test fallaria.
   ``test_does_not_lock_while_the_measurement_is_still_noisy`` fija una
   ventana ya llena (40 empujones contra una ventana de 30) pero con
   ruido por encima del umbral: si el candado dependiera solo de que la
   ventana este llena, este test fallaria.

3. Que la ventana deslizante EFECTIVAMENTE descarta la muestra mas vieja,
   no solo que el conteo quede acotado en 30.
   ``test_window_slides_and_drops_the_oldest_sample`` empuja primero 35
   valores muy distintos (5.0 mm) y luego exactamente 30 valores de
   190.0 mm: si la ventana no deslizara (p. ej. un buffer que solo
   recorta por longitud al leer, sin descartar en cada empuje, o que
   conserva los primeros N en vez de los ultimos N), la mediana
   reportada seria 5.0 en vez de 190.0. Afirmar solo ``samples == 30``
   (como hace el borrador original de la tarea) no distingue estos dos
   casos -- de ahi la aserción adicional sobre ``length_mm``.
"""

from __future__ import annotations

import numpy as np
import pytest

from codex_caliper.aggregation import MeasurementAggregator, TrackAggregate
from codex_caliper.geometry.measures import BananaMeasures


def measures(
    length: float = 190.0,
    caliper: float = 38.0,
    area: float = 1000.0,
    volume: float = 50_000.0,
) -> BananaMeasures:
    """Construye un ``BananaMeasures`` minimo para las pruebas de este modulo.

    ``width_profile`` no participa en la agregacion temporal, asi que se
    deja vacio.
    """
    return BananaMeasures(
        codex_length_mm=length,
        caliper_mm=caliper,
        area_mm2=area,
        volume_mm3=volume,
        width_profile=(np.array([]), np.array([])),
    )


_rng = np.random.default_rng(seed=42)
jittered_lengths = (190.0 + _rng.normal(0.0, 0.5, size=30)).tolist()
very_noisy_lengths = [170.0 if i % 2 == 0 else 210.0 for i in range(40)]


def test_reports_the_median_not_the_mean():
    # un frame atipico no debe arrastrar el resultado
    agg = MeasurementAggregator(window=30)
    for _ in range(29):
        agg.push(1, measures(length=190.0))
    out = agg.push(1, measures(length=900.0))  # frame malo
    assert out.length_mm == pytest.approx(190.0, abs=0.1)


def test_the_mean_would_not_survive_the_same_outlier():
    # control: confirma que el caso anterior es un caso real para la
    # mediana, no un empate casual con la media.
    values = [190.0] * 29 + [900.0]
    assert np.mean(values) != pytest.approx(190.0, abs=0.1)
    assert np.median(values) == pytest.approx(190.0, abs=0.1)


def test_std_is_reported_as_the_uncertainty():
    agg = MeasurementAggregator(window=30)
    out = None
    for v in jittered_lengths:
        out = agg.push(1, measures(length=v))
    assert agg.get(1).length_std_mm == pytest.approx(np.std(jittered_lengths), rel=0.05)
    assert out.length_std_mm == pytest.approx(np.std(jittered_lengths), rel=0.05)


def test_does_not_lock_until_the_window_is_full():
    agg = MeasurementAggregator(window=30)
    out = None
    for _ in range(29):
        out = agg.push(1, measures(length=190.0))
    assert out.locked is False
    assert agg.push(1, measures(length=190.0)).locked is True


def test_does_not_lock_while_the_measurement_is_still_noisy():
    agg = MeasurementAggregator(window=30, lock_std_mm=1.0)
    out = None
    for v in very_noisy_lengths[:40]:
        out = agg.push(1, measures(length=v))
    assert out.locked is False


def test_window_slides_and_drops_the_oldest_sample():
    agg = MeasurementAggregator(window=30)
    for _ in range(35):
        agg.push(1, measures(length=5.0))
    out = None
    for _ in range(30):
        out = agg.push(1, measures(length=190.0))
    assert out.samples == 30
    assert out.length_mm == pytest.approx(190.0, abs=0.1)


def test_get_returns_none_for_an_unknown_track():
    agg = MeasurementAggregator(window=30)
    assert agg.get(99) is None


def test_forget_discards_the_track_buffer():
    agg = MeasurementAggregator(window=30)
    agg.push(1, measures(length=190.0))
    agg.forget(1)
    assert agg.get(1) is None
    assert 1 not in agg.active_ids()


def test_active_ids_reports_only_tracks_with_data():
    agg = MeasurementAggregator(window=30)
    agg.push(1, measures(length=190.0))
    agg.push(2, measures(length=150.0))
    assert sorted(agg.active_ids()) == [1, 2]
    agg.forget(1)
    assert agg.active_ids() == [2]


def test_push_returns_a_track_aggregate_with_the_pushed_track_id():
    agg = MeasurementAggregator(window=30)
    out = agg.push(7, measures(length=190.0))
    assert isinstance(out, TrackAggregate)
    assert out.track_id == 7

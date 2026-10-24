"""Tests de validacion, regresion de densidad y metricas de error (Task 15).

Cuatro de los tests del borrador original del plan no discriminaban una
implementacion incorrecta (ver informe de la tarea para la evidencia de
ruptura deliberada de cada uno); aqui quedan reforzados en vez de copiados:

1. ``test_log_roundtrips_through_csv`` original solo afirmaba
   ``len(...) == 3`` -- una bitacora que escribiera tres filas de basura
   pasaria igual. Esta version afirma los VALORES de cada columna, fila por
   fila, incluyendo que ``actual_length_mm`` sobrevive tanto presente como
   ausente (``NaN``).
2. ``test_dimensional_check_returns_signed_error_in_mm`` original solo
   probaba ``+0.4`` -- una implementacion con ``abs()`` adentro pasaria
   igual, y la palabra "signed" del nombre quedaria sin sustento. Se agrega
   un caso negativo.
3. ``test_fit_density_requires_a_minimum_sample_size`` original usaba 2
   filas contra un requisito de 10 -- no distingue un error de un-por-uno
   (``> 10`` en vez de ``>= 10``). Aqui se fija la FRONTERA exacta: 9 filas
   lanza, 10 no.
4. ``test_density_is_fitted_not_assumed`` original generaba datos
   perfectamente colineales por el origen, donde una regresion CON
   intercepto tambien recupera pendiente 0.94 y R²=1 -- no distingue la
   regresion por el origen que exige el plan de una con ordenada libre. Se
   agrega una fila con un corrimiento (``+15`` sobre un solo punto) que
   separa las dos pendientes: por el origen da ~0.944 (dentro de la
   tolerancia), con intercepto da ~0.858 (bien afuera).

Se agrega ademas un test dedicado a fijar la convencion de R² (no centrada,
ver docstring de ``codex_caliper.validation``) con un caso donde las dos
convenciones de verdad difieren -- uno de ellos da R² negativo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from codex_caliper.geometry.measures import BananaMeasures
from codex_caliper.validation import (
    DensityFit,
    ValidationLog,
    dimensional_check,
    error_metrics,
    fit_density,
)


def measures(
    length: float = 180.0,
    caliper: float = 36.0,
    area: float = 900.0,
    volume: float = 45_000.0,
) -> BananaMeasures:
    """Construye un ``BananaMeasures`` minimo; ``width_profile`` no participa aqui."""
    return BananaMeasures(
        codex_length_mm=length,
        caliper_mm=caliper,
        area_mm2=area,
        volume_mm3=volume,
        width_profile=(np.array([]), np.array([])),
    )


def _density_frame(volume_cm3: np.ndarray, actual_weight_g: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "volume_mm3": volume_cm3 * 1000.0,
            "actual_weight_g": actual_weight_g,
        }
    )


# ---------------------------------------------------------------------------
# ValidationLog
# ---------------------------------------------------------------------------


def test_log_roundtrips_through_csv(tmp_path):
    path = tmp_path / "v.csv"
    log = ValidationLog(path)
    log.record(
        "s1",
        measures(length=180.3, caliper=36.1, area=900.5, volume=45_000.2),
        actual_weight_g=132.4,
        actual_length_mm=181.0,
    )
    log.record(
        "s2",
        measures(length=150.0, caliper=28.0, area=700.0, volume=30_000.0),
        actual_weight_g=95.0,
        # sin longitud a mano -- debe sobrevivir como NaN, no como basura
    )
    log.record(
        "s3",
        measures(length=200.0, caliper=40.0, area=1_100.0, volume=60_000.0),
        actual_weight_g=170.2,
        actual_length_mm=198.5,
    )

    # Reabrir desde disco con una instancia NUEVA: esto es el round trip real.
    reloaded = ValidationLog(path).to_frame()

    assert list(reloaded["sample_id"]) == ["s1", "s2", "s3"]
    assert reloaded["codex_length_mm"].tolist() == pytest.approx([180.3, 150.0, 200.0])
    assert reloaded["caliper_mm"].tolist() == pytest.approx([36.1, 28.0, 40.0])
    assert reloaded["area_mm2"].tolist() == pytest.approx([900.5, 700.0, 1_100.0])
    assert reloaded["volume_mm3"].tolist() == pytest.approx([45_000.2, 30_000.0, 60_000.0])
    assert reloaded["actual_weight_g"].tolist() == pytest.approx([132.4, 95.0, 170.2])
    assert reloaded["actual_length_mm"].iloc[0] == pytest.approx(181.0)
    assert pd.isna(reloaded["actual_length_mm"].iloc[1])
    assert reloaded["actual_length_mm"].iloc[2] == pytest.approx(198.5)


def test_log_accumulates_over_a_fresh_instance_on_the_same_path(tmp_path):
    # Reabrir una bitacora existente debe retomar las filas anteriores, no
    # empezar de cero -- confirma que el constructor lee el CSV existente.
    path = tmp_path / "v.csv"
    ValidationLog(path).record("s1", measures(), actual_weight_g=100.0)
    log2 = ValidationLog(path)
    log2.record("s2", measures(), actual_weight_g=200.0)
    assert list(ValidationLog(path).to_frame()["sample_id"]) == ["s1", "s2"]


# ---------------------------------------------------------------------------
# dimensional_check
# ---------------------------------------------------------------------------


def test_dimensional_check_returns_signed_error_in_mm():
    # Caso positivo (sobreestima) y NEGATIVO (subestima): un `abs()` dentro
    # de la implementacion pasaria el caso original pero fallaria este.
    assert dimensional_check(24.4, 24.0) == pytest.approx(0.4)
    assert dimensional_check(23.6, 24.0) == pytest.approx(-0.4)


def test_dimensional_check_against_the_iso_card_reference():
    # Caso de uso real del spec: tarjeta ISO/IEC 7810 ID-1, 85.60 mm de lado largo.
    assert dimensional_check(85.9, 85.60) == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# fit_density
# ---------------------------------------------------------------------------


def test_fit_density_requires_a_minimum_sample_size_boundary():
    # Frontera exacta: 9 filas debe lanzar, 10 no (no un ">10" en vez de ">=10").
    volume_cm3 = np.arange(1, 11, dtype=float)  # 1..10
    weight_g = 0.94 * volume_cm3

    with pytest.raises(ValueError):
        fit_density(_density_frame(volume_cm3[:9], weight_g[:9]))

    fit_density(_density_frame(volume_cm3, weight_g))  # 10 filas: no debe lanzar


def test_density_is_fitted_not_assumed():
    # Datos NO perfectamente colineales por el origen: un solo punto trae un
    # corrimiento de +15 g. Sobre datos perfectos (`w = k*v` exacto) una
    # regresion con intercepto tambien recuperaria pendiente 0.94 y R²=1,
    # asi que ese caso no discrimina (ver docstring del modulo de tests).
    # Aqui la pendiente POR EL ORIGEN (sum(vw)/sum(v^2)) y la pendiente CON
    # intercepto (cov(v,w)/var(v)) difieren de verdad: ~0.944 vs ~0.858.
    volume_cm3 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    weight_g = 0.94 * volume_cm3
    weight_g[0] += 15.0  # una muestra pesada de mas

    # Control: confirmar que el caso realmente separa ambas pendientes, para
    # que la asercion de abajo no sea un empate casual.
    slope_with_intercept = np.sum((volume_cm3 - volume_cm3.mean()) * (weight_g - weight_g.mean())) / np.sum(
        (volume_cm3 - volume_cm3.mean()) ** 2
    )
    assert slope_with_intercept != pytest.approx(0.94, rel=0.02)

    fit = fit_density(_density_frame(volume_cm3, weight_g))

    assert isinstance(fit, DensityFit)
    assert fit.g_per_cm3 == pytest.approx(0.94, rel=0.02)
    assert fit.g_per_cm3 != pytest.approx(slope_with_intercept, rel=0.02)
    assert fit.r_squared > 0.99
    assert fit.n == 10


def test_r_squared_uses_the_uncentered_through_origin_convention():
    # Volumenes muy agrupados (95..105 cm3) con un intercepto real de 60 g:
    # la regresion por el origen ajusta bien en terminos absolutos (R² no
    # centrado > 0.99) pero PEOR que predecir la media en terminos
    # centrados, lo que vuelve negativa la convencion centrada. Es el caso
    # donde las dos convenciones de verdad difieren -- fija cual se eligio.
    volume_cm3 = np.array([95, 96, 97, 98, 99, 100, 101, 102, 103, 105], dtype=float)
    actual_weight_g = 60.0 + 0.3 * (volume_cm3 - 100.0)

    fit = fit_density(_density_frame(volume_cm3, actual_weight_g))

    predicted = fit.g_per_cm3 * volume_cm3
    ss_res = float(np.sum((actual_weight_g - predicted) ** 2))
    uncentered_r2 = 1.0 - ss_res / float(np.sum(actual_weight_g**2))
    centered_r2 = 1.0 - ss_res / float(np.sum((actual_weight_g - actual_weight_g.mean()) ** 2))

    assert centered_r2 < 0.0  # la convencion centrada se vuelve negativa aqui
    assert uncentered_r2 > 0.99  # mientras la no centrada sigue siendo "publicable"
    assert fit.r_squared == pytest.approx(uncentered_r2)


# ---------------------------------------------------------------------------
# error_metrics
# ---------------------------------------------------------------------------


def test_mae_is_zero_for_a_perfect_fit():
    volume_cm3 = np.linspace(20.0, 120.0, 10)
    fit = DensityFit(g_per_cm3=0.94, r_squared=1.0, n=10)
    frame = _density_frame(volume_cm3, fit.g_per_cm3 * volume_cm3)

    metrics = error_metrics(frame, fit)

    assert metrics.mae_g == pytest.approx(0.0, abs=1e-6)
    assert metrics.mape_pct == pytest.approx(0.0, abs=1e-6)
    assert metrics.r_squared == pytest.approx(1.0, abs=1e-9)


def test_mape_is_reported_in_percent():
    # El peso predicho queda, a proposito, un 10% por encima del real en
    # TODAS las filas -- MAPE debe salir 10.0, no 0.10 (fraccion) ni 1000.0
    # (doble multiplicacion por 100).
    volume_cm3 = np.array([30.0, 45.0, 60.0, 75.0, 90.0])
    fit = DensityFit(g_per_cm3=1.0, r_squared=1.0, n=5)
    predicted = fit.g_per_cm3 * volume_cm3
    actual_weight_g = predicted / 1.10

    metrics = error_metrics(_density_frame(volume_cm3, actual_weight_g), fit)

    assert metrics.mape_pct == pytest.approx(10.0, rel=0.01)


def test_mape_denominator_is_the_actual_value_not_the_predicted_value():
    # Con denominador = predicho en vez de real, este caso da un MAPE
    # distinto: real=50 y predicho=100 da 100% (num/real) pero 50% si el
    # denominador fuera el predicho. Fija cual de los dos es.
    fit = DensityFit(g_per_cm3=1.0, r_squared=1.0, n=1)
    frame = _density_frame(np.array([100.0]), np.array([50.0]))

    metrics = error_metrics(frame, fit)

    assert metrics.mape_pct == pytest.approx(100.0, rel=0.01)


def test_error_metrics_reports_hand_measured_length_mae_when_present():
    # El companero natural del error de peso (spec §7.b): el peso no es
    # sensible a donde se corto el pedicelo, asi que esta columna es la
    # unica que valida ese limite. Una de las cinco filas no trae longitud
    # a mano (frecuente: no todas las muestras de banano real se miden con
    # regla) y no debe contarse ni arrastrar un NaN al promedio.
    volume_cm3 = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
    fit = DensityFit(g_per_cm3=0.94, r_squared=1.0, n=5)
    frame = _density_frame(volume_cm3, fit.g_per_cm3 * volume_cm3)
    frame["codex_length_mm"] = [180.0, 185.0, 190.0, 178.0, 200.0]
    frame["actual_length_mm"] = [183.0, 185.0, 187.0, 181.0, np.nan]

    metrics = error_metrics(frame, fit)

    expected = np.mean(np.abs(np.array([180.0, 185.0, 190.0, 178.0]) - np.array([183.0, 185.0, 187.0, 181.0])))
    assert metrics.mae_length_mm == pytest.approx(expected)
    assert metrics.n_length == 4
    # n (muestras de peso) y n_length (muestras de longitud) son conteos
    # INDEPENDIENTES -- este fixture los pone a proposito en valores
    # distintos (5 vs 4) para que confundir uno con el otro sea detectable.
    assert metrics.n == 5


def test_error_metrics_length_fields_are_none_without_hand_measured_length():
    # Validacion solo-con-bascula (sin regla): no debe inventar un numero.
    volume_cm3 = np.array([50.0, 60.0, 70.0])
    fit = DensityFit(g_per_cm3=0.94, r_squared=1.0, n=3)
    frame = _density_frame(volume_cm3, fit.g_per_cm3 * volume_cm3)

    metrics = error_metrics(frame, fit)

    assert metrics.mae_length_mm is None
    assert metrics.n_length == 0


def test_error_metrics_reports_the_sample_count_for_weight_metrics():
    # Round 1, Finding 1: mae_g/mape_pct/r_squared no traian ningun conteo
    # de cuantas filas los respaldan -- justo lo que este modulo existe
    # para impedir. n debe ser la cantidad de filas REALMENTE usadas para
    # las metricas de peso, que aqui es len(frame) porque ninguna fila se
    # excluye de esas metricas (a diferencia de las de longitud).
    volume_cm3 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    fit = DensityFit(g_per_cm3=0.94, r_squared=1.0, n=7)
    frame = _density_frame(volume_cm3, fit.g_per_cm3 * volume_cm3)

    metrics = error_metrics(frame, fit)

    assert metrics.n == 7
    assert metrics.n == len(frame)


def test_error_metrics_raises_on_an_empty_frame():
    # Round 1, Finding 2: un lote vacio es el estado INICIAL de cualquier
    # ValidationLog, antes de la primera muestra -- no un caso de borde
    # remoto. Sin la guarda, np.mean sobre un arreglo vacio avisa con un
    # RuntimeWarning ("Mean of empty slice") en vez de levantar un error
    # claro (ver evidencia de ruptura en el informe de la tarea: quitar la
    # guarda hace que este `pytest.raises` falle).
    fit = DensityFit(g_per_cm3=0.94, r_squared=1.0, n=10)
    empty_frame = _density_frame(np.array([]), np.array([]))

    with pytest.raises(ValueError):
        error_metrics(empty_frame, fit)


def test_error_metrics_raises_on_a_zero_actual_weight():
    # Un banano real nunca pesa 0 g en la bascula -- un cero ahi es un dato
    # mal capturado, no un caso valido, y divide-por-cero en el
    # denominador del MAPE es la misma clase de nan silencioso que un lote
    # vacio. Una sola fila con actual_weight_g == 0.0 en medio de filas
    # normales debe bastar para que todo el lote levante.
    volume_cm3 = np.array([10.0, 20.0, 30.0])
    actual_weight_g = np.array([9.4, 18.8, 0.0])
    fit = DensityFit(g_per_cm3=0.94, r_squared=1.0, n=3)

    with pytest.raises(ValueError):
        error_metrics(_density_frame(volume_cm3, actual_weight_g), fit)

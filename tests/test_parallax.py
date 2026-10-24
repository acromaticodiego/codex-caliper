"""Tests de la correccion de paralaje (Task 9).

Idea central verificada con cuidado, no de pasada: una asercion que solo
pide "el valor corregido es menor que el crudo" pasa con CUALQUIER factor
por debajo de 1, incluido uno mal calculado (una potencia equivocada, un
``h`` sin dividir por dos, etc.). Por eso, ademas de los tests pedidos por
el brief, este archivo:

- fija el factor contra su forma cerrada y contra un numero concreto
  (``test_scale_factor_uses_half_the_object_height``,
  ``test_closed_form_factor_is_the_fixed_point``,
  ``test_applied_factor_matches_the_closed_form_exactly``), y
- pin los EXPONENTES de area y volumen contra invariantes adimensionales
  que cualquier exponente equivocado rompe
  (``test_area_exponent_is_two_not_one``,
  ``test_volume_exponent_is_three_not_two``,
  ``test_area_and_volume_stay_consistent_after_correction``) -- no solo
  "area y volumen bajaron", y
- cierra la trampa del perfil de grosor
  (``test_area_and_volume_recomputed_from_the_corrected_profile_match``):
  recalcular area y volumen integrando el perfil CORREGIDO sobre el tramo
  del fruto tiene que dar los mismos escalares que el modulo reporta. Sobre
  el perfil crudo se iba +13.94% y +21.62%.

``test_closed_form_factor_is_the_fixed_point`` reemplaza al
``test_two_iterations_converge`` del brief, que FALLABA: el esquema
iterativo no converge en dos pasadas, oscila alrededor del punto fijo y dos
pasadas se pasan del limite por 0.082 mm. El test de equivalencia es una
propiedad estrictamente mas fuerte que la convergencia -- demuestra que la
forma cerrada ES el punto fijo, no solo que algo deja de moverse.

Los exponentes se verificaron ademas rompiendolos a proposito: cambiar
``factor ** 3`` por ``factor ** 2`` en ``correct_measures`` hace fallar
``test_volume_exponent_is_three_not_two``,
``test_volume_scales_with_the_cube_of_the_factor`` (el del brief) y
``test_area_and_volume_stay_consistent_after_correction`` -- ver el reporte
de la tarea para la evidencia numerica.
"""

from __future__ import annotations

import numpy as np
import pytest

from codex_caliper.config import AppConfig
from codex_caliper.geometry.measures import (
    _area_mm2,
    _volume_mm3,
    measure_mask,
    trim_pedicel,
)
from codex_caliper.parallax import correct_measures, scale_factor
from tests.fixtures.synthetic import make_banana

RADIUS_MM = 220.0
ARC_RAD = 0.85
WIDTH_MM = 36.0
MM_PER_PX = 0.2

# Forma cerrada sobre los valores de referencia del fixture:
# H_ef = 270.0 - 3.0 = 267.0 y calibre crudo = 36.0, asi que
# factor = 2 * 267 / (2 * 267 + 36) = 534 / 570.
EXPECTED_FACTOR = 534.0 / 570.0  # 0.9368421052631579


def _cfg(**overrides) -> AppConfig:
    # espesor de marcador = 3.0 mm: con working_distance_mm por defecto
    # (270.0), da H_ef = 267.0, el mismo valor que usa el ejemplo del brief
    # para scale_factor.
    base = dict(marker_thickness_mm=3.0)
    base.update(overrides)
    return AppConfig(**base)


@pytest.fixture
def raw():
    """Medidas crudas (sin corregir) de un banano sintetico de referencia."""
    cfg = _cfg()
    mask, _gt = make_banana(RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX)
    m = measure_mask(mask, MM_PER_PX, cfg)
    assert m is not None
    return m


# ---------------------------------------------------------------------------
# Step 1 del brief (con `raw`/`cfg` resueltos via fixture). El cuarto test
# del brief, `test_two_iterations_converge`, esta reemplazado por el test de
# equivalencia de la seccion siguiente: ver docstring del modulo.
# ---------------------------------------------------------------------------


def test_scale_factor_uses_half_the_object_height():
    # la silueta vista desde arriba corresponde a la seccion mas ancha,
    # que esta a media altura del objeto
    assert scale_factor(267.0, 35.0) == pytest.approx((267.0 - 17.5) / 267.0)


def test_flat_object_is_left_untouched():
    assert scale_factor(267.0, 0.0) == 1.0


def test_uncorrected_banana_is_overestimated_by_about_six_percent(raw):
    cfg = _cfg()
    corrected = correct_measures(raw, cfg)
    assert raw.codex_length_mm / corrected.codex_length_mm == pytest.approx(1.065, rel=0.02)


def test_volume_scales_with_the_cube_of_the_factor(raw):
    cfg = _cfg()
    corrected = correct_measures(raw, cfg)
    f = corrected.caliper_mm / raw.caliper_mm
    assert corrected.volume_mm3 == pytest.approx(raw.volume_mm3 * f ** 3, rel=1e-6)


# ---------------------------------------------------------------------------
# El factor: equivalencia con el punto fijo del esquema iterativo, y pin
# contra el numero concreto que sale de la forma cerrada.
# ---------------------------------------------------------------------------


def test_closed_form_factor_is_the_fixed_point(raw):
    """La forma cerrada ES el punto fijo del esquema iterativo.

    Reemplaza al ``test_two_iterations_converge`` del brief, que fallaba (el
    esquema oscila: dos pasadas se pasan del limite por 0.082 mm). Esto es
    mas fuerte que convergencia: se verifica (a) que la forma cerrada
    satisface la ecuacion de punto fijo ``h = c * (H - h/2) / H`` de la que
    se derivo, y (b) que coincide con el LIMITE de la iteracion, replicada
    aqui de forma independiente de ``correct_measures`` (la recurrencia
    escrita a mano, no importada).
    """
    cfg = _cfg()
    h_ef = cfg.effective_height_mm
    c = raw.caliper_mm

    f_closed = 2.0 * h_ef / (2.0 * h_ef + c)

    # (a) la ecuacion de punto fijo se cumple: la altura implicada por el
    #     factor, c * f, es exactamente la altura que produce ese factor.
    assert f_closed == pytest.approx((h_ef - (f_closed * c) / 2.0) / h_ef, rel=1e-15)

    # (b) la iteracion, corrida hasta el fondo, llega al mismo numero.
    f_iter = 1.0
    h = c
    for _ in range(200):
        f_iter = (h_ef - h / 2.0) / h_ef
        h = f_iter * c
    assert f_closed == pytest.approx(f_iter, rel=1e-15)

    # y el limite al que la iteracion tiende NO es lo que daban dos pasadas:
    # el test del brief fallaba por este margen.
    f_two = 1.0
    h = c
    for _ in range(2):
        f_two = (h_ef - h / 2.0) / h_ef
        h = f_two * c
    assert abs(raw.codex_length_mm * (f_two - f_closed)) > 0.05


def test_applied_factor_matches_the_closed_form_exactly(raw):
    # Pin del factor contra el numero concreto de la forma cerrada, no solo
    # "menor que 1" o "menor que el crudo", que cualquier factor mal
    # calculado pasaria igual. Fija ademas que longitud y calibre escalan a
    # la PRIMERA potencia del mismo factor.
    cfg = _cfg()
    corrected = correct_measures(raw, cfg)
    assert corrected.codex_length_mm / raw.codex_length_mm == pytest.approx(
        EXPECTED_FACTOR, rel=1e-12
    )
    assert corrected.caliper_mm / raw.caliper_mm == pytest.approx(EXPECTED_FACTOR, rel=1e-12)
    assert corrected.codex_length_mm == pytest.approx(266.910981, abs=1e-5)


def test_factor_is_not_compounded(raw):
    # El modelo fisico es que existe un UNICO factor verdadero; aplicarlo
    # sobre una magnitud YA corregida compondria correcciones
    # independientes como si el paralaje actuara varias veces sobre la misma
    # escena. Componer dos veces el factor subiria el cociente
    # crudo/corregido de ~1.067 a ~1.14: este test lo distingue fijando el
    # cociente exacto, (2*H_ef + c) / (2*H_ef).
    cfg = _cfg()
    corrected = correct_measures(raw, cfg)
    ratio = raw.codex_length_mm / corrected.codex_length_mm
    assert ratio == pytest.approx(1.0 / EXPECTED_FACTOR, rel=1e-12)
    assert ratio < 1.10


# ---------------------------------------------------------------------------
# Los exponentes, contra invariantes que un exponente equivocado rompe.
# ---------------------------------------------------------------------------


def test_area_exponent_is_two_not_one(raw):
    # Invariante adimensional: area / longitud**2 no depende de la escala,
    # asi que tiene que ser IGUAL antes y despues de corregir -- a menos
    # que el area se reescale al cuadrado del mismo factor que la
    # longitud. Si alguien reescalara area con f**1 (o f**3), este
    # cociente cambiaria y el test fallaria; "el area bajo" no alcanza
    # para distinguir el exponente correcto de uno equivocado, esto si.
    cfg = _cfg()
    corrected = correct_measures(raw, cfg)
    raw_ratio = raw.area_mm2 / raw.codex_length_mm ** 2
    corrected_ratio = corrected.area_mm2 / corrected.codex_length_mm ** 2
    assert corrected_ratio == pytest.approx(raw_ratio, rel=1e-9)


def test_volume_exponent_is_three_not_two(raw):
    # Mismo argumento que el test anterior, para volumen: volumen /
    # longitud**3 es adimensional e invariante de escala solo si volumen
    # escala al CUBO del factor. Verificado rompiendolo: con
    # `factor ** 2` en vez de `factor ** 3` en correct_measures este test
    # falla (0.0097058836 obtenido contra 0.0090928804 esperado, un 6.7%
    # de desvio contra una tolerancia de 1e-9) -- ver el reporte.
    cfg = _cfg()
    corrected = correct_measures(raw, cfg)
    raw_ratio = raw.volume_mm3 / raw.codex_length_mm ** 3
    corrected_ratio = corrected.volume_mm3 / corrected.codex_length_mm ** 3
    assert corrected_ratio == pytest.approx(raw_ratio, rel=1e-9)


def test_area_and_volume_stay_consistent_after_correction(raw):
    # La consistencia que Task 8 fijo a rel=1e-6 entre area y volumen
    # (las dos integran el mismo perfil sobre el mismo tramo) tiene que
    # sobrevivir a la correccion: escalar area por f**2 y volumen por
    # f**3 (cada uno a SU exponente dimensional correcto) es precisamente
    # lo que preserva volume/area**1.5 (adimensional, invariante de
    # escala). Si se escalaran de forma inconsistente (p.ej. los dos por
    # el mismo exponente), esta razon cambiaria.
    cfg = _cfg()
    corrected = correct_measures(raw, cfg)
    raw_shape = raw.volume_mm3 / raw.area_mm2 ** 1.5
    corrected_shape = corrected.volume_mm3 / corrected.area_mm2 ** 1.5
    assert corrected_shape == pytest.approx(raw_shape, rel=1e-6)


# ---------------------------------------------------------------------------
# Perfil de grosor y degeneraciones.
# ---------------------------------------------------------------------------


def test_width_profile_is_scaled_on_both_axes(raw):
    # Abscisa Y anchos, las dos a la PRIMERA potencia del mismo factor:
    # las dos son longitudes medidas en el plano de calibracion y las dos
    # cargan la misma sobreestimacion. Reescalar una sola dejaria el perfil
    # en unidades mezcladas.
    cfg = _cfg()
    corrected = correct_measures(raw, cfg)
    raw_s, raw_w = raw.width_profile
    corrected_s, corrected_w = corrected.width_profile

    np.testing.assert_allclose(corrected_s, raw_s * EXPECTED_FACTOR, rtol=1e-12)
    np.testing.assert_allclose(corrected_w, raw_w * EXPECTED_FACTOR, rtol=1e-6)

    # el dtype de cada eje se conserva (s_mm float64, w_mm float32): el
    # perfil corregido no cuesta mas memoria que el crudo.
    assert corrected_s.dtype == raw_s.dtype
    assert corrected_w.dtype == raw_w.dtype


def test_area_and_volume_recomputed_from_the_corrected_profile_match(raw):
    """El perfil corregido y los escalares corregidos son la MISMA figura.

    Este es el test que cierra la trampa: un consumidor que reproduce
    ``trim_pedicel`` e integra el perfil sobre el tramo del fruto -- el
    calculo exacto que ``_area_mm2`` y ``_volume_mm3`` ya hacen -- tiene que
    aterrizar en los mismos ``area_mm2`` y ``volume_mm3`` que
    ``correct_measures`` reporta. Con el perfil CRUDO se iba +13.94% en area
    y +21.62% en volumen, en silencio.

    Reescalar los DOS ejes es lo que lo hace salir solo: ``integral(w ds)``
    escala con f**2 y ``integral(pi*(w/2)**2 ds)`` con f**3, que son
    justamente los exponentes con que se corrigen los dos escalares.

    Tolerancia ``rel=1e-6`` y no mas apretada porque ``w_mm`` es float32
    (eps 1.2e-7): el residuo medido es ~3e-8 en area y ~5e-8 en volumen,
    puro redondeo de float32 al multiplicar por el factor. Sigue siendo
    cuatro ordenes de magnitud mas ajustado que la discrepancia del 14-22%
    que este test existe para prevenir, y es la misma tolerancia con la que
    Task 8 fijo la consistencia entre area y volumen.
    """
    cfg = _cfg()
    corrected = correct_measures(raw, cfg)
    s_mm, w_mm = corrected.width_profile

    start, end = trim_pedicel(s_mm, w_mm, cfg.pedicel_drop_ratio)
    abscissa_mm = np.asarray(s_mm[start : end + 1], dtype=np.float64)

    assert _area_mm2(abscissa_mm, w_mm, start, end) == pytest.approx(
        corrected.area_mm2, rel=1e-6
    )
    assert _volume_mm3(abscissa_mm, w_mm, start, end) == pytest.approx(
        corrected.volume_mm3, rel=1e-6
    )


def test_trim_lands_on_the_same_segment_before_and_after_correction(raw):
    # El criterio de trim_pedicel compara |dw/ds| / w_max (unidades 1/mm)
    # contra un umbral ABSOLUTO, y esa cantidad escala con 1/f: sobre el
    # perfil corregido crece ~6.7%. El umbral sigue bracketeando los dos
    # casos de la calibracion de Task 8, asi que el recorte cae en los
    # mismos indices -- de lo cual depende el test anterior. Si una
    # recalibracion futura de _DERIVATIVE_THRESHOLD_PER_MM rompe eso, este
    # test lo dice directamente en vez de dejarlo aparecer como una
    # discrepancia inexplicable de area.
    cfg = _cfg()
    corrected = correct_measures(raw, cfg)
    assert trim_pedicel(*corrected.width_profile, cfg.pedicel_drop_ratio) == trim_pedicel(
        *raw.width_profile, cfg.pedicel_drop_ratio
    )


def test_degenerate_measures_returns_none_without_raising():
    cfg = _cfg()
    assert correct_measures(None, cfg) is None


def test_misconfigured_rig_does_not_raise(raw):
    # marker_thickness_mm >= working_distance_mm es un montaje mal medido
    # (H_ef <= 0): la correccion degenera a 1.0 (sin corregir) en vez de
    # dividir por cero o devolver un factor negativo.
    cfg = _cfg(working_distance_mm=3.0, marker_thickness_mm=3.0)
    corrected = correct_measures(raw, cfg)
    assert corrected.codex_length_mm == pytest.approx(raw.codex_length_mm)
    assert corrected.caliper_mm == pytest.approx(raw.caliper_mm)
    assert corrected.area_mm2 == pytest.approx(raw.area_mm2)
    assert corrected.volume_mm3 == pytest.approx(raw.volume_mm3)

"""Tests del recorte de pedicelo, calibre, area y volumen (Task 8).

Ideas verificadas con cuidado, no de pasada:

1. Que ``trim_pedicel`` distingue un adelgazamiento GRADUAL (extremo floral)
   de una transicion ABRUPTA (union con el pedicelo) por la DERIVADA de
   ``w(s)``, no por su valor absoluto -- ``test_trim_fires_across_fruit_sizes``
   y ``test_gradual_blossom_taper_is_not_trimmed`` son un par point a point:
   si un umbral absoluto de ancho bastara para pasar los dos, el fixture
   seria demasiado permisivo. Ver
   ``test_absolute_threshold_would_wrongly_trim_the_gradual_taper``, que
   reproduce deliberadamente ese criterio mas simple (ingenuo) y confirma
   que SI falla al distinguirlos -- la derivada es necesaria, no un adorno.

2. Que el criterio de corte funciona en MAS DE UN calibre de fruto, no solo
   en el banano de referencia -- ``test_trim_fires_across_fruit_sizes``
   parametriza w=36/pedicelo 9, w=24/pedicelo 6 y w=12/pedicelo 4 (ronda de
   revision 1, finding 1).

3. Que el recorte efectivamente elimina (de forma medible, no solo
   booleana) la longitud del pedicelo de la curva exterior --
   ``test_trimming_removes_approximately_the_pedicel_length`` compara la
   curva exterior SIN recortar contra la recortada, en la MISMA mascara,
   en los mismos tres calibres (ronda de revision 1, finding 2).

Nota sobre la ronda de revision 1
----------------------------------
La primera version de este archivo comparaba un banano con pedicelo
RECORTADO contra un banano SIN pedicelo (dos mascaras distintas). La
revision encontro que esa comparacion es mal planteada: el extremo del
pedicelo, en el fixture, conserva la MISMA tapa redondeada completa que
tendria el fruto sin pedicelo (ver docstring de
``tests/fixtures/synthetic.py``), y un corte de plano a traves de esa tapa
solo puede conservar APROXIMADAMENTE la mitad de su contribucion al
contorno -- el banano sin pedicelo conserva toda la tapa. Comparar las dos
mascaras mide esa diferencia de tapas, no (solo) si el recorte funciono.
Medido: media tapa es ``pi*w/4 = 28.27 mm`` con w=36, un 9.9% de la
longitud del banano sin pedicelo -- del mismo orden que la brecha que la
primera version de este archivo intentaba tapar ensanchando la tolerancia.

Esta version compara, en cambio, la MISMA mascara con pedicelo sin recortar
contra si misma recortada: ambos lados de la comparacion comparten
exactamente la misma tapa del fruto (y, cuando aplica, el mismo
adelgazamiento floral), asi que esa tapa se cancela y lo que queda es,
precisamente, la longitud que el recorte quito -- que debe aproximarse a
``pedicel_mm``. Medido en w=36 (pedicelo 25/9), w=24 (pedicelo 25/6), w=12
(pedicelo 25/4), w=12 (pedicelo 25/3) y w=8 (pedicelo 25/3): la diferencia
cae entre 82% y 105% de los 25 mm del pedicelo. ``PEDICEL_REMOVAL_TOLERANCE
= 0.20`` cubre ese rango real con margen en ambos extremos; no es una
tolerancia elegida para que el test pase, es la dispersion medida. La
causa de que no sea exactamente 100%: el corte no cae en el borde
geometrico exacto entre pedicelo y fruto, sino donde ``trim_pedicel``
confirma que el ancho ya se estabilizo cerca del maximo (ver docstring de
``codex_caliper.geometry.measures``) -- la zona de transicion en si misma
tiene una extension fisica comparable al semi-calibre local, no es un
corte de navaja, asi que "la longitud del pedicelo" tiene un margen de
interpretacion de ese orden incluso en la geometria pura.

La causa RAIZ de por que la primera implementacion fallaba en calibres
chicos (w=12/pedicelo 3, w=8/pedicelo 3: el banano recortado media 14-25%
MAS largo que el banano sin pedicelo, en vez de mas corto) no era el umbral
de la derivada -- ``trim_pedicel`` identificaba el indice de corte correcto
en todos los calibres probados (verificado por separado, inspeccionando
``start``/``end`` directamente). El defecto estaba en como ``measure_mask``
traducia ese indice en una curva exterior: volvia a partir el CONTORNO
CERRADO COMPLETO usando el punto recortado del eje como si fuera una punta
nueva, y esa busqueda de "punto de contorno mas cercano" es ambigua/inestable
en un punto interior del cuerpo (no en una punta real) -- para calibres
chicos, elegia el lado equivocado del contorno y dejaba todo el pedicelo
adentro. La correccion (``_restrict_outer_to_fruit`` en
``codex_caliper.geometry.measures``) ya no vuelve a partir el contorno
completo; recorta la curva exterior ya seleccionada de forma robusta con
las puntas ORIGINALES del eje. Aun asi, ``_SMOOTH_WINDOW_MM`` (la ventana
de suavizado de la derivada, ahora en milimetros en vez de muestras) se
corrigio en la misma ronda porque un numero fijo de MUESTRAS no es una
cantidad invariante de escala (depende de la resolucion del esqueleto y de
``mm_per_px``), aunque no era la causa de este defecto en particular.

Nota sobre la ronda de revision 2
----------------------------------
Dos cambios, ninguno sobre el recorte en si:

1. ``area_mm2`` ahora se restringe al tramo del fruto, igual que
   ``codex_length_mm`` y ``volume_mm3`` -- antes contaba los pixeles de la
   mascara COMPLETA (pedicelo incluido), una inconsistencia entre las
   cuatro magnitudes que la revision señalo. Ver
   ``test_area_is_restricted_to_the_fruit_when_pedicel_is_present``.
2. ``measure_mask`` ya no envuelve toda la funcion en un
   ``except Exception: return None``. Ese ``except`` generico convertia
   cualquier error de programacion futuro en esta cadena (un typo, una
   firma que cambio) en el mismo ``None`` silencioso que una mascara
   degenerada -- indistinguible de "no hay nada que medir". La
   degeneracion geometrica ahora se chequea explicitamente en cada paso
   (igual que antes, pero sin la red de seguridad generica detras); un
   error de programacion real se propaga. Ver
   ``test_a_programming_error_propagates_instead_of_returning_none``.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from codex_caliper.config import AppConfig
from codex_caliper.geometry.measures import (
    BananaMeasures,
    measure_mask,
    touches_border,
    trim_pedicel,
)
from codex_caliper.geometry.medial import medial_axis_path, width_profile_mm
from codex_caliper.geometry.outline import outer_curve, smoothed_arc_length_mm
from tests.fixtures.synthetic import make_banana

RADIUS_MM = 220.0
ARC_RAD = 0.85
WIDTH_MM = 36.0
MM_PER_PX = 0.2
PEDICEL_MM = 25.0

# Ver nota del docstring del modulo: dispersion REAL medida (82%-105% de
# pedicel_mm) a lo largo de cinco combinaciones de calibre/pedicelo, no un
# numero elegido para que el test pase.
PEDICEL_REMOVAL_TOLERANCE = 0.20

# Geometrias pedidas explicitamente en la ronda de revision 1, finding 1:
# w=36/pedicelo 9 (referencia), w=24/pedicelo 6, w=12/pedicelo 4.
FRUIT_GEOMETRIES = [
    (36.0, 9.0),
    (24.0, 6.0),
    (12.0, 4.0),
]


def _cfg(**overrides) -> AppConfig:
    base = dict(marker_thickness_mm=2.0)
    base.update(overrides)
    return AppConfig(**base)


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(np.float64)


def test_caliper_matches_ground_truth_within_half_a_millimetre():
    cfg = _cfg()
    mask, gt = make_banana(RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX)
    m = measure_mask(mask, MM_PER_PX, cfg)
    assert m is not None
    assert m.caliper_mm == pytest.approx(gt.caliber_mm, abs=0.5)


def test_area_and_volume_match_ground_truth_within_3_percent():
    # Banano SIN pedicelo: start=0, end=n-1, asi que el area "del fruto"
    # (ronda de revision 2) coincide con el area de la mascara completa --
    # este test no cambia de comportamiento, solo de que camino de codigo
    # lo calcula. Ver test_area_is_restricted_to_the_fruit_when_pedicel_is_present
    # para el caso donde SI hay diferencia.
    cfg = _cfg()
    mask, gt = make_banana(RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX)
    m = measure_mask(mask, MM_PER_PX, cfg)
    assert m is not None
    assert m.area_mm2 == pytest.approx(gt.area_mm2, rel=0.03)
    assert m.volume_mm3 == pytest.approx(gt.volume_mm3, rel=0.03)


@pytest.mark.parametrize("width_mm,pedicel_width_mm", FRUIT_GEOMETRIES)
def test_area_is_restricted_to_the_fruit_when_pedicel_is_present(width_mm, pedicel_width_mm):
    # Ronda de revision 2, finding 2: area_mm2 debe describir el mismo
    # tramo que codex_length_mm y volume_mm3 (el fruto), no la mascara
    # completa -- reportar un area que incluye el tallo junto a una
    # longitud que lo excluye no describe el mismo objeto.
    cfg = _cfg()
    mask, _gt = make_banana(
        RADIUS_MM, ARC_RAD, width_mm, MM_PER_PX,
        pedicel_mm=PEDICEL_MM, pedicel_width_mm=pedicel_width_mm,
    )
    full_mask_area_mm2 = float(np.count_nonzero(mask)) * (MM_PER_PX ** 2)

    m = measure_mask(mask, MM_PER_PX, cfg)
    assert m is not None

    # el area del fruto es estrictamente menor que la de la mascara
    # completa (se quito algo, no es un no-op)...
    assert m.area_mm2 < full_mask_area_mm2
    # ...por una cantidad que no es ruido numerico: el pedicelo mas la
    # porcion de la transicion que el corte tambien quita son una fraccion
    # real del area total, no un redondeo de un par de pixeles.
    removed_fraction = (full_mask_area_mm2 - m.area_mm2) / full_mask_area_mm2
    assert removed_fraction > 0.02


def test_volume_requires_the_hemisphere_cap_correction():
    # Finding 3, ronda de revision 1: la correccion de casquete esferico se
    # acepto como correcta, no como un parche -- este test prueba que es
    # necesaria. El eje medial se detiene en el CENTRO de cada tapa
    # redondeada (ver medial.py), asi que integrar unicamente
    # pi*(w/2)**2*ds sobre el perfil crudo, SIN los casquetes, subestima el
    # volumen real. Si alguien quitara la correccion de
    # codex_caliper.geometry.measures._volume_mm3, este test fallaria
    # (la integral sola queda a ~7.7% del valor real, fuera del 3% que
    # exige test_area_and_volume_match_ground_truth_within_3_percent).
    mask, gt = make_banana(RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX)
    axis = medial_axis_path(mask)
    assert axis is not None
    s_mm, w_mm = width_profile_mm(axis, MM_PER_PX)
    integrand = np.pi * (w_mm / 2.0) ** 2
    volume_without_caps = float(np.trapezoid(integrand, s_mm))

    cfg = _cfg()
    m = measure_mask(mask, MM_PER_PX, cfg)
    assert m is not None

    # la version SIN casquetes se queda notoriamente corta...
    assert volume_without_caps < gt.volume_mm3 * 0.95
    # ...mientras que measure_mask, CON la correccion, cae dentro del 3%
    # exigido (ya verificado arriba, repetido aqui para que el contraste
    # quede explicito en este mismo test).
    assert m.volume_mm3 == pytest.approx(gt.volume_mm3, rel=0.03)


@pytest.mark.parametrize("width_mm,pedicel_width_mm", FRUIT_GEOMETRIES)
def test_trim_fires_across_fruit_sizes(width_mm, pedicel_width_mm):
    # Ronda de revision 1, finding 1: el criterio tenia que dispararse en
    # mas de un calibre de fruto, no solo en el banano de referencia.
    cfg = _cfg()
    mask, _gt = make_banana(
        RADIUS_MM, ARC_RAD, width_mm, MM_PER_PX,
        pedicel_mm=PEDICEL_MM, pedicel_width_mm=pedicel_width_mm,
    )
    axis = medial_axis_path(mask)
    assert axis is not None
    s_mm, w_mm = width_profile_mm(axis, MM_PER_PX)
    start, end = trim_pedicel(s_mm, w_mm, cfg.pedicel_drop_ratio)

    # el pedicelo (abrupto) se recorta...
    assert start > 0
    # ...y el extremo floral (sin adelgazamiento aqui, tapa redondeada
    # completa) no se toca.
    assert end == len(w_mm) - 1


@pytest.mark.parametrize("width_mm,pedicel_width_mm", FRUIT_GEOMETRIES)
def test_trimming_removes_approximately_the_pedicel_length(width_mm, pedicel_width_mm):
    # Ronda de revision 1, finding 2: comparacion DENTRO de la misma
    # mascara (sin recortar contra recortada), no contra una mascara
    # distinta sin pedicelo -- ver docstring del modulo para por que esa
    # segunda comparacion esta mal planteada.
    cfg = _cfg()
    mask, _gt = make_banana(
        RADIUS_MM, ARC_RAD, width_mm, MM_PER_PX,
        pedicel_mm=PEDICEL_MM, pedicel_width_mm=pedicel_width_mm,
    )
    axis = medial_axis_path(mask)
    assert axis is not None
    contour = _largest_contour(mask)
    untrimmed_length = smoothed_arc_length_mm(outer_curve(contour, axis.tips_px), MM_PER_PX)

    m = measure_mask(mask, MM_PER_PX, cfg)
    assert m is not None

    removed = untrimmed_length - m.codex_length_mm
    assert removed == pytest.approx(PEDICEL_MM, rel=PEDICEL_REMOVAL_TOLERANCE)


def test_gradual_blossom_taper_is_not_trimmed():
    # el extremo floral se adelgaza de forma gradual y debe conservarse;
    # solo la caida abrupta del pedicelo se corta.
    cfg = _cfg()
    tapered_with_pedicel, _ = make_banana(
        RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX,
        pedicel_mm=PEDICEL_MM, pedicel_width_mm=9.0, blossom_taper_mm=30.0,
    )

    axis = medial_axis_path(tapered_with_pedicel)
    assert axis is not None
    s_mm, w_mm = width_profile_mm(axis, MM_PER_PX)
    start, end = trim_pedicel(s_mm, w_mm, cfg.pedicel_drop_ratio)
    # el extremo floral (gradual) no se toca: el indice final del tramo del
    # fruto sigue siendo la ultima muestra del perfil.
    assert end == len(w_mm) - 1
    # mientras que el extremo del pedicelo (abrupto) si se recorta -- sin
    # esta asercion, "nunca recortar nada" pasaria la linea de arriba por
    # las razones equivocadas.
    assert start > 0

    # Comparacion DENTRO de la misma mascara (ver docstring del modulo):
    # sin recortar contra recortada, ambas comparten el MISMO
    # adelgazamiento floral, asi que si el recorte comiera parte de ese
    # extremo gradual por error, "removed" se alejaria bastante mas de
    # pedicel_mm de lo que la tolerancia medida permite.
    contour = _largest_contour(tapered_with_pedicel)
    untrimmed_length = smoothed_arc_length_mm(outer_curve(contour, axis.tips_px), MM_PER_PX)
    m = measure_mask(tapered_with_pedicel, MM_PER_PX, cfg)
    assert m is not None
    removed = untrimmed_length - m.codex_length_mm
    assert removed == pytest.approx(PEDICEL_MM, rel=PEDICEL_REMOVAL_TOLERANCE)


def test_absolute_threshold_would_wrongly_trim_the_gradual_taper():
    # Prueba deliberada de ruptura (sin cambios en esta ronda): un criterio
    # NAIVE que solo mira el ancho absoluto (sin derivada) no distingue el
    # adelgazamiento floral gradual de la transicion abrupta del pedicelo
    # -- ambos cruzan el mismo umbral absoluto de ancho tarde o temprano.
    # Confirma por que trim_pedicel necesita la derivada, no solo
    # drop_ratio * w_max.
    mask, _gt = make_banana(
        RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX, blossom_taper_mm=30.0
    )
    axis = medial_axis_path(mask)
    assert axis is not None
    _s_mm, w_mm = width_profile_mm(axis, MM_PER_PX)
    w_max = float(np.max(w_mm))
    threshold_w = 0.45 * w_max

    # el criterio ingenuo: el extremo floral gradual tambien cae por debajo
    # del umbral absoluto en algun punto (es un adelgazamiento real, solo
    # que gradual) -- un umbral puramente absoluto lo "encontraria" igual
    # que encuentra el pedicelo, confirmando que la derivada es necesaria.
    assert w_mm[0] < threshold_w


def test_mask_touching_the_canvas_border_is_rejected():
    cfg = _cfg()
    mask_cut_by_edge = np.zeros((80, 80), np.uint8)
    mask_cut_by_edge[0:40, 0:40] = 255  # toca la esquina superior izquierda
    assert touches_border(mask_cut_by_edge) is True
    assert measure_mask(mask_cut_by_edge, MM_PER_PX, cfg) is None


def test_mask_away_from_the_border_is_not_rejected():
    mask, _gt = make_banana(RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX)
    assert touches_border(mask) is False


def test_degenerate_mask_returns_none_without_raising():
    cfg = _cfg()
    assert measure_mask(np.zeros((80, 80), np.uint8), MM_PER_PX, cfg) is None


def test_a_programming_error_propagates_instead_of_returning_none(monkeypatch):
    # Ronda de revision 2, finding 1: measure_mask debe distinguir
    # degeneracion GEOMETRICA (mascara vacia, que toca el borde, sin dos
    # puntas -- todos cubiertos arriba, siempre devuelven None) de un error
    # de PROGRAMACION real en la cadena de llamadas. Antes de esta ronda,
    # un ``except Exception`` generico envolvia toda la funcion y
    # convertia cualquier bug futuro en este mismo None silencioso --
    # indistinguible, para quien llama, de "esta mascara no es medible".
    # Este test rompe deliberadamente una llamada interna (no una
    # degeneracion geometrica: la mascara es un banano perfectamente
    # valido) y confirma que la excepcion se propaga, no que measure_mask
    # la trague.
    import codex_caliper.geometry.measures as measures_module

    def _broken_width_profile_mm(*_args, **_kwargs):
        raise AttributeError("bug simulado deliberadamente, no degeneracion geometrica")

    monkeypatch.setattr(measures_module, "width_profile_mm", _broken_width_profile_mm)

    cfg = _cfg()
    mask, _gt = make_banana(RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX)
    with pytest.raises(AttributeError):
        measures_module.measure_mask(mask, MM_PER_PX, cfg)

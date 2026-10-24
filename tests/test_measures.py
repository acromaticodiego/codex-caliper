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

Nota sobre la ronda de revision 3
----------------------------------
El area ya no se cuenta del raster: se INTEGRA del perfil de grosor sobre
el mismo tramo ``[start, end]`` que el volumen (ver docstring de
``codex_caliper.geometry.measures``, ronda de revision 3). La forma en que
la ronda 2 la restringia al fruto -- recortar el raster con un semiplano
que pasaba por el punto de recorte, evaluado sobre TODO el lienzo -- borra
fruto real en un banano curvo, porque ese plano vuelve a cortar el cuerpo
del otro lado del cuadro.

Lo relevante para este archivo es que
``test_area_is_restricted_to_the_fruit_when_pedicel_is_present`` no podia
detectarlo: solo exigia ``removed_fraction > 0.02``, y el bug quitaba
DEMASIADO (-19.4%, -13.3% y -6.8% del area analitica en w=36/24/12), asi
que cuanto peor era el bug, con mas comodidad pasaba el test. Lo reemplaza
``test_fruit_area_matches_the_analytic_fruit_area_with_a_pedicel_present``,
que compara contra un valor en FORMA CERRADA (``gt.area_mm2`` menos el
medio disco que el plano de corte deja afuera, ver el comentario del test)
con ``rel=0.03`` de DOS LADOS -- falla si el area sobra y si falta.
Verificado contra la implementacion de la ronda 2: los tres calibres
fallan (-13.70%, -9.18%, -4.51% contra ese valor esperado).

Se agrega ademas ``test_area_and_volume_agree_about_the_fruit_segment``:
despeja de cada magnitud (ya sin sus terminos de tapa) la longitud del
tramo que integro y exige que las dos longitudes implicitas coincidan
entre si y con ``arc_length_px`` del tramo recortado del eje, con
``rel=1e-6``. Antes de esta ronda area y volumen se contradecian sobre
donde termina el fruto (hasta 19% de diferencia en el area con el volumen
dentro del 2%) y ningun test lo notaba. Verificado que falla tanto con el
conteo de pixeles de la ronda 2 como con una sola de las dos magnitudes
integrada sobre la abscisa cruda del esqueleto.

Nota sobre la ronda de revision 4
----------------------------------
La correccion del sesgo de escalera de la abscisa se movio de este modulo a
``width_profile_mm`` (Task 6), donde la reciben todos los consumidores del
perfil y no solo Task 8 -- ver ``tests/test_medial.py::
test_width_profile_abscissa_matches_the_analytic_centerline``, que fija la
abscisa contra la longitud central analitica dentro del 2% en w=36, 24 y 12.
``measure_mask`` ya no reescala nada: recorta ``s_mm`` al tramo del fruto y
pasa ESE arreglo a las dos integrales.

El fixture de w=12 se conserva en los tests que verifican que el RECORTE se
dispara en cualquier calibre (``test_trim_fires_across_fruit_sizes``,
``test_trimming_removes_approximately_the_pedicel_length``), que es donde
corresponde: son tests de robustez del criterio, no de exactitud metrica.
Pero no se le exige una verdad analitica de VOLUMEN, porque un banano de 12
mm de calibre esta fuera del rango de operacion del CODEX
(minimo del estandar, ``CODEX_MIN_CALIPER_MM = 27.0`` en el plan del
proyecto) y ahi domina el ruido de rasterizado
subpixel del ancho, que no es un sesgo corregible -- ver el docstring de
``codex_caliper.geometry.measures`` para los numeros medidos.
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
from codex_caliper.geometry.medial import arc_length_px, medial_axis_path, width_profile_mm
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
def test_fruit_area_matches_the_analytic_fruit_area_with_a_pedicel_present(width_mm, pedicel_width_mm):
    # Ronda de revision 3: el test anterior solo pedia
    # "removed_fraction > 0.02" -- es decir, solo que el area BAJARA. El bug
    # que reemplaza (recortar el RASTER con un semiplano evaluado sobre todo
    # el lienzo, que en un banano curvo vuelve a cortar el cuerpo del otro
    # lado del cuadro) quitaba DEMASIADO: -19.4%, -13.3% y -6.8% contra la
    # verdad analitica en estos tres calibres. Un test que solo mira "bajo"
    # lo pasaba con mas comodidad cuanto peor era el bug. Este lo reemplaza
    # por una comparacion de DOS LADOS contra un valor en forma cerrada.
    #
    # El objetivo: el fixture es una capsula, cuya area exacta es
    # ``w*R*theta + pi*(w/2)**2`` (rectangulo barrido + dos medios discos,
    # ver tests/fixtures/synthetic.py) -- y gt.area_mm2 describe la capsula
    # PURA, sin el pedicelo, que es exactamente el fruto que se quiere
    # medir. La unica diferencia conocida y cuantificable: el corte del
    # pedicelo es un PLANO que pasa por la base del pedicelo, donde el
    # fixture estampa igual una tapa redondeada completa del fruto (su
    # docstring lo dice); el plano se queda con la mitad de esa tapa, asi
    # que el fruto recortado mide un MEDIO disco menos que la capsula
    # entera. Ese termino es tambien forma cerrada, no una tolerancia:
    # expected = gt.area_mm2 - 0.5*pi*(w/2)**2.
    cfg = _cfg()
    mask, gt = make_banana(
        RADIUS_MM, ARC_RAD, width_mm, MM_PER_PX,
        pedicel_mm=PEDICEL_MM, pedicel_width_mm=pedicel_width_mm,
    )

    m = measure_mask(mask, MM_PER_PX, cfg)
    assert m is not None

    half_cap_mm2 = 0.5 * np.pi * (width_mm / 2.0) ** 2
    expected_fruit_area_mm2 = gt.area_mm2 - half_cap_mm2
    # dos lados: el area tiene que caer DENTRO del 3% por arriba Y por
    # abajo. Medido: -0.3% (w=36), -0.5% (w=24), -1.6% (w=12).
    assert m.area_mm2 == pytest.approx(expected_fruit_area_mm2, rel=0.03)

    # y, por separado, que sigue siendo estrictamente menor que la mascara
    # completa: el pedicelo no entra (la direccion que el test anterior
    # verificaba, conservada).
    full_mask_area_mm2 = float(np.count_nonzero(mask)) * (MM_PER_PX ** 2)
    assert m.area_mm2 < full_mask_area_mm2


@pytest.mark.parametrize("pedicel_width_mm", [0.0, 9.0])
def test_area_and_volume_agree_about_the_fruit_segment(pedicel_width_mm):
    # Ronda de revision 3: area y volumen tienen que describir el MISMO
    # tramo del fruto. Antes no lo hacian -- el area contaba pixeles de un
    # raster recortado por un semiplano global (que borraba fruto real del
    # otro lado del cuadro) y el volumen integraba localmente sobre
    # [start, end], asi que los dos numeros se contradecian sobre donde
    # termina el fruto: hasta 19% de diferencia en el area mientras el
    # volumen quedaba dentro del 2%.
    #
    # Ahora los dos se integran del MISMO perfil sobre el MISMO tramo y la
    # MISMA abscisa, y eso es lo que este test fija: se le quitan a cada
    # magnitud sus terminos de tapa (medio disco / casquete hemisferico, que
    # solo existen en un extremo SIN recortar) y se despeja de cada una la
    # longitud del tramo que integro. Las dos longitudes implicitas tienen
    # que coincidir entre si, y con la media de w sobre [start, end] --
    # area/longitud = ancho medio, que es la forma que pide la revision.
    # Si alguien volviera a calcular el area sobre otra region (el raster
    # completo, un recorte distinto, otro tramo de indices), estas dos
    # longitudes implicitas se separarian y el test fallaria.
    cfg = _cfg()
    mask, _gt = make_banana(
        RADIUS_MM, ARC_RAD, WIDTH_MM, MM_PER_PX,
        pedicel_mm=PEDICEL_MM if pedicel_width_mm > 0.0 else 0.0,
        pedicel_width_mm=pedicel_width_mm,
    )
    axis = medial_axis_path(mask)
    assert axis is not None
    s_mm, w_mm = width_profile_mm(axis, MM_PER_PX)
    start, end = trim_pedicel(s_mm, w_mm, cfg.pedicel_drop_ratio)
    n = len(w_mm)

    m = measure_mask(mask, MM_PER_PX, cfg)
    assert m is not None

    # terminos de tapa de cada extremo SIN recortar, con el radio local de
    # ese extremo (medio disco para el area, casquete hemisferico para el
    # volumen -- ver codex_caliper.geometry.measures).
    cap_area_mm2 = 0.0
    cap_volume_mm3 = 0.0
    for index, trimmed in ((0, start > 0), (n - 1, end < n - 1)):
        if trimmed:
            continue
        r = float(w_mm[index]) / 2.0
        cap_area_mm2 += 0.5 * np.pi * r ** 2
        cap_volume_mm3 += (2.0 / 3.0) * np.pi * r ** 3

    swept_area_mm2 = m.area_mm2 - cap_area_mm2
    swept_volume_mm3 = m.volume_mm3 - cap_volume_mm3

    # media de w y del disco pi*(w/2)**2 sobre [start, end], ponderada por
    # la misma regla trapezoidal que usa la implementacion.
    segment_s = s_mm[start : end + 1]
    segment_w = w_mm[start : end + 1]
    segment_span_mm = float(segment_s[-1] - segment_s[0])
    mean_width_mm = float(np.trapezoid(segment_w, segment_s)) / segment_span_mm
    mean_disc_mm2 = float(np.trapezoid(np.pi * (segment_w / 2.0) ** 2, segment_s)) / segment_span_mm

    # longitud del tramo que cada magnitud integro, despejada de ella misma
    implied_length_from_area_mm = swept_area_mm2 / mean_width_mm
    implied_length_from_volume_mm = swept_volume_mm3 / mean_disc_mm2

    # area / longitud == ancho medio sobre [start, end], y lo mismo para el
    # volumen: las dos longitudes implicitas son la MISMA longitud.
    assert implied_length_from_area_mm == pytest.approx(
        implied_length_from_volume_mm, rel=1e-6
    )
    # y esa longitud es la del tramo [start, end] de la abscisa que entrega
    # width_profile_mm (ya corregida del sesgo de escalera en Task 6, ronda
    # de revision 4), no la del eje completo ni la del raster.
    assert implied_length_from_area_mm == pytest.approx(segment_span_mm, rel=1e-6)

    # Y, de forma independiente del perfil: esa misma longitud es la que
    # mide la primitiva de Task 6 sobre el tramo recortado del camino. No
    # coincide al bit porque width_profile_mm reescala la abscisa con el
    # factor de escalera del camino COMPLETO (una sola correccion para todos
    # los consumidores) mientras que esto lo recalcula sobre el sub-tramo;
    # medido, la diferencia es +0.009% (w=36 sin pedicelo y con el) y hasta
    # 0.14% en los calibres chicos, muy por debajo de los 6% del sesgo que
    # esta asercion vigila.
    fruit_axis_length_mm = arc_length_px(axis.path_px[start : end + 1]) * MM_PER_PX
    assert implied_length_from_area_mm == pytest.approx(fruit_axis_length_mm, rel=0.005)


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

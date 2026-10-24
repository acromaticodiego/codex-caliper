"""Recorte del pedicelo, calibre, area y volumen (Task 8).

Este modulo cierra la cadena de medicion: a partir del perfil de grosor
``w(s)`` (Task 6) y la curva exterior (Task 7), identifica donde termina el
pedicelo y empieza el fruto, y combina las piezas en las magnitudes que el
CODEX STAN 205-1997 exige -- longitud, calibre, area y volumen.

Recorte del pedicelo (``trim_pedicel``)
---------------------------------------
El CODEX mide la longitud "desde el extremo floral hasta la base del
pedicelo donde termina la pulpa comestible" -- el pedicelo no cuenta. Pero
los dos extremos de ``w(s)`` se adelgazan: el floral de forma GRADUAL (el
banano se afina hacia la punta) y la union con el pedicelo de forma ABRUPTA
(el ancho cae de golpe del calibre del fruto al del pedicelo, mucho mas
fino). Por eso el criterio de corte usa la DERIVADA normalizada de ``w``, no
su valor absoluto: un umbral absoluto de ancho recortaria ambos extremos por
igual y acortaria el fruto por el lado floral.

Calibrado contra el banano sintetico de referencia (radius=220, arc=0.85,
width=36, mm_per_px=0.2, ver ``tests/test_measures.py``). Los numeros de
abajo son POSTERIORES a la correccion de la abscisa de la ronda de revision
4 (ver mas abajo): la abscisa se acorto un 5.5%, asi que toda derivada
``dw/ds`` crecio en la misma proporcion y la calibracion previa ya no
describia el codigo. Medidos sobre el perfil SUAVIZADO (``_smooth`` con
``_SMOOTH_WINDOW_MM``), con la derivada normalizada ``|dw/ds| / w_max`` en
1/mm, sobre una sola mascara que lleva los dos extremos a la vez
(``pedicel_mm=25``, ``pedicel_width_mm=9``, ``blossom_taper_mm=30``), cada
extremo medido sobre su propio tramo "delgado -> meseta":

- Ruido de fondo del perfil (cuerpo del banano, entre el 20% y el 80% de
  ``s``, lejos de los dos extremos): hasta ~0.0115 1/mm puntual.
- Adelgazamiento floral deliberadamente GRADUAL
  (``blossom_taper_mm=30``): banda puntual ~0.026-0.036 1/mm (maximo
  0.0404), y lo que el criterio realmente compara -- la pendiente PROMEDIO
  del tramo -- **0.0307 1/mm**, por DEBAJO del umbral: no se corta.
- Transicion al pedicelo (``pedicel_mm=25``, ``pedicel_width_mm=9``, sobre
  un fruto de 36 mm de calibre): banda puntual ~0.051-0.059 1/mm (maximo
  0.0604) sostenida sobre decenas de muestras, y pendiente promedio del
  tramo **0.0508 1/mm**, por ENCIMA del umbral: se corta.

El umbral ``_DERIVATIVE_THRESHOLD_PER_MM = 0.04`` sigue cayendo entre los
dos casos, con 0.0093 1/mm de margen del lado gradual (0.0307) y 0.0108 del
lado abrupto (0.0508), y mas de tres veces el ruido de fondo (0.0115).

**Como movio la correccion de la abscisa estos margenes** (misma medicion,
antes y despues del cambio de la ronda 4, para quien vuelva a tocar el
umbral): la pendiente promedio del adelgazamiento gradual subio de 0.0291 a
0.0307, asi que el margen de ESE lado -- el que decide que el extremo floral
NO se recorte -- se angosto de 0.0109 a 0.0093 1/mm; la del pedicelo subio
de 0.0481 a 0.0508, asi que el margen del lado abrupto se ENSANCHO de 0.0081
a 0.0108. El umbral no se retoco: ambos casos siguen bracketeados con todos
los tests pasando, y mover una constante calibrada para comprar 0.0016 1/mm
de un lado a costa del otro no se justifica. Quien lo revise: el lado
gradual es ahora el ajustado de los dos, al reves que antes.

El perfil crudo tiene
ademas un artefacto puntual en la ultima muestra de una punta afilada (el
refinamiento de punta de ``medial.py`` asume una tapa convexa redondeada,
no un cono, y se desvia un poco justo en el ultimo punto): por eso la
derivada se calcula sobre ``w`` suavizado con una media movil corta, no
sobre el perfil crudo -- un pico aislado de 1-2 muestras no debe disparar
un corte.

**Ronda de revision 1 -- escala de la ventana de suavizado.** La ventana
original era un numero fijo de MUESTRAS (``_SMOOTH_WINDOW = 5``), pero la
cantidad de muestras por milimetro depende de la resolucion del esqueleto
(``medial.py``, factor de reduccion 2) y de ``mm_per_px``, no es una
propiedad del banano. ``_SMOOTH_WINDOW_MM = 2.0`` fija la ventana en
MILIMETROS y la convierte a muestras segun el espaciado real de ``s_mm``
en cada llamada -- asi el suavizado cubre la misma distancia fisica sin
importar la resolucion de captura. (La derivada normalizada en si --
``|dw/ds| / w_max`` -- ya era adimensional respecto del calibre del fruto
desde la primera version; lo que no era invariante de escala era esta
ventana. Ver el informe de la ronda de revision 1 para la medicion que
mostro que, aun con la derivada ya normalizada, el criterio solo fallaba
por una inestabilidad distinta, en la reconstruccion de la curva exterior
recortada -- corregida abajo, no aqui.)

El punto de corte no es donde la derivada cruza el umbral por primera vez
(eso cae a mitad de la subida, con el ancho todavia muy por debajo del
calibre del fruto), sino donde el ancho TERMINA de subir y se estabiliza
cerca del calibre maximo (``_PLATEAU_FRACTION``): ese es el punto donde el
perfil vuelve a describir fruto, no la transicion.

Longitud recortada: por que no se vuelve a partir el contorno completo
------------------------------------------------------------------------
La primera version de ``measure_mask`` llamaba a
``outer_curve(contour, (axis.path_px[start], axis.path_px[end]))`` --
re-partir el CONTORNO CERRADO COMPLETO usando los puntos recortados del eje
medial como si fueran puntas nuevas. ``split_contour_at_tips`` busca, para
cada punta, el punto del contorno mas cercano; en una punta real (centro de
una tapa redondeada) esa busqueda es robusta porque el punto esta rodeado
de contorno en todas direcciones de forma mas o menos simetrica. Pero el
punto recortado NO es una punta: cae a mitad del cuerpo del fruto, cerca de
la union con el pedicelo, donde el contorno tiene una geometria mas
complicada (el cuerpo ancho y el tubo delgado del pedicelo conviven a
distancias similares del punto). Medido: para radius=220, arc=0.85,
mm_per_px=0.2, pedicel_mm=25, variando solo el calibre del fruto, la
longitud recortada resultante se disparaba por encima de la del banano SIN
recortar en vez de quedar por debajo (w=12/pedicel_width=3: +13.7%;
w=8/pedicel_width=3: +20.0%; ver informe de la ronda de revision 1) --la
busqueda de "punto mas cercano" elegia, para calibres chicos, el lado
equivocado del contorno y dejaba todo el pedicelo adentro de la cadena
"exterior". ``trim_pedicel`` SI identificaba el indice correcto en todos
los casos (verificado por separado); el defecto estaba en como ese indice
se traducia a una curva exterior.

La correccion: en vez de re-partir el contorno completo, se usa la curva
exterior YA seleccionada de forma robusta con las puntas ORIGINALES del eje
(``outer_curve(contour, axis.tips_px)``, el mismo camino que Task 7 valido)
y se la recorta buscando el punto mas cercano DENTRO de esa cadena ya
ordenada -- una busqueda de vecino mas cercano en una secuencia 1D simple,
sin la ambiguedad topologica de partir un contorno cerrado por un punto
interior. Medido en los mismos cuatro calibres (36, 24, 12, 8 mm, pedicelo
de 25 mm): la diferencia entre la longitud sin recortar y la recortada,
en la MISMA mascara, cae entre 87% y 105% de los 25 mm del pedicelo -- sin
ningun signo invertido ni disparo instable (ver ``tests/test_measures.py``
y el informe de la ronda de revision 1 para la tabla completa).

Calibre, area y volumen -- las cuatro magnitudes son del FRUTO, no de la
mascara completa
------------------------------------------------------------------------
El calibre se toma en el punto medio del eje YA RECORTADO (no en el maximo
ni en el eje sin recortar): el CODEX fija el punto de medicion en la mitad
del fruto, y medir en el maximo circulo inscripto o sobre el eje sin
recortar da un numero distinto (ver informe de esta tarea).

**Ronda de revision 2 -- el area tambien se restringe al fruto.** La
primera version contaba los pixeles de la mascara COMPLETA (incluido el
pedicelo) mientras que ``codex_length_mm`` y ``volume_mm3`` ya se
restringian al tramo recortado -- una inconsistencia que la ronda de
revision 2 señalo: reportar un area que incluye el tallo junto a una
longitud que lo excluye no describe el mismo objeto. Las CUATRO magnitudes
que devuelve ``measure_mask`` -- longitud, calibre, area y volumen --
describen el mismo tramo del fruto, nunca el pedicelo.

**Ronda de revision 3 -- el area se INTEGRA del perfil, no se cuenta del
raster.** La ronda 2 implemento esa restriccion con ``_fruit_pixel_count``:
cortaba el RASTER con un semiplano que pasaba por el punto de recorte y se
evaluaba sobre TODO el lienzo. En un banano curvo ese plano vuelve a cortar
el cuerpo en otra parte y borra fruto real del lado opuesto del cuadro. La
revision lo confirmo tomando las componentes conexas de los pixeles
eliminados: DOS manchas grandes y separadas (~21k y ~19k px) en lados
opuestos del cuadro, no una sola region compacta junto al pedicelo. Medido
contra la verdad analitica del fixture, el area salia 19.4% (w=36), 13.3%
(w=24) y 6.8% (w=12) por debajo -- mientras ``volume_mm3``, que integra
LOCALMENTE sobre ``[start, end]``, se mantenia dentro del ~2%: area y
volumen se contradecian sobre donde termina el fruto. Costaba ademas 20.6
ms por llamada sobre el lienzo de referencia (478x1334) -- contra un
presupuesto de ~11 ms por banano por cuadro -- porque construia un
``mgrid`` y una grilla booleana de lienzo completo en cada llamada.

``_fruit_pixel_count`` se elimino. El area ahora se INTEGRA del perfil de
grosor, ``A = integral(w(s) ds)`` sobre ``[start, end]``, mas un termino de
tapa en cada extremo NO recortado -- el analogo bidimensional exacto de los
casquetes hemisfericos que el volumen ya tenia. Para una capsula el area
cerrada es ``w*R*theta + pi*(w/2)**2`` (un rectangulo barrido mas dos medios
discos), que es precisamente esta descomposicion y es como
``tests/fixtures/synthetic.py`` calcula ``GroundTruth.area_mm2``: por eso el
termino de tapa es MEDIO disco (``0.5*pi*r**2``) por extremo sin recortar, y
las dos tapas juntas suman un disco completo. Esto resuelve los tres
problemas a la vez: no mira nunca el cuadro (ninguna envolvente posible),
integra el MISMO ``w(s)`` sobre el MISMO ``[start, end]`` que el volumen (no
pueden discrepar), y una integral 1D sobre el perfil cuesta microsegundos.
El precio aceptado: el area pasa a ser una cantidad DE MODELO, que hereda el
supuesto de seccion circular que el volumen ya hacia, en vez de un conteo
crudo de pixeles -- cuatro numeros consistentes bajo un supuesto declarado
valen mas que un numero crudo que contradice a los otros tres.

El volumen asume seccion circular, ``V = suma(pi*(w/2)**2 * ds)``, integrada
SOLO sobre el tramo del fruto (``trim_pedicel``). El eje medial retrocede
hasta el CENTRO de cada tapa redondeada (ver ``medial.py``), asi que
``w`` no cae a cero en una punta sin recortar -- la integral sobre el
perfil crudo subestima el volumen real en ~7.7% (medido) porque le falta el
casquete esferico mas alla de ese centro. Por eso se agrega explicitamente
el volumen de un casquete esferico, ``(2/3)*pi*r**3`` con el radio local del
extremo, en cada extremo que NO fue recortado (si el extremo fue recortado,
no hay casquete fisico que agregar: el corte es un plano, no una tapa).

La abscisa de integracion: corregida en Task 6, no aqui
--------------------------------------------------------
Las dos integrales necesitan una abscisa en milimetros REALES. El camino del
esqueleto recorre la curva en zigzag, asi que su longitud acumulada cruda
sobreestima la longitud verdadera -- medido +5.94% (w=36), +5.78% (w=24) y
+5.77% (w=12) contra una longitud central analitica de 187.00 mm. Integrar
sobre esa abscisa arrastra el sesgo al resultado: el area del banano sin
pedicelo salia +3.6% (w=36) y +4.3% (w=24) sobre la verdad analitica, por
encima del 3% exigido, con todo el exceso en el tramo barrido y nada en las
tapas.

**Ronda de revision 4 -- la correccion se movio a ``width_profile_mm``.** La
ronda 3 la aplicaba aqui, en un ayudante local (``_fruit_abscissa_mm``) que
reescalaba la abscisa del tramo del fruto. Era la correccion correcta en el
lugar equivocado: ``width_profile_mm`` promete milimetros y devolvia un 6%
de mas sin que nada en su firma lo advirtiera, asi que cada consumidor del
perfil (Tasks 9 y 14 lo consumen tambien) tendria que redescubrir el mismo
parche o heredar el error en silencio. Task 6 ya es dueña de
``arc_length_px`` y de la simplificacion Douglas-Peucker, asi que la
correccion vive ahi ahora, aplicada una sola vez (ver docstring de
``medial.py``). Este modulo simplemente consume ``s_mm`` tal como llega y
recorta el tramo ``[start, end]``: las dos integrales comparten LA MISMA
abscisa, que es lo que hace imposible que area y volumen discrepen.

Rango de validez de la exactitud del 3%
----------------------------------------
``area_mm2`` y ``volume_mm3`` caen dentro del 3% de la verdad analitica para
los calibres del rango del CODEX (el minimo del estandar,
``CODEX_MIN_CALIPER_MM = 27.0`` en el plan del proyecto; el
Cavendish comercial va de 34 a 42+ mm). Por debajo de ese rango domina el
ruido de rasterizado SUBPIXEL del ancho, que no es un desplazamiento
constante y por lo tanto no es corregible: medido, la transformada de
distancia mide el semiancho medio -0.429 mm (w=36), -0.111 mm (w=24) y
-0.208 mm (w=12), que elevado al cuadrado da -2.37%, -0.93% y -3.44% en el
volumen. El caso w=12 queda apenas afuera del 3%, pero un banano de 12 mm de
calibre no es un banano: esta fuera del rango de operacion del propio
estandar. Los fixtures de 12 mm se conservan en los tests que verifican que
el RECORTE se dispara en cualquier calibre (eso es robustez del criterio, y
ahi corresponde), pero no se les exige una verdad analitica de volumen.

Contrato de excepciones -- ronda de revision 2
------------------------------------------------
``measure_mask`` no lanza una excepcion por una mascara GEOMETRICAMENTE
degenerada: vacia, demasiado chica, que toca el borde, sin dos puntas
significativas. Esos casos estan explicitamente cubiertos por los ``if``
de la funcion (no por un ``except`` generico) y devuelven ``None``, igual
que el contrato de ``medial_axis_path`` (Task 6). Un ``except Exception``
que envolviera toda la funcion, en cambio, convertiria CUALQUIER error de
programacion futuro en esta cadena de llamadas -- un typo, una firma que
cambio, un indice mal calculado -- en el mismo ``None`` silencioso y
permanente que una mascara degenerada, indistinguible de "no hay nada que
medir" para quien llama. Por eso no existe: la degeneracion geometrica se
chequea explicitamente ANTES de cada paso riesgoso, y un error de
programacion real se propaga como la excepcion que es.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np

from codex_caliper.config import AppConfig
from codex_caliper.geometry.medial import MedialAxis, medial_axis_path, width_profile_mm
from codex_caliper.geometry.outline import outer_curve, smoothed_arc_length_mm

# Derivada normalizada (|dw/ds| / w_max, en 1/mm) por encima de la cual un
# tramo de ``w(s)`` se considera una transicion ABRUPTA (pedicelo), no un
# adelgazamiento gradual (extremo floral). Ver docstring del modulo para la
# calibracion medida.
_DERIVATIVE_THRESHOLD_PER_MM = 0.04

# Fraccion de w_max que debe alcanzarse, tras una transicion abrupta, para
# considerar que el perfil "se estabilizo" de vuelta en el cuerpo del fruto.
_PLATEAU_FRACTION = 0.9

# Ventana de la media movil que suaviza w(s) antes de derivar, en
# MILIMETROS (no muestras -- ver docstring del modulo, ronda de revision
# 1): filtra el pico aislado de 1-2 muestras en la ultima punta de un
# adelgazamiento conico, sin borrar la transicion sostenida del pedicelo
# (que dura decenas de milimetros).
_SMOOTH_WINDOW_MM = 2.0

# Factor del volumen de un casquete esferico de radio r: (2/3)*pi*r**3 (la
# mitad de la esfera completa, (4/3)*pi*r**3). Se usa para compensar que el
# eje medial se detiene en el CENTRO de la tapa (ver docstring del modulo),
# no en su borde fisico.
_HEMISPHERE_VOLUME_FACTOR = 2.0 / 3.0

# Analogo bidimensional del casquete: el area de MEDIO disco de radio r es
# 0.5*pi*r**2. Mismo motivo (el eje se detiene en el centro de la tapa) y
# misma convencion que el fixture sintetico, donde las dos tapas juntas
# suman un disco completo, ``pi*(w/2)**2`` (ver docstring del modulo).
_HALF_DISC_AREA_FACTOR = 0.5


@dataclasses.dataclass(frozen=True)
class BananaMeasures:
    """Medidas finales de un banano, listas para reportar.

    Las cuatro magnitudes -- ``codex_length_mm``, ``caliper_mm``,
    ``area_mm2`` y ``volume_mm3`` -- describen SOLO el fruto: el tramo que
    ``trim_pedicel`` deja entre el extremo floral y la base del pedicelo,
    nunca el pedicelo mismo. ``area_mm2`` y ``volume_mm3`` se integran las
    dos del perfil de grosor sobre ese mismo tramo, bajo el supuesto de
    SECCION CIRCULAR (ver docstring del modulo): el area no es un conteo de
    pixeles de la mascara.

    ``width_profile`` es la tupla ``(s_mm, w_mm)`` sobre el eje medial
    COMPLETO (sin recortar), tal como la entrega ``width_profile_mm``: su
    abscisa esta en milimetros reales, ya corregida del sesgo de escalera
    del esqueleto en Task 6. Se llama asi, distinto de ``width_profile_mm``
    (la funcion de Task 6), para no chocar con ella.
    """

    codex_length_mm: float
    caliper_mm: float
    area_mm2: float
    volume_mm3: float
    width_profile: tuple[np.ndarray, np.ndarray]


def _smooth(w_mm: np.ndarray, s_mm: np.ndarray, window_mm: float) -> np.ndarray:
    """Media movil de ``w_mm`` sobre una ventana de ``window_mm`` MILIMETROS
    (no muestras -- ver docstring del modulo, ronda de revision 1), con
    relleno por borde (``edge``) para no acortar el arreglo ni introducir
    sesgo en los extremos.

    La ventana en milimetros se convierte a una cantidad de muestras segun
    el espaciado medio real de ``s_mm`` en esta llamada, para que cubra la
    misma distancia fisica sin importar la resolucion de captura
    (``mm_per_px``) ni el factor de reduccion del esqueleto.
    """
    n = len(w_mm)
    if n < 3:
        return w_mm.copy()
    span_mm = float(s_mm[-1] - s_mm[0])
    if span_mm <= 0.0:
        return w_mm.copy()
    step_mm = span_mm / (n - 1)
    window = max(1, int(round(window_mm / step_mm)))
    window = min(window, n - 1 if n % 2 == 0 else n)
    if window <= 1:
        return w_mm.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(w_mm, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:n]


def trim_pedicel(s_mm: np.ndarray, w_mm: np.ndarray, drop_ratio: float = 0.45) -> tuple[int, int]:
    """Indices de inicio y fin del tramo del FRUTO dentro de ``w_mm``.

    Recorre ``w(s)`` desde cada extremo hacia el centro buscando una
    transicion ABRUPTA (derivada normalizada de ``w`` suavizado por encima
    de ``_DERIVATIVE_THRESHOLD_PER_MM``) que arranca con el ancho por debajo
    de ``drop_ratio * w_max`` y termina cuando el ancho se estabiliza cerca
    del calibre maximo (``_PLATEAU_FRACTION * w_max``). Si no se encuentra
    una transicion con esas caracteristicas en un extremo (p. ej. un
    adelgazamiento GRADUAL, como el floral, o un extremo sin pedicelo), ese
    extremo no se recorta.

    Nunca lanza una excepcion: con menos de 2 muestras, o un perfil sin
    ancho positivo, devuelve el rango completo (sin recortar nada) o vacio,
    segun corresponda.
    """
    s_mm = np.asarray(s_mm, dtype=np.float64)
    w_mm = np.asarray(w_mm, dtype=np.float64)
    n = len(w_mm)
    if n < 2:
        return 0, max(n - 1, 0)

    w_max = float(np.max(w_mm))
    if w_max <= 0.0:
        return 0, n - 1

    threshold_w = drop_ratio * w_max
    plateau_w = _PLATEAU_FRACTION * w_max

    # La pendiente promedio del tramo se calcula sobre w SUAVIZADO (ver
    # docstring del modulo): un pico aislado de 1-2 muestras en la punta de
    # un adelgazamiento conico no debe disparar un corte, pero el cruce de
    # umbral de ``drop_ratio * w_max`` sigue decidiendose con el ancho
    # CRUDO (no tiene sentido desplazar donde el ancho realmente cruza).
    smoothed = _smooth(w_mm, s_mm, _SMOOTH_WINDOW_MM)

    def _avg_slope(lo: int, hi: int) -> float:
        return (smoothed[hi] - smoothed[lo]) / (s_mm[hi] - s_mm[lo])

    def scan_forward_smoothed() -> int:
        lo = None
        for idx in range(n):
            if w_mm[idx] < threshold_w:
                lo = idx
            else:
                break
        if lo is None:
            return 0
        hi = None
        for idx in range(lo, n):
            if w_mm[idx] >= plateau_w:
                hi = idx
                break
        if hi is None or hi == lo:
            return 0
        if abs(_avg_slope(lo, hi)) / w_max >= _DERIVATIVE_THRESHOLD_PER_MM:
            return hi
        return 0

    def scan_backward_smoothed() -> int:
        lo = None
        for idx in range(n - 1, -1, -1):
            if w_mm[idx] < threshold_w:
                lo = idx
            else:
                break
        if lo is None:
            return n - 1
        hi = None
        for idx in range(lo, -1, -1):
            if w_mm[idx] >= plateau_w:
                hi = idx
                break
        if hi is None or hi == lo:
            return n - 1
        if abs(_avg_slope(lo, hi)) / w_max >= _DERIVATIVE_THRESHOLD_PER_MM:
            return hi
        return n - 1

    start = scan_forward_smoothed()
    end = scan_backward_smoothed()
    if start >= end:
        return 0, n - 1
    return start, end


def touches_border(mask: np.ndarray, margin_px: int = 2) -> bool:
    """Verdadero si algun pixel de ``mask`` cae dentro de ``margin_px`` del
    borde del lienzo -- es decir, si la mascara probablemente esta cortada
    por el encuadre, no completa.

    Una mascara vacia no "toca" el borde (no hay nada que midan mal): el
    llamador la rechazara de todos modos por otra via (eje medial nulo).
    """
    mask_bool = np.asarray(mask).astype(bool)
    if not mask_bool.any():
        return False

    height, width = mask_bool.shape
    margin = max(0, margin_px)
    if margin == 0:
        return bool(
            mask_bool[0, :].any() or mask_bool[-1, :].any()
            or mask_bool[:, 0].any() or mask_bool[:, -1].any()
        )

    top = mask_bool[:margin, :]
    bottom = mask_bool[height - margin :, :]
    left = mask_bool[:, :margin]
    right = mask_bool[:, width - margin :]
    return bool(top.any() or bottom.any() or left.any() or right.any())


def _largest_contour_px(mask: np.ndarray) -> np.ndarray | None:
    """Contorno exterior mas grande de ``mask``, como arreglo ``(N, 2)``."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(np.float64)


def _restrict_outer_to_fruit(
    full_outer: np.ndarray, axis: MedialAxis, start: int, end: int, n: int
) -> np.ndarray:
    """Recorta ``full_outer`` (la curva exterior YA seleccionada de forma
    robusta con las puntas ORIGINALES del eje medial, via
    ``outer_curve(contour, axis.tips_px)``) a los indices del fruto.

    No vuelve a partir el contorno cerrado completo por los puntos
    recortados (eso fue el defecto de la ronda de revision 1: el punto
    recortado cae a mitad del cuerpo, no en una punta, y la busqueda de
    "punto mas cercano" sobre el contorno completo es ambigua/inestable
    alli -- ver docstring del modulo). En cambio, busca el punto mas
    cercano DENTRO de la cadena ``full_outer`` ya ordenada (una busqueda de
    vecino mas cercano en una secuencia 1D, sin la ambiguedad topologica de
    un contorno cerrado) y conserva el tramo entre esos puntos.
    """
    if start == 0 and end == n - 1:
        return full_outer

    def nearest_chain_index(point: np.ndarray) -> int:
        diffs = full_outer - point
        return int(np.argmin(np.sum(diffs * diffs, axis=1)))

    # orientacion de la cadena respecto al orden del eje: el indice de la
    # cadena, crece en el mismo sentido que el indice del eje, o al reves?
    dist_chain_start_to_tip0 = float(np.linalg.norm(full_outer[0] - axis.tips_px[0]))
    dist_chain_end_to_tip0 = float(np.linalg.norm(full_outer[-1] - axis.tips_px[0]))
    chain_increases_with_axis = dist_chain_start_to_tip0 <= dist_chain_end_to_tip0

    lo_bound = 0
    hi_bound = len(full_outer) - 1

    if start > 0:
        i = nearest_chain_index(axis.path_px[start])
        if chain_increases_with_axis:
            lo_bound = max(lo_bound, i)
        else:
            hi_bound = min(hi_bound, i)
    if end < n - 1:
        i = nearest_chain_index(axis.path_px[end])
        if chain_increases_with_axis:
            hi_bound = min(hi_bound, i)
        else:
            lo_bound = max(lo_bound, i)

    if lo_bound >= hi_bound:
        return full_outer
    return full_outer[lo_bound : hi_bound + 1]


def _area_mm2(abscissa_mm: np.ndarray, w_mm: np.ndarray, start: int, end: int) -> float:
    """Area del FRUTO, integrada del perfil de grosor: ``integral(w ds)``
    sobre ``[start, end]`` mas MEDIO disco en cada extremo que NO fue
    recortado.

    Es el analogo bidimensional exacto de los casquetes hemisfericos del
    volumen, y la misma descomposicion que la forma cerrada de una capsula
    (``w*R*theta + pi*(w/2)**2`` = rectangulo barrido + dos medios discos),
    que es como el fixture sintetico calcula su verdad analitica. El area
    NO se cuenta del raster: contar pixeles obliga a recortar la mascara con
    un plano que, en un banano curvo, vuelve a cortar el cuerpo del otro
    lado del cuadro y borra fruto real (ver docstring del modulo, ronda de
    revision 3).

    Si un extremo fue recortado no se le agrega tapa: alli el fruto termina
    en un plano de corte, no en una tapa redondeada.
    """
    segment_w = np.asarray(w_mm[start : end + 1], dtype=np.float64)
    if len(abscissa_mm) < 2 or len(segment_w) < 2:
        return 0.0
    area = float(np.trapezoid(segment_w, abscissa_mm))

    n = len(w_mm)
    if start == 0:
        r0 = float(w_mm[0]) / 2.0
        area += _HALF_DISC_AREA_FACTOR * math.pi * r0 ** 2
    if end == n - 1:
        r1 = float(w_mm[-1]) / 2.0
        area += _HALF_DISC_AREA_FACTOR * math.pi * r1 ** 2
    return area


def _volume_mm3(abscissa_mm: np.ndarray, w_mm: np.ndarray, start: int, end: int) -> float:
    """Volumen de seccion circular sobre el tramo ``[start, end]``, mas un
    casquete esferico en cada extremo que NO fue recortado (ver docstring
    del modulo: el eje medial se detiene en el centro de la tapa, asi que
    la integral sola subestima el volumen real de un extremo redondeado sin
    recortar).

    Integra sobre la MISMA abscisa (el tramo ``[start, end]`` de la que
    devuelve ``width_profile_mm``, ya corregida en Task 6) y el MISMO tramo
    que ``_area_mm2``: area y volumen no pueden discrepar sobre donde
    termina el fruto.
    """
    segment_w = np.asarray(w_mm[start : end + 1], dtype=np.float64)
    if len(abscissa_mm) < 2 or len(segment_w) < 2:
        return 0.0
    integrand = math.pi * (segment_w / 2.0) ** 2
    volume = float(np.trapezoid(integrand, abscissa_mm))

    n = len(w_mm)
    if start == 0:
        r0 = float(w_mm[0]) / 2.0
        volume += _HEMISPHERE_VOLUME_FACTOR * math.pi * r0 ** 3
    if end == n - 1:
        r1 = float(w_mm[-1]) / 2.0
        volume += _HEMISPHERE_VOLUME_FACTOR * math.pi * r1 ** 3
    return volume


def measure_mask(mask: np.ndarray, mm_per_px: float, cfg: AppConfig) -> BananaMeasures | None:
    """Mide un banano a partir de su mascara binaria.

    Encadena: ``touches_border`` -> ``medial_axis_path`` -> ``width_profile_mm``
    -> ``trim_pedicel`` -> ``outer_curve`` (con las puntas ORIGINALES, robusta)
    recortada a los indices del fruto via ``_restrict_outer_to_fruit`` ->
    ``smoothed_arc_length_mm``. Calibre = 2 * radio en el punto medio del
    eje YA recortado.

    Las CUATRO magnitudes devueltas describen el mismo tramo del FRUTO,
    nunca el pedicelo. Area y volumen se calculan los dos integrando el
    perfil de grosor sobre el MISMO tramo ``[start, end]`` y la MISMA
    abscisa (la de ``width_profile_mm``, corregida del sesgo de escalera en
    Task 6 -- ver docstring del modulo), bajo el supuesto de SECCION
    CIRCULAR: ``A = integral(w ds)`` mas medio disco por extremo sin
    recortar (``_area_mm2``) y ``V = integral(pi*(w/2)**2 ds)`` mas un
    casquete hemisferico por extremo sin recortar (``_volume_mm3``). El area
    es por lo tanto una cantidad DE MODELO, no un conteo crudo de pixeles
    (ver docstring del modulo, ronda de revision 3, para por que contar
    pixeles recortados por un plano borraba fruto real en un banano curvo).

    Devuelve ``None`` por una mascara GEOMETRICAMENTE degenerada -- cada
    caso esta chequeado explicitamente abajo, ANTES del paso que lo usa, no
    capturado con un ``except`` generico (ver docstring del modulo, ronda
    de revision 2: capturar ciegamente convertiria un error de programacion
    futuro en esta cadena en el mismo ``None`` silencioso que una mascara
    degenerada). No lanza una excepcion por degeneracion geometrica; SI la
    lanza si una llamada interna falla por cualquier otro motivo.
    """
    if touches_border(mask, cfg.border_margin_px):
        return None

    axis: MedialAxis | None = medial_axis_path(mask)
    if axis is None:
        return None

    s_mm, w_mm = width_profile_mm(axis, mm_per_px)
    if len(s_mm) < 2:
        return None

    start, end = trim_pedicel(s_mm, w_mm, cfg.pedicel_drop_ratio)
    if end <= start:
        return None

    contour = _largest_contour_px(mask)
    if contour is None or len(contour) < 2:
        return None

    full_outer = outer_curve(contour, axis.tips_px)
    if full_outer is None or len(full_outer) < 2:
        return None
    outer = _restrict_outer_to_fruit(full_outer, axis, start, end, len(w_mm))
    codex_length_mm = smoothed_arc_length_mm(outer, mm_per_px)

    mid = (start + end) // 2
    caliper_mm = 2.0 * float(axis.radius_px[mid]) * mm_per_px

    # UNA sola abscisa para las dos integrales: el tramo del fruto de la
    # abscisa que devuelve ``width_profile_mm``, ya corregida del sesgo de
    # escalera en Task 6 (ver docstring del modulo). Area y volumen no
    # pueden discrepar sobre donde termina el fruto porque no tienen dos
    # abscisas entre las que discrepar.
    abscissa_mm = np.asarray(s_mm[start : end + 1], dtype=np.float64)
    area_mm2 = _area_mm2(abscissa_mm, w_mm, start, end)
    volume_mm3 = _volume_mm3(abscissa_mm, w_mm, start, end)

    return BananaMeasures(
        codex_length_mm=codex_length_mm,
        caliper_mm=caliper_mm,
        area_mm2=area_mm2,
        volume_mm3=volume_mm3,
        width_profile=(s_mm, w_mm),
    )

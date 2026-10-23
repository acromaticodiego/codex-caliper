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
width=36, mm_per_px=0.2, ver ``tests/test_measures.py``):

- El ruido de fondo del perfil (cuerpo del banano, lejos de cualquier
  extremo) tiene una derivada normalizada (``|dw/ds| / w_max``) de hasta
  ~0.016 1/mm.
- Un adelgazamiento floral deliberadamente GRADUAL (``blossom_taper_mm=30``)
  sostiene una derivada normalizada de ~0.027-0.031 1/mm durante toda la
  caida.
- La transicion al pedicelo (``pedicel_mm=25``, ``pedicel_width_mm=9``,
  sobre un fruto de 36 mm de calibre) sostiene ~0.04-0.055 1/mm durante
  decenas de muestras consecutivas.

El umbral ``_DERIVATIVE_THRESHOLD_PER_MM = 0.04`` cae comodo entre el
adelgazamiento gradual (0.03) y la transicion abrupta (0.04-0.055), con
margen de sobra sobre el ruido de fondo (0.016). El perfil crudo tiene
ademas un artefacto puntual en la ultima muestra de una punta afilada (el
refinamiento de punta de ``medial.py`` asume una tapa convexa redondeada,
no un cono, y se desvia un poco justo en el ultimo punto): por eso la
derivada se calcula sobre ``w`` suavizado con una media movil corta
(``_SMOOTH_WINDOW``), no sobre el perfil crudo -- un pico aislado de 1-2
muestras no debe disparar un corte.

El punto de corte no es donde la derivada cruza el umbral por primera vez
(eso cae a mitad de la subida, con el ancho todavia muy por debajo del
calibre del fruto), sino donde el ancho TERMINA de subir y se estabiliza
cerca del calibre maximo (``_PLATEAU_FRACTION``): ese es el punto donde el
perfil vuelve a describir fruto, no la transicion.

Calibre, area y volumen
------------------------
El calibre se toma en el punto medio del eje YA RECORTADO (no en el maximo
ni en el eje sin recortar): el CODEX fija el punto de medicion en la mitad
del fruto, y medir en el maximo circulo inscripto o sobre el eje sin
recortar da un numero distinto (ver informe de esta tarea).

El area es el conteo de pixeles de la mascara completa por ``mm_per_px**2``
-- no se restringe al tramo del fruto.

El volumen asume seccion circular, ``V = suma(pi*(w/2)**2 * ds)``, integrada
SOLO sobre el tramo del fruto (``trim_pedicel``). El eje medial retrocede
hasta el CENTRO de cada tapa redondeada (ver ``medial.py``), asi que
``w`` no cae a cero en una punta sin recortar -- la integral sobre el
perfil crudo subestima el volumen real en ~7.7% (medido) porque le falta el
casquete esferico mas alla de ese centro. Por eso se agrega explicitamente
el volumen de un casquete esferico, ``(2/3)*pi*r**3`` con el radio local del
extremo, en cada extremo que NO fue recortado (si el extremo fue recortado,
no hay casquete fisico que agregar: el corte es un plano, no una tapa).
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

# Ventana de la media movil que suaviza w(s) antes de derivar: filtra el
# pico aislado de 1-2 muestras en la ultima punta de un adelgazamiento
# conico (ver docstring del modulo), sin borrar la transicion sostenida del
# pedicelo (que dura decenas de muestras).
_SMOOTH_WINDOW = 5

# Factor del volumen de un casquete esferico de radio r: (2/3)*pi*r**3 (la
# mitad de la esfera completa, (4/3)*pi*r**3). Se usa para compensar que el
# eje medial se detiene en el CENTRO de la tapa (ver docstring del modulo),
# no en su borde fisico.
_HEMISPHERE_VOLUME_FACTOR = 2.0 / 3.0


@dataclasses.dataclass(frozen=True)
class BananaMeasures:
    """Medidas finales de un banano, listas para reportar.

    ``width_profile`` es la tupla ``(s_mm, w_mm)`` sobre el eje medial
    COMPLETO (sin recortar) -- se llama asi, distinto de
    ``width_profile_mm`` (la funcion de Task 6), para no chocar con ella.
    """

    codex_length_mm: float
    caliper_mm: float
    area_mm2: float
    volume_mm3: float
    width_profile: tuple[np.ndarray, np.ndarray]


def _smooth(w_mm: np.ndarray, window: int) -> np.ndarray:
    """Media movil corta de ``w_mm``, con relleno por borde (``edge``) para
    no acortar el arreglo ni introducir sesgo en los extremos."""
    n = len(w_mm)
    if window <= 1 or n < window:
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
    smoothed = _smooth(w_mm, _SMOOTH_WINDOW)

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


def _volume_mm3(s_mm: np.ndarray, w_mm: np.ndarray, start: int, end: int) -> float:
    """Volumen de seccion circular sobre el tramo ``[start, end]``, mas un
    casquete esferico en cada extremo que NO fue recortado (ver docstring
    del modulo: el eje medial se detiene en el centro de la tapa, asi que
    la integral sola subestima el volumen real de un extremo redondeado sin
    recortar)."""
    segment_s = s_mm[start : end + 1]
    segment_w = w_mm[start : end + 1]
    if len(segment_s) < 2:
        return 0.0
    integrand = math.pi * (segment_w / 2.0) ** 2
    volume = float(np.trapezoid(integrand, segment_s))

    n = len(w_mm)
    if start == 0:
        r0 = w_mm[0] / 2.0
        volume += _HEMISPHERE_VOLUME_FACTOR * math.pi * r0 ** 3
    if end == n - 1:
        r1 = w_mm[-1] / 2.0
        volume += _HEMISPHERE_VOLUME_FACTOR * math.pi * r1 ** 3
    return volume


def measure_mask(mask: np.ndarray, mm_per_px: float, cfg: AppConfig) -> BananaMeasures | None:
    """Mide un banano a partir de su mascara binaria.

    Encadena: ``touches_border`` -> ``medial_axis_path`` -> ``width_profile_mm``
    -> ``trim_pedicel`` -> ``outer_curve`` recortada a los indices del fruto
    -> ``smoothed_arc_length_mm``. Calibre = 2 * radio en el punto medio del
    eje YA recortado. Area = pixeles * mm_per_px**2. Volumen = suma de
    discos circulares sobre el tramo del fruto (mas los casquetes de los
    extremos sin recortar, ver ``_volume_mm3``).

    Nunca lanza una excepcion: cualquier fallo en la cadena (mascara
    degenerada, eje medial nulo, mascara que toca el borde) devuelve
    ``None``, igual que el contrato de ``medial_axis_path`` (Task 6).
    """
    try:
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

        trimmed_tips = (axis.path_px[start], axis.path_px[end])
        outer = outer_curve(contour, trimmed_tips)
        codex_length_mm = smoothed_arc_length_mm(outer, mm_per_px)

        mid = (start + end) // 2
        caliper_mm = 2.0 * float(axis.radius_px[mid]) * mm_per_px

        area_mm2 = float(np.count_nonzero(mask)) * (mm_per_px ** 2)

        volume_mm3 = _volume_mm3(s_mm, w_mm, start, end)

        return BananaMeasures(
            codex_length_mm=codex_length_mm,
            caliper_mm=caliper_mm,
            area_mm2=area_mm2,
            volume_mm3=volume_mm3,
            width_profile=(s_mm, w_mm),
        )
    except Exception:
        return None

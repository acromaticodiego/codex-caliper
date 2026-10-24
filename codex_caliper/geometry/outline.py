"""Curva exterior del contorno y longitud Codex (Task 7).

El CODEX STAN 205-1997 define la longitud del banano medida "a lo largo de
la curva exterior", desde el extremo floral hasta la base del pedicelo --
no la cuerda punta a punta (lo que daria un bounding box) ni la linea media
del eje medial (Task 6): en un objeto curvo el borde convexo es mas largo
que la linea media, que a su vez es mas larga que el borde concavo. Este
modulo parte el contorno 2D de la mascara en sus dos mitades (exterior e
interior) usando las puntas del eje medial, y mide la mas larga de las dos
con la misma correccion de escalonado que ``codex_caliper.geometry.medial``
ya implemento para el eje medial 1D.

``smoothed_arc_length_mm`` NO reimplementa el suavizado: llama a
``simplify_path_px`` (Douglas-Peucker via ``cv2.approxPolyDP``), la misma
primitiva que Task 6 construyo y midio para el eje medial. Es una decision
del controlador tomada al construir Task 6: la correccion de escalonado
vive en un solo lugar y Task 7 la consume, en vez de ajustar un spline
suavizante por separado. El parametro ``smoothing`` de
``smoothed_arc_length_mm`` es ese epsilon de Douglas-Peucker expresado en
milimetros (convertido a pixeles via ``mm_per_px`` antes de llamar a
``simplify_path_px``); Task 6 midio que el resultado es insensible a este
epsilon entre 0.2 y 2.0 mm, y el valor por defecto (1.0 mm) cae comodo en
el medio de ese rango.
"""

from __future__ import annotations

import numpy as np

from codex_caliper.geometry.medial import simplify_path_px

# Longitud minima (en puntos) de un contorno para que "partirlo en dos"
# tenga sentido. Por debajo de esto no hay informacion suficiente.
_MIN_CONTOUR_POINTS = 2


def _as_points(path: np.ndarray) -> np.ndarray:
    """Normaliza ``path`` a un arreglo ``(N, 2)`` de tipo ``float64``.

    Acepta tanto el formato ``(N, 2)`` que usan las funciones propias de
    este proyecto como el formato ``(N, 1, 2)`` que devuelve
    ``cv2.findContours`` directamente, para que el llamador no tenga que
    aplanar el contorno antes de pasarlo.
    """
    array = np.asarray(path)
    if array.ndim == 3:
        array = array.reshape(-1, 2)
    return array.astype(np.float64)


def _nearest_index(points: np.ndarray, target: np.ndarray) -> int:
    """Indice del punto de ``points`` mas cercano (distancia euclidiana) a
    ``target``. No asume que ``target`` caiga exactamente sobre ``points``:
    funciona igual de bien si ``target`` esta en el borde o a unos pocos
    milimetros adentro de el (ver docstring de ``split_contour_at_tips``)."""
    diffs = points - np.asarray(target, dtype=np.float64)
    dist_sq = np.sum(diffs * diffs, axis=1)
    return int(np.argmin(dist_sq))


def split_contour_at_tips(
    contour_px: np.ndarray, tips_px: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Parte el contorno cerrado ``contour_px`` en dos cadenas abiertas,
    cortando en los dos puntos del contorno mas cercanos a ``tips_px``.

    ``tips_px`` son las puntas del eje medial (Task 6), que por construccion
    del eje medial NO tienen por que caer exactamente sobre el contorno: en
    una tapa redondeada caen en el centro de esa tapa (a un radio local de
    distancia del borde), y en un corte plano real (ver nota de revision de
    Task 6) pueden quedar varios milimetros adentro del cuerpo porque el eje
    medial retrocede frente a un borde recto. Esta funcion no asume lo
    contrario en ningun momento: busca, para cada punta, el punto del
    contorno euclidianamente mas cercano, sea cual sea esa distancia, y
    nunca lanza una excepcion por una punta imprecisa -- en el peor caso
    (corte muy plano, punta muy adentro) el punto elegido simplemente se
    desliza un poco a lo largo del borde mas cercano a esa zona, sin
    romperse. Lo que SI asume, y no se puede evitar con solo dos puntos y un
    contorno, es que la tapa en la punta no sea tan ancha que su material
    redondeado domine la comparacion entre las dos cadenas resultantes (ver
    el modulo de tests para la magnitud real de ese efecto).

    Las dos cadenas resultantes cubren el contorno completo sin solaparse
    (salvo en los dos puntos de corte, que quedan compartidos como extremos
    de ambas). Caso degenerado: si el contorno tiene menos de 2 puntos, o si
    ambas puntas resuelven al mismo indice (empate exacto, posible con
    mascaras casi puntuales), se devuelve el contorno completo como primera
    cadena y un arreglo vacio como segunda -- nunca una excepcion -- para
    que el llamador pueda detectar el caso revisando ``len(...)``.
    """
    contour = _as_points(contour_px)
    empty = np.empty((0, 2), dtype=np.float64)
    if len(contour) < _MIN_CONTOUR_POINTS:
        return contour.copy(), empty.copy()

    idx_a = _nearest_index(contour, tips_px[0])
    idx_b = _nearest_index(contour, tips_px[1])
    if idx_a == idx_b:
        return contour.copy(), empty.copy()

    lo, hi = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
    chain_lo_hi = contour[lo : hi + 1]
    chain_wrap = np.concatenate([contour[hi:], contour[: lo + 1]], axis=0)
    return chain_lo_hi, chain_wrap


def outer_curve(contour_px: np.ndarray, tips_px: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """La cadena mas larga de las dos que resultan de partir ``contour_px``
    en ``tips_px`` -- el borde convexo (exterior) del banano, que en un
    objeto curvo es mas largo que el borde concavo (ver docstring del
    modulo). "Mas larga" se mide en cantidad de puntos del contorno, que
    para una densidad de muestreo aproximadamente uniforme como la de
    ``cv2.findContours`` es un proxy fiel del orden de la longitud real
    (confirmado en los tests de este modulo contra la longitud suavizada
    real, no solo contra el conteo)."""
    chain_a, chain_b = split_contour_at_tips(contour_px, tips_px)
    return chain_a if len(chain_a) >= len(chain_b) else chain_b


def smoothed_arc_length_mm(chain_px: np.ndarray, mm_per_px: float, smoothing: float = 1.0) -> float:
    """Longitud en milimetros de ``chain_px`` tras corregir el sesgo de
    escalonado con la misma simplificacion Douglas-Peucker de Task 6
    (``simplify_path_px``), nunca sumando los segmentos crudos entre
    pixeles vecinos (ver docstring del modulo).

    ``smoothing`` es el epsilon de esa simplificacion en milimetros
    (convertido a pixeles dividiendo por ``mm_per_px``); el valor por
    defecto, 1.0 mm, cae dentro del rango 0.2-2.0 mm que Task 6 midio como
    insensible. Acepta tanto cadenas abiertas (el resultado de
    ``outer_curve``) como contornos cerrados de ``cv2.findContours``
    (``closed=True`` originalmente): en ambos casos se trata como una
    secuencia abierta de punto a punto, lo que para un contorno cerrado
    omite el ultimo segmento de cierre -- una fraccion insignificante de la
    longitud total frente a la tolerancia que usan los tests de este
    modulo.
    """
    chain = _as_points(chain_px)
    if len(chain) < _MIN_CONTOUR_POINTS:
        return 0.0

    epsilon_px = smoothing / mm_per_px if mm_per_px > 0.0 else 0.0
    simplified = simplify_path_px(chain, epsilon_px)
    if len(simplified) < _MIN_CONTOUR_POINTS:
        return 0.0

    diffs = np.diff(simplified, axis=0)
    length_px = float(np.sum(np.sqrt(np.sum(diffs * diffs, axis=1))))
    return length_px * mm_per_px

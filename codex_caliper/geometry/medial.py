"""Eje medial podado (camino geodesico mas largo) y perfil de grosor.

El eje medial crudo de una mascara binaria tiene ramas espurias cerca de
cualquier irregularidad del contorno (ver docstring del modulo de tests). La
poda robusta es estructural, no un conjunto de reglas ad hoc: se construye un
grafo sobre los pixeles del esqueleto (vecindad 8, peso 1 ortogonal y
``sqrt(2)`` diagonal) y se toma el **camino geodesico mas largo** --
Dijkstra desde un extremo arbitrario para hallar el nodo mas lejano, y otra
vez desde ese nodo para hallar el extremo opuesto. Los dos extremos de ese
camino son, por construccion, las puntas del banano, y cualquier rama que no
quede sobre el camino se descarta automaticamente.

Decisiones de rendimiento (medidas sobre un banano sintetico de referencia,
radius_mm=220, arc_rad=0.85, width_mm=36, mm_per_px=0.2 -- lienzo
478x1288 px, tipico de lo que entrega ``rectify`` en produccion):

- ``skimage.morphology.medial_axis(mask, return_distance=True)`` (la opcion
  obvia, que calcula esqueleto y transformada de distancia en una sola
  llamada) tarda ~98-115 ms sobre esa mascara. A 30 fps con varios bananos
  por cuadro, eso agota el presupuesto de cuadro completo.
- ``skimage.morphology.skeletonize`` (solo esqueleto, sin distancia) mas
  ``cv2.distanceTransform(..., cv2.DIST_L2, cv2.DIST_MASK_PRECISE)`` por
  separado da el mismo resultado (radios dentro de 0.1 px) en ~53 ms a
  resolucion completa, y calculando el esqueleto a MITAD de resolucion cae
  a ~9 ms -- dentro del presupuesto de ~11 ms/banano a 30 fps con 3 bananos
  por cuadro -- perdiendo menos de 0.1 punto porcentual de precision de
  longitud frente a resolucion completa (ver ``test_medial.py``, que exige
  2%). Un cuarto de resolucion es demasiado agresivo: el error de longitud
  sube a ~2.9%, por encima de esa tolerancia. Por eso el factor de
  reduccion es fijo en 2, no mas.
- La transformada de distancia, en cambio, SI se calcula a resolucion
  completa (cuesta unos pocos ms extra sobre la mascara completa): es
  mucho mas barata que el esqueleto y da el radio local con precision
  subpixel real, que es justo lo que ``width_profile_mm`` expone.

El otro problema practico -- y no uno de rendimiento -- es que la secuencia
de pixeles del esqueleto en zigzag ("escalera") sobreestima la longitud real
de una curva suave aun pesando las diagonales con ``sqrt(2)``: medido sobre
el mismo banano de referencia, la suma cruda de segmentos da +6.8% sobre la
longitud analitica, muy por encima del 2% exigido. ``simplify_path_px``
corrige esto simplificando el camino con ``cv2.approxPolyDP`` (Douglas-
Peucker) antes de medir: reemplaza los microescalones por cuerdas rectas,
sin alisar ni perder las puntas. ``arc_length_px`` aplica esa simplificacion
internamente -- nunca suma los segmentos crudos -- y Task 7 puede llamar a
``simplify_path_px`` directamente para construir su propia longitud en mm
sobre el mismo camino ya simplificado, en vez de reimplementar la
correccion.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

# Factor de reduccion de resolucion para el calculo del esqueleto (no de la
# transformada de distancia, que se calcula a resolucion completa). 2 cae
# dentro del presupuesto de cuadro con una perdida de precision
# insignificante; 4 ya rompe la tolerancia de longitud del 2% (ver
# docstring del modulo).
_SKELETON_DOWNSAMPLE = 2

# Vecindad 8 del grafo del esqueleto: (delta_fila, delta_columna, peso).
_NEIGHBOR_OFFSETS: tuple[tuple[int, int, float], ...] = (
    (-1, -1, float(np.sqrt(2))), (-1, 0, 1.0), (-1, 1, float(np.sqrt(2))),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, float(np.sqrt(2))), (1, 0, 1.0), (1, 1, float(np.sqrt(2))),
)

# Una mascara con menos de 2 pixeles de esqueleto no tiene camino alguno.
_MIN_SKELETON_PIXELS = 2
# Un camino mas corto que esto no distingue dos puntas reales (ruido de
# discretizacion, tipicamente un blob casi puntual).
_MIN_PATH_LENGTH_PX = 4.0
# elongacion = longitud_del_camino / (2 * radio_maximo). Un disco perfecto
# mide ~1.0-1.1 (el camino mas largo del esqueleto apenas cruza el centro,
# de borde a borde del circulo inscripto mas grande); un banano tipico mide
# 5 o mas. El umbral queda bien adentro de ese hueco, con margen de sobra
# para ambos lados.
_MIN_ELONGATION = 2.0
# Epsilon por defecto (en pixeles) de la simplificacion Douglas-Peucker que
# corrige el sesgo de "escalera" del camino crudo del esqueleto. Medido
# insensible en un rango amplio (ver docstring del modulo); este valor cae
# comodo en el medio de ese rango.
_DEFAULT_SIMPLIFY_EPSILON_PX = 2.0

# Paso del barrido 1D que busca el borde de la mascara al refinar una punta
# (ver ``_refine_tip_px``) y limite de pasos como multiplo del radio local,
# para no recorrer indefinidamente sobre una mascara patologica.
_TIP_REFINE_STEP_PX = 0.5
_TIP_REFINE_MAX_RADIUS_MULT = 4.0
_TIP_REFINE_MIN_STEPS = 64


@dataclasses.dataclass(frozen=True)
class MedialAxis:
    """Eje medial podado de un banano, de punta a punta.

    ``path_px`` son coordenadas ``(x, y)`` en pixeles -- la misma convencion
    que el resto del proyecto (ver ``codex_caliper.homography``) -- ordenadas
    de una punta a la otra. ``radius_px[i]`` es el radio del circulo
    inscripto en ``path_px[i]`` (la mitad del ancho local), tal como lo
    entrega la transformada de distancia. ``tips_px`` son los dos extremos
    del camino, es decir ``path_px[0]`` y ``path_px[-1]``.
    """

    path_px: np.ndarray
    radius_px: np.ndarray
    tips_px: tuple[np.ndarray, np.ndarray]


def _downsample_mask(mask_bool: np.ndarray, factor: int) -> tuple[np.ndarray, float, float]:
    """Reduce ``mask_bool`` por ``factor`` usando promedio de area (igual
    criterio de "mitad del area cubierta" que usa el fixture sintetico,
    ver Task 2) y devuelve la mascara reducida junto con los factores de
    escala ``(sx, sy)`` para volver a pixeles de resolucion completa."""
    height, width = mask_bool.shape
    small_width = max(1, round(width / factor))
    small_height = max(1, round(height / factor))
    mask_u8 = mask_bool.astype(np.uint8) * 255
    small = cv2.resize(mask_u8, (small_width, small_height), interpolation=cv2.INTER_AREA)
    small_bool = small >= 128
    scale_x = width / small_width
    scale_y = height / small_height
    return small_bool, scale_x, scale_y


def _longest_geodesic_path(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Camino geodesico mas largo entre dos pixeles del esqueleto.

    Construye el grafo de vecindad 8 de forma vectorizada (sin recorrer
    pixel por pixel en Python), corre Dijkstra desde un nodo arbitrario para
    hallar el extremo mas lejano y otra vez desde ese extremo para hallar el
    opuesto, y devuelve las filas/columnas (en la grilla de ``skeleton``) de
    los nodos del camino entre ambos, en orden de punta a punta. ``None`` si
    el esqueleto tiene menos de dos pixeles.
    """
    ys, xs = np.nonzero(skeleton)
    n = len(xs)
    if n < _MIN_SKELETON_PIXELS:
        return None

    height, width = skeleton.shape
    node_id = np.full((height, width), -1, dtype=np.int64)
    node_id[ys, xs] = np.arange(n)

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for dy, dx, weight in _NEIGHBOR_OFFSETS:
        ny, nx = ys + dy, xs + dx
        in_bounds = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
        neighbor_ids = np.full(n, -1, dtype=np.int64)
        neighbor_ids[in_bounds] = node_id[ny[in_bounds], nx[in_bounds]]
        valid = neighbor_ids >= 0
        src_parts.append(np.nonzero(valid)[0])
        dst_parts.append(neighbor_ids[valid])
        weight_parts.append(np.full(int(valid.sum()), weight))

    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    weight = np.concatenate(weight_parts)
    graph = csr_matrix((weight, (src, dst)), shape=(n, n))

    dist_from_0 = dijkstra(graph, directed=True, indices=0)
    far_end_a = int(np.argmax(np.where(np.isfinite(dist_from_0), dist_from_0, -1.0)))

    dist_from_a, predecessors = dijkstra(graph, directed=True, indices=far_end_a, return_predecessors=True)
    far_end_b = int(np.argmax(np.where(np.isfinite(dist_from_a), dist_from_a, -1.0)))

    path_node_ids = [far_end_b]
    current = far_end_b
    while current != far_end_a:
        current = predecessors[current]
        path_node_ids.append(current)
    path_node_ids.reverse()
    path_node_ids = np.array(path_node_ids, dtype=np.int64)

    return ys[path_node_ids], xs[path_node_ids]


def _refine_tip_px(mask_bool: np.ndarray, distance_full: np.ndarray, approx_xy: np.ndarray, inward_xy: np.ndarray) -> np.ndarray:
    """Refina la posicion de una punta, perdida por la cuadricula de media
    resolucion en la que se calculo el esqueleto (ver docstring del modulo).

    Para una capsula de ancho constante con tapa redondeada -- la forma del
    banano sintetico, y una aproximacion razonable de las puntas reales --
    el radio del circulo inscripto es EXACTAMENTE constante (igual al radio
    local ``r``) a lo largo de todo el eje central, desde el cuerpo hasta el
    CENTRO de la tapa, que es justo donde termina el eje medial. Mas alla de
    ese centro, a la misma distancia ``r``, esta el borde exterior de la
    mascara. Por eso, caminar desde la punta aproximada en la direccion
    local (estimada contra un punto bien adentro del cuerpo) hasta salir de
    la mascara, y retroceder ``r`` desde ese borde, reconstruye el centro de
    la tapa con precision subpixel -- sin importar que la posicion de
    partida haya sido imprecisa por la cuadricula reducida.

    Esta reconstruccion asume una tapa convexa (redondeada): es exacta para
    el extremo floral/pedicelo tipico de un banano real, pero NO para un
    corte plano abrupto (alli el eje medial real termina antes, no en
    ``borde - r``). Si el barrido no encuentra un borde en un numero
    razonable de pasos, o el punto de partida ya esta fuera de la mascara,
    devuelve ``approx_xy`` sin modificar en vez de arriesgar un valor peor.
    """
    height, width = mask_bool.shape

    def is_inside(point: np.ndarray) -> bool:
        col, row = int(round(point[0])), int(round(point[1]))
        if col < 0 or col >= width or row < 0 or row >= height:
            return False
        return bool(mask_bool[row, col])

    delta = approx_xy - inward_xy
    norm = np.linalg.norm(delta)
    if norm < 1e-6:
        return approx_xy
    direction = delta / norm

    row0 = int(np.clip(round(approx_xy[1]), 0, height - 1))
    col0 = int(np.clip(round(approx_xy[0]), 0, width - 1))
    local_radius = float(distance_full[row0, col0])

    point = approx_xy.copy()
    if not is_inside(point):
        # El punto aproximado puede caer justo afuera por redondeo: retroceder
        # un poco hacia el cuerpo antes de empezar el barrido hacia afuera.
        nudged = False
        for _ in range(8):
            point = point - direction * _TIP_REFINE_STEP_PX
            if is_inside(point):
                nudged = True
                break
        if not nudged:
            return approx_xy

    max_steps = max(_TIP_REFINE_MIN_STEPS, int(round(local_radius * _TIP_REFINE_MAX_RADIUS_MULT / _TIP_REFINE_STEP_PX)))
    last_inside = point.copy()
    found_boundary = False
    for _ in range(max_steps):
        point = point + direction * _TIP_REFINE_STEP_PX
        if not is_inside(point):
            found_boundary = True
            break
        last_inside = point.copy()

    if not found_boundary:
        return approx_xy

    return last_inside - direction * local_radius


def medial_axis_path(mask: np.ndarray) -> MedialAxis | None:
    """Eje medial podado (camino geodesico mas largo) de una mascara binaria.

    Devuelve ``None`` si la mascara es degenerada: vacia, demasiado pequena
    para tener un esqueleto de al menos 2 pixeles, o sin dos puntas
    significativas (un disco perfecto, cuya elongacion -- longitud del
    camino sobre el doble del radio maximo -- es cercana a 1, muy por debajo
    del umbral de un banano real). Nunca lanza una excepcion por una mascara
    degenerada.
    """
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return None

    small_mask, scale_x, scale_y = _downsample_mask(mask_bool, _SKELETON_DOWNSAMPLE)
    skeleton_small = skeletonize(small_mask)

    path = _longest_geodesic_path(skeleton_small)
    if path is None:
        return None
    rows_small, cols_small = path

    distance_full = cv2.distanceTransform(mask_bool.astype(np.uint8) * 255, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    height, width = mask_bool.shape

    x_full = cols_small * scale_x
    y_full = rows_small * scale_y
    path_px = np.stack([x_full, y_full], axis=1)

    rows_full = np.clip(np.round(y_full).astype(np.int64), 0, height - 1)
    cols_full = np.clip(np.round(x_full).astype(np.int64), 0, width - 1)
    radius_px = distance_full[rows_full, cols_full]

    path_length_px = arc_length_px(path_px)
    max_radius_px = float(radius_px.max()) if len(radius_px) else 0.0
    elongation = path_length_px / (2.0 * max_radius_px) if max_radius_px > 0.0 else 0.0

    if path_length_px < _MIN_PATH_LENGTH_PX or elongation < _MIN_ELONGATION:
        return None

    # La cuadricula de media resolucion pierde hasta un par de pixeles de
    # precision justo en las dos puntas (ver docstring del modulo): se
    # refinan aqui, a resolucion completa, antes de devolver el resultado.
    # El punto "hacia adentro" para estimar la direccion local se toma a un
    # 10% de la longitud del camino desde cada punta, no al vecino inmediato,
    # para promediar el zigzag de la cuadricula reducida.
    inward_index = max(2, int(round(len(path_px) * 0.1)))
    inward_index = min(inward_index, len(path_px) - 1)
    path_px[0] = _refine_tip_px(mask_bool, distance_full, path_px[0], path_px[inward_index])
    path_px[-1] = _refine_tip_px(mask_bool, distance_full, path_px[-1], path_px[-1 - inward_index])
    for end_index in (0, -1):
        row = int(np.clip(round(path_px[end_index][1]), 0, height - 1))
        col = int(np.clip(round(path_px[end_index][0]), 0, width - 1))
        radius_px[end_index] = distance_full[row, col]

    return MedialAxis(path_px=path_px, radius_px=radius_px, tips_px=(path_px[0], path_px[-1]))


def simplify_path_px(path_px: np.ndarray, epsilon_px: float = _DEFAULT_SIMPLIFY_EPSILON_PX) -> np.ndarray:
    """Simplifica ``path_px`` con Douglas-Peucker (``cv2.approxPolyDP``).

    El camino crudo extraido del esqueleto en zigzag ("escalera") entre
    pixeles vecinos sobreestima la longitud real de una curva suave aun
    pesando las diagonales con ``sqrt(2)``: reemplazar los microescalones
    por cuerdas rectas, dentro de ``epsilon_px`` del camino original, corrige
    ese sesgo sin alisar ni desplazar las puntas (los extremos del camino
    siempre se conservan). ``epsilon_px`` es insensible en un rango amplio
    (de 1 a 10 px sobre el banano de referencia a 0.2 mm/px); el valor por
    defecto cae comodo en el medio.

    Expuesta por separado (no solo usada dentro de ``arc_length_px``) porque
    Task 7 construye su propia longitud en milimetros sobre este mismo
    camino ya simplificado, en vez de reimplementar la correccion.
    """
    if len(path_px) < 3:
        return np.asarray(path_px, dtype=np.float64)
    contour = np.asarray(path_px, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(contour, epsilon_px, closed=False)
    return simplified.reshape(-1, 2).astype(np.float64)


def arc_length_px(path_px: np.ndarray) -> float:
    """Longitud, en pixeles, de ``path_px`` tras corregir el sesgo de
    "escalera" del camino crudo del esqueleto con ``simplify_path_px``.

    Sumar los segmentos crudos entre pixeles vecinos del esqueleto
    sobreestima la longitud real de una curva suave (medido +6.8% sobre el
    banano de referencia, muy por encima de cualquier tolerancia razonable);
    por eso esta funcion nunca suma el camino sin simplificar.
    """
    path_px = np.asarray(path_px, dtype=np.float64)
    if len(path_px) < 2:
        return 0.0
    simplified = simplify_path_px(path_px)
    if len(simplified) < 2:
        return 0.0
    diffs = np.diff(simplified, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))


def width_profile_mm(axis: MedialAxis, mm_per_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Perfil de grosor ``w(s)`` en milimetros, a lo largo de ``axis``.

    ``s`` es la abscisa curvilinea (longitud acumulada del camino CRUDO,
    sin simplificar: cada valor de ``s`` debe corresponder al mismo indice
    de pixel que su ``w``, y simplificar perderia esa correspondencia
    punto a punto) en milimetros desde la primera punta. ``w`` es el ancho
    local (el diametro del circulo inscripto, el doble del radio que
    entrega la transformada de distancia) tambien en milimetros.
    """
    diffs = np.diff(axis.path_px, axis=0)
    step_lengths_px = np.sqrt(np.sum(diffs ** 2, axis=1))
    s_px = np.concatenate([[0.0], np.cumsum(step_lengths_px)])
    s_mm = s_px * mm_per_px
    w_mm = axis.radius_px * 2.0 * mm_per_px
    return s_mm, w_mm

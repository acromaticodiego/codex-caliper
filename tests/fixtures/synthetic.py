"""Generador de bananos sinteticos con verdad analitica exacta.

Un banano se modela como una **capsula de ancho constante barrida sobre un
arco de circunferencia** de radio ``R`` y angulo ``theta``: esa forma tiene
longitud, area y volumen en forma cerrada, asi que las pruebas de geometria
(Tasks 6, 7, 8) se validan contra una respuesta exacta, no contra otra
medicion.

Geometria interna (coordenadas locales en mm, antes de pasar a pixeles):

- La circunferencia que sostiene el arco esta centrada en el origen.
- La linea media se parametriza por ``phi`` en ``[-theta/2, theta/2]``:
  ``point(phi) = (R*sin(phi), -R*cos(phi))``. El extremo del pedicelo esta
  en ``phi = -theta/2`` y el extremo floral en ``phi = +theta/2``; el arco
  queda simetrico respecto al eje Y, con la "panza" del banano hacia abajo.
- La capsula se rasteriza estampando circulos (``cv2.circle``) de radio
  ``width_mm/2`` en puntos muestreados finamente a lo largo del arco: los
  circulos de las puntas producen, por construccion, las tapas redondeadas,
  y el area total de ambas tapas equivale a un circulo completo
  (``pi*(w/2)**2``), que es justo el termino que aparece en la formula
  analitica del area.
- El rasterizado ocurre en una grilla sobremuestreada (``_SUPERSAMPLE``
  veces mas fina que ``mm_per_px``) que luego se reduce a la resolucion
  pedida promediando cada bloque y fijando el pixel de salida si al menos
  la mitad de su area quedo cubierta. ``cv2.circle`` en resolucion nativa
  sesga el area hacia arriba (incluye generosamente los pixeles de borde);
  ese sesgo crece cuanto mas grueso es ``mm_per_px`` frente al calibre, y
  sobremuestrear lo mantiene por debajo del 1% en todo el rango de
  parametros, no solo en el caso que se probo primero.
- El pedicelo es un segundo barrido, de radio menor y constante, que
  arranca en la punta del extremo ``phi=-theta/2`` y se aleja en linea
  recta, tangente al arco en ese punto (no vuelve a curvarse).
- El adelgazamiento floral (``blossom_taper_mm``) reduce linealmente el
  radio de los circulos estampados en el ultimo tramo de esa longitud antes
  del extremo floral, hasta llegar a radio 0 justo en la punta: es una
  caida *gradual*, a diferencia de la caida *abrupta* de ancho en la union
  con el pedicelo (donde el ancho pasa de golpe de ``width_mm`` a
  ``pedicel_width_mm``). Esa diferencia de perfil es lo que permite a la
  Task 8 distinguir un extremo del otro.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np

# Pasos de muestreo mas finos que un pixel evitan el "escalonado" del borde
# que produce estampar circulos demasiado espaciados (ver Task 2, nota 5).
_SAMPLE_STEP_FACTOR = 0.5
_MIN_SAMPLES = 8
# Bits de precision subpixel usados en cv2.circle via el parametro `shift`.
_CIRCLE_SHIFT_BITS = 8
# Factor de sobremuestreo del rasterizado: la grilla interna tiene
# `_SUPERSAMPLE` veces mas pixeles por eje que `mm_per_px`, y se reduce
# promediando con umbral de 50% de area cubierta (ver fix round 1, Task 2:
# el rasterizado a resolucion nativa sobreestimaba el area hasta ~3.4% en
# el extremo grueso/delgado del rango de parametros).
_SUPERSAMPLE = 8

_SHADOW_OFFSET_MM = (5.0, 5.0)
_SHADOW_BGR = (120, 120, 120)
_YELLOW_BGR = (0, 216, 255)
_BANANA_GAP_MM = 30.0


@dataclasses.dataclass(frozen=True)
class GroundTruth:
    """Verdad analitica de un banano sintetico (capsula sobre arco).

    Depende solo de ``radius_mm``, ``arc_rad`` y ``width_mm``: el pedicelo y
    el adelgazamiento floral cambian la mascara rasterizada pero no estas
    cantidades, que describen la capsula "pura".
    """

    centerline_length_mm: float
    outer_length_mm: float
    caliber_mm: float
    area_mm2: float
    volume_mm3: float


def _centerline_point(radius_mm: float, phi: np.ndarray) -> np.ndarray:
    """Puntos ``(x, y)`` en mm sobre la circunferencia de radio ``radius_mm``."""
    return np.stack([radius_mm * np.sin(phi), -radius_mm * np.cos(phi)], axis=-1)


def _tangent_unit(phi: float) -> np.ndarray:
    """Vector unitario tangente a la circunferencia en el angulo ``phi``."""
    return np.array([math.cos(phi), math.sin(phi)])


def _num_samples(length_mm: float, mm_per_px: float) -> int:
    """Cantidad de muestras para recorrer ``length_mm`` sin dejar huecos.

    El paso se fija en una fraccion de pixel (``mm_per_px``) para que los
    circulos estampados consecutivos se superpongan ampliamente y el borde
    rasterizado no quede escalonado.
    """
    step_mm = mm_per_px * _SAMPLE_STEP_FACTOR
    return max(_MIN_SAMPLES, int(math.ceil(length_mm / step_mm)) + 1)


def _arc_stamps(radius_mm: float, arc_rad: float, width_mm: float, mm_per_px: float, blossom_taper_mm: float) -> list[tuple[np.ndarray, float]]:
    """Centros y radios (en mm) de los circulos que forman el cuerpo del banano."""
    half_w = width_mm / 2.0
    centerline_length_mm = radius_mm * arc_rad
    n = _num_samples(centerline_length_mm, mm_per_px)
    phis = np.linspace(-arc_rad / 2.0, arc_rad / 2.0, n)
    points = _centerline_point(radius_mm, phis)
    arc_lengths = radius_mm * (phis + arc_rad / 2.0)

    radii = np.full(n, half_w)
    if blossom_taper_mm > 0.0:
        taper_start = centerline_length_mm - blossom_taper_mm
        in_taper = arc_lengths > taper_start
        remaining = np.clip(centerline_length_mm - arc_lengths, 0.0, None)
        radii[in_taper] = half_w * np.clip(remaining[in_taper] / blossom_taper_mm, 0.0, 1.0)

    return list(zip(points, radii))


def _pedicel_stamps(radius_mm: float, arc_rad: float, pedicel_mm: float, pedicel_width_mm: float, mm_per_px: float) -> list[tuple[np.ndarray, float]]:
    """Centros y radios (en mm) del pedicelo: un barrido recto desde la punta."""
    if pedicel_mm <= 0.0 or pedicel_width_mm <= 0.0:
        return []
    tip_phi = -arc_rad / 2.0
    tip = _centerline_point(radius_mm, np.array([tip_phi]))[0]
    direction = -_tangent_unit(tip_phi)
    n = _num_samples(pedicel_mm, mm_per_px)
    ts = np.linspace(0.0, pedicel_mm, n)
    points = tip + np.outer(ts, direction)
    radius = pedicel_width_mm / 2.0
    return [(p, radius) for p in points]


def _rasterize(stamps: list[tuple[np.ndarray, float]], mm_per_px: float, canvas_margin_mm: float) -> np.ndarray:
    """Convierte una lista de (centro_mm, radio_mm) en una mascara uint8 0/255.

    Rasteriza en una grilla ``_SUPERSAMPLE`` veces mas fina que ``mm_per_px``
    y reduce con ``cv2.resize(..., INTER_AREA)``: sobre un factor entero
    exacto, ``INTER_AREA`` promedia cada bloque de subpixeles sin
    interpolar, asi que el valor resultante por pixel es exactamente la
    fraccion de su area cubierta por la mascara. Umbralizar ese promedio en
    128 (la mitad de 255) implementa la regla de "al menos la mitad del
    area del pixel cubierta" y elimina el sesgo sistematico de
    ``cv2.circle`` a resolucion nativa, que sobreestima el area tanto mas
    cuanto mas grueso es ``mm_per_px`` frente al calibre del banano.
    """
    xs = np.array([p[0] for p, _ in stamps])
    ys = np.array([p[1] for p, _ in stamps])
    radii = np.array([r for _, r in stamps])

    min_x = float((xs - radii).min()) - canvas_margin_mm
    max_x = float((xs + radii).max()) + canvas_margin_mm
    min_y = float((ys - radii).min()) - canvas_margin_mm
    max_y = float((ys + radii).max()) + canvas_margin_mm

    width_px = max(1, int(math.ceil((max_x - min_x) / mm_per_px)))
    height_px = max(1, int(math.ceil((max_y - min_y) / mm_per_px)))

    supersample = _SUPERSAMPLE
    mm_per_px_hi = mm_per_px / supersample
    width_px_hi = width_px * supersample
    height_px_hi = height_px * supersample
    hi_res_mask = np.zeros((height_px_hi, width_px_hi), dtype=np.uint8)

    shift = _CIRCLE_SHIFT_BITS
    scale = (1 << shift) / mm_per_px_hi
    for (x_mm, y_mm), r_mm in stamps:
        if r_mm <= 0.0:
            continue
        cx = int(round((x_mm - min_x) * scale))
        cy = int(round((y_mm - min_y) * scale))
        r = int(round(r_mm * scale))
        cv2.circle(hi_res_mask, (cx, cy), r, 255, -1, cv2.LINE_8, shift)

    coverage = cv2.resize(hi_res_mask, (width_px, height_px), interpolation=cv2.INTER_AREA)
    mask = np.where(coverage >= 128, np.uint8(255), np.uint8(0))

    return mask


def make_banana(
    radius_mm: float,
    arc_rad: float,
    width_mm: float,
    mm_per_px: float,
    pedicel_mm: float = 0.0,
    pedicel_width_mm: float = 0.0,
    blossom_taper_mm: float = 0.0,
    canvas_margin_mm: float = 20.0,
) -> tuple[np.ndarray, GroundTruth]:
    """Genera un banano sintetico y su verdad analitica exacta.

    El banano es una capsula de ancho ``width_mm`` barrida sobre un arco de
    circunferencia de radio ``radius_mm`` y angulo ``arc_rad``. ``pedicel_mm``
    / ``pedicel_width_mm`` agregan un pedicelo recto en la punta opuesta al
    extremo floral; ``blossom_taper_mm`` adelgaza gradualmente ese extremo
    floral. Ninguno de los dos cambia la verdad analitica, que describe solo
    la capsula pura.
    """
    half_w = width_mm / 2.0
    centerline_length_mm = radius_mm * arc_rad
    outer_length_mm = (radius_mm + half_w) * arc_rad
    area_mm2 = width_mm * radius_mm * arc_rad + math.pi * half_w ** 2
    volume_mm3 = math.pi * half_w ** 2 * radius_mm * arc_rad + (4.0 / 3.0) * math.pi * half_w ** 3
    gt = GroundTruth(
        centerline_length_mm=centerline_length_mm,
        outer_length_mm=outer_length_mm,
        caliber_mm=width_mm,
        area_mm2=area_mm2,
        volume_mm3=volume_mm3,
    )

    stamps = _arc_stamps(radius_mm, arc_rad, width_mm, mm_per_px, blossom_taper_mm)
    stamps += _pedicel_stamps(radius_mm, arc_rad, pedicel_mm, pedicel_width_mm, mm_per_px)
    mask = _rasterize(stamps, mm_per_px, canvas_margin_mm)

    return mask, gt


def _stamp_masked(canvas: np.ndarray, mask: np.ndarray, top_left_px: tuple[int, int], color: tuple[int, int, int]) -> None:
    """Pinta ``color`` en ``canvas`` donde ``mask`` es 255, recortando a sus bordes."""
    x0, y0 = top_left_px
    h, w = mask.shape
    height, width = canvas.shape[:2]

    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + w, width), min(y0 + h, height)
    if cx0 >= cx1 or cy0 >= cy1:
        return

    mx0, my0 = cx0 - x0, cy0 - y0
    mx1, my1 = mx0 + (cx1 - cx0), my0 + (cy1 - cy0)

    roi = canvas[cy0:cy1, cx0:cx1]
    mroi = mask[my0:my1, mx0:mx1]
    roi[mroi == 255] = color


def _stamp_full(canvas: np.ndarray, patch_bgr: np.ndarray, top_left_px: tuple[int, int]) -> None:
    """Copia ``patch_bgr`` entero sobre ``canvas``, recortando a sus bordes."""
    x0, y0 = top_left_px
    h, w = patch_bgr.shape[:2]
    height, width = canvas.shape[:2]

    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + w, width), min(y0 + h, height)
    if cx0 >= cx1 or cy0 >= cy1:
        return

    px0, py0 = cx0 - x0, cy0 - y0
    px1, py1 = px0 + (cx1 - cx0), py0 + (cy1 - cy0)

    canvas[cy0:cy1, cx0:cx1] = patch_bgr[py0:py1, px0:px1]


def make_banana_scene(
    bananas: list[dict],
    mm_per_px: float,
    markers: dict[int, tuple[float, float]] | None = None,
    marker_dict: str = "DICT_6X6_50",
) -> np.ndarray:
    """Compone una escena BGR con bananos amarillos sobre fondo blanco.

    Cada banano se renderiza con :func:`make_banana` a partir de las claves
    de su dict (mismos nombres que los parametros de ``make_banana``, mas un
    ``position_mm`` opcional para fijar su esquina superior izquierda en la
    escena; si se omite, los bananos se acomodan en fila). Cada uno proyecta
    una sombra gris neutra (B=G=R), desplazada unos milimetros: oscura en
    luminancia pero sin saturacion, para que la segmentacion por canal de
    saturacion (Task 13) pueda distinguirla de un banano real.

    Si ``markers`` no es ``None``, dibuja un ArUco de 24 mm (``marker_dict``,
    via ``cv2.aruco.generateImageMarker``) centrado en cada posicion (mm)
    dada, para que la escena sea calibrable por el pipeline (Task 14).
    """
    rendered = []
    cursor_x_mm = 0.0
    for spec in bananas:
        kwargs = {
            "radius_mm": spec["radius_mm"],
            "arc_rad": spec["arc_rad"],
            "width_mm": spec["width_mm"],
            "mm_per_px": mm_per_px,
            "pedicel_mm": spec.get("pedicel_mm", 0.0),
            "pedicel_width_mm": spec.get("pedicel_width_mm", 0.0),
            "blossom_taper_mm": spec.get("blossom_taper_mm", 0.0),
            "canvas_margin_mm": spec.get("canvas_margin_mm", 20.0),
        }
        mask, _ = make_banana(**kwargs)
        mask_w_mm = mask.shape[1] * mm_per_px
        mask_h_mm = mask.shape[0] * mm_per_px
        position_mm = spec.get("position_mm")
        if position_mm is None:
            position_mm = (cursor_x_mm, 0.0)
            cursor_x_mm += mask_w_mm + _BANANA_GAP_MM
        rendered.append((mask, position_mm, mask_w_mm, mask_h_mm))

    scene_margin_mm = 20.0
    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    for _, (x, y), w_mm, h_mm in rendered:
        min_x, max_x = min(min_x, x), max(max_x, x + w_mm)
        min_y, max_y = min(min_y, y), max(max_y, y + h_mm)
    if markers:
        half_marker_mm = 12.0
        for x_mm, y_mm in markers.values():
            min_x = min(min_x, x_mm - half_marker_mm)
            max_x = max(max_x, x_mm + half_marker_mm)
            min_y = min(min_y, y_mm - half_marker_mm)
            max_y = max(max_y, y_mm + half_marker_mm)
    if not rendered and not markers:
        min_x, min_y, max_x, max_y = 0.0, 0.0, 0.0, 0.0

    min_x -= scene_margin_mm
    min_y -= scene_margin_mm
    max_x += scene_margin_mm
    max_y += scene_margin_mm

    width_px = max(1, int(math.ceil((max_x - min_x) / mm_per_px)))
    height_px = max(1, int(math.ceil((max_y - min_y) / mm_per_px)))
    scene = np.full((height_px, width_px, 3), 255, np.uint8)

    def to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
        return (
            int(round((x_mm - min_x) / mm_per_px)),
            int(round((y_mm - min_y) / mm_per_px)),
        )

    shadow_dx_px = int(round(_SHADOW_OFFSET_MM[0] / mm_per_px))
    shadow_dy_px = int(round(_SHADOW_OFFSET_MM[1] / mm_per_px))

    for mask, (x_mm, y_mm), _, _ in rendered:
        top_left_px = to_px(x_mm, y_mm)
        shadow_top_left_px = (top_left_px[0] + shadow_dx_px, top_left_px[1] + shadow_dy_px)
        _stamp_masked(scene, mask, shadow_top_left_px, _SHADOW_BGR)

    for mask, (x_mm, y_mm), _, _ in rendered:
        top_left_px = to_px(x_mm, y_mm)
        _stamp_masked(scene, mask, top_left_px, _YELLOW_BGR)

    if markers:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, marker_dict))
        marker_side_px = max(1, int(round(24.0 / mm_per_px)))
        for marker_id, (x_mm, y_mm) in markers.items():
            marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_side_px)
            marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
            cx_px, cy_px = to_px(x_mm, y_mm)
            top_left_px = (cx_px - marker_side_px // 2, cy_px - marker_side_px // 2)
            _stamp_full(scene, marker_bgr, top_left_px)

    return scene

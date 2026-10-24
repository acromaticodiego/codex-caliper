"""Refinamiento de contorno por saturacion (Task 13).

Hinge entre deteccion neuronal y medicion clasica: YOLO (Task 12) entrega una
mascara GRUESA que dice DONDE esta el banano -- rapida, pero una mascara
COCO esta anotada a mano con criterio de velocidad, no de precision (ver
spec S6). Medir longitud/calibre/area directamente sobre esa mascara hereda
el error del anotador humano. ``refine_mask`` no confia en el contorno de
YOLO; solo usa su caja como punto de partida para recortar una region de
interes, y repone el contorno real con CV clasica sobre esa region.

Por que saturacion y no luminancia (spec S5.1.c, S5.1.d)
---------------------------------------------------------
Sobre el fondo blanco del montaje, la sombra propia del banano es oscura en
LUMINANCIA pero tan desaturada como el fondo (B=G=R, saturacion ~0): un
umbral de luminancia metería la sombra dentro de la mascara (es oscura);
un umbral de SATURACION no, porque la sombra y el fondo caen del mismo
lado. En el rig real, de 913 372 pixeles oscuros-pero-desaturados (la
sombra propia mas las patas negras del soporte), solo 57 terminaron dentro
de la mascara de banano tras segmentar por saturacion -- 0.006% (S5.1.c).
Esa es la evidencia empirica que decide la pregunta que S7 etapa 5 deja
abierta ("alternativas a comparar empiricamente": LAB b*, GrabCut); para
este montaje ya esta respondida, asi que este modulo implementa UNICAMENTE
saturacion (Ruling 4 de Task 13) -- agregar los otros dos canales/metodos
seria alcance no solicitado sobre una pregunta ya cerrada.

El piso absoluto de saturacion (Ruling 2)
------------------------------------------
Otsu SIEMPRE devuelve un umbral, incluso sobre un recorte sin nada real que
segmentar: ante una imagen practicamente uniforme (fondo blanco plano mas
ruido de sensor), Otsu parte ese ruido en dos clases y entrega una mascara
con la FORMA de ruido, no la ausencia de fruta -- un numero plausible y
silenciosamente falso, el patron de fallo que este proyecto ya repitio. Por
eso, antes de confiar en Otsu, se exige un piso ABSOLUTO: si muy pocos
pixeles del recorte superan ``_ABSOLUTE_SATURATION_FLOOR``, no hay nada que
segmentar y se devuelve ``None``, sin correr Otsu.

Valor elegido: ``_ABSOLUTE_SATURATION_FLOOR = 20``. En el rig real (S5.1.d):
la cartulina blanca esta en saturacion ~7, el parpadeo de los LED la mueve
+-1.2 a 1.8 unidades (techo de ruido realista ~10), y el umbral operativo de
Otsu cae cerca de 83 con margen ~55 sobre ese fondo. Un piso de 20 queda
~10 unidades por encima del techo de ruido (no dispara con cartulina mas
parpadeo) y ~25 unidades por debajo de la zona mas oscura que puede tener
un banano REAL muy maduro (manchas en 45-61, S5.1.d) -- nunca descarta fruta
real, incluso muy madura, pero si descarta un recorte sin fruta. No es el
umbral de SEGMENTACION (ese lo decide Otsu, o ``saturation_threshold`` si se
pasa); es el umbral de "hay algo aqui o no".

``_MIN_SATURATED_PIXELS = 25`` evita que un unico pixel caliente de ruido
(un solo pixel por encima del piso, por azar) cuente como "hay fruta": exige
un minimo de pixeles por encima del piso, no solo un valor extremo aislado.

Cierre y apertura morfologicos (spec S5.1.d)
---------------------------------------------
Las manchas oscuras de un banano muy maduro caen a saturacion 45-61, POR
DEBAJO del umbral operativo (~83) que separa fruta de fondo -- sin cierre,
esas manchas quedan como agujeros (o cortan la mascara) en vez de contarse
como fruta. "El cierre es estructural, no cosmetico" (ver Ruling de Task
13): perder area en una fruta muy madura con mucha superficie oscura es un
riesgo abierto, documentado aqui, a validar contra fruta real en distintos
estados de madurez -- el fixture sintetico no modela madurez/manchas, asi
que esa validacion queda pendiente de imagenes reales, no de este modulo.
La apertura, antes del cierre, quita motas pequenas (ruido de sensor, una
fibra del soporte) que de otro modo sobrevivirian como componentes propios.

Margen de recorte del bbox (Ruling 3)
---------------------------------------
Recortar exactamente al bbox de ``coarse_mask`` (como pide la letra del
Task Brief) hace que el refinamiento NUNCA pueda crecer mas alla de la caja
que dibujo YOLO: hereda el error del anotador/detector, que es precisamente
lo que S6 declara inaceptable y la razon de que exista medicion clasica en
este diseño. ``_BBOX_MARGIN_MM = 12.0`` (60 px a ``_CANVAS_MM_PER_PX = 0.2``,
la escala fija del lienzo rectificado en todo el proyecto) le da margen a
una caja YOLO que recorto parte de la fruta real para recuperarla igual, sin
llegar a alcanzar al banano vecino mas cercano en la escena sintetica de
pruebas (separados 30 mm, ``_BANANA_GAP_MM`` en ``tests/fixtures/synthetic``)
ni en el rig real, donde los bananos del plato se colocan separados a mano.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Escala fija del lienzo rectificado en todo el proyecto (ver
#: ``AppConfig.canvas_mm_per_px``). ``refine_mask`` no recibe ``mm_per_px``
#: porque esta funcion opera siempre sobre esa escala -- nunca sobre el
#: frame crudo.
_CANVAS_MM_PER_PX = 0.2

#: Margen de recorte alrededor del bbox de ``coarse_mask``, en mm y en px a
#: la escala fija del lienzo (ver docstring del modulo, Ruling 3).
_BBOX_MARGIN_MM = 12.0
_BBOX_MARGIN_PX = round(_BBOX_MARGIN_MM / _CANVAS_MM_PER_PX)

#: Piso absoluto de saturacion para decidir si hay algo que segmentar (ver
#: docstring del modulo, Ruling 2).
_ABSOLUTE_SATURATION_FLOOR = 20

#: Minimo de pixeles por encima del piso para no confundir un pixel de
#: ruido aislado con la presencia real de fruta.
_MIN_SATURATED_PIXELS = 25

#: Lado (px) del kernel cuadrado de apertura (quita motas chicas) y de
#: cierre (recupera manchas oscuras / agujeros chicos). Ver docstring del
#: modulo.
_OPEN_KERNEL_PX = 3
_CLOSE_KERNEL_PX = 9


def _padded_bbox(coarse_mask: np.ndarray, margin_px: int) -> tuple[int, int, int, int] | None:
    """Caja ``(x0, y0, x1, y1)`` (inclusive) del bbox de ``coarse_mask``, con
    ``margin_px`` de margen por lado y recortada a los limites de la mascara.

    ``None`` si ``coarse_mask`` esta vacia -- no hay caja que calcular.
    """
    mask_bool = np.asarray(coarse_mask).astype(bool)
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return None

    height, width = mask_bool.shape
    x0 = max(0, int(xs.min()) - margin_px)
    y0 = max(0, int(ys.min()) - margin_px)
    x1 = min(width - 1, int(xs.max()) + margin_px)
    y1 = min(height - 1, int(ys.max()) + margin_px)
    return x0, y0, x1, y1


def refine_mask(
    rectified_bgr: np.ndarray,
    coarse_mask: np.ndarray,
    saturation_threshold: int | None = None,
) -> np.ndarray | None:
    """Refina ``coarse_mask`` segmentando por el canal de SATURACION de
    ``rectified_bgr`` dentro de su bbox (con margen, ver Ruling 3).

    Devuelve una mascara booleana del mismo tamano que ``rectified_bgr``
    (``True`` solo dentro del componente conexo mas grande encontrado), o
    ``None`` si ``coarse_mask`` esta vacia, si el recorte no tiene suficiente
    saturacion real para confiar en Otsu (ver ``_ABSOLUTE_SATURATION_FLOOR``),
    o si no sobrevive ningun componente conexo tras la limpieza morfologica.

    ``saturation_threshold`` fuerza el umbral de segmentacion en vez de
    calcularlo con Otsu (util para pruebas y para calibrar el rig real);
    el piso absoluto se exige igual, independientemente del umbral usado --
    decide si hay algo que segmentar, no como segmentarlo.
    """
    bbox = _padded_bbox(coarse_mask, _BBOX_MARGIN_PX)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox

    crop_bgr = rectified_bgr[y0 : y1 + 1, x0 : x1 + 1]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1]

    if int(np.count_nonzero(saturation > _ABSOLUTE_SATURATION_FLOOR)) < _MIN_SATURATED_PIXELS:
        return None

    if saturation_threshold is None:
        threshold, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        threshold = float(saturation_threshold)

    binary = (saturation > threshold).astype(np.uint8) * 255

    open_kernel = np.ones((_OPEN_KERNEL_PX, _OPEN_KERNEL_PX), np.uint8)
    close_kernel = np.ones((_CLOSE_KERNEL_PX, _CLOSE_KERNEL_PX), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)

    num_labels, labels = cv2.connectedComponents(closed)
    if num_labels <= 1:
        # label 0 es el fondo; sin un label 1 no sobrevivio ningun componente.
        return None

    counts = np.bincount(labels.ravel())
    counts[0] = 0  # el fondo nunca puede "ganar" como componente mas grande.
    largest_label = int(np.argmax(counts))
    if counts[largest_label] == 0:
        return None

    full_mask = np.zeros(rectified_bgr.shape[:2], dtype=bool)
    full_mask[y0 : y1 + 1, x0 : x1 + 1] = labels == largest_label
    return full_mask

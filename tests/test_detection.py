"""Tests de Task 12: deteccion YOLO con tracking y reasociacion.

Convencion de este archivo: ``ultralytics`` nunca se importa (ni directa ni
indirectamente) -- todos los tests inyectan un doble via ``model=...`` o
construyen ``Instance`` a mano. El import real de ``ultralytics.YOLO`` vive
dentro de ``BananaDetector.__init__`` y solo se ejecuta cuando no se inyecta
un modelo; ningun test de este archivo toma esa rama.
"""

from __future__ import annotations

import numpy as np
import pytest

from codex_caliper.detection import (
    YOLO_BANANA_CLASS_ID,
    BananaDetector,
    CentroidReassigner,
    Instance,
    masks_are_adjacent,
)


# ---------------------------------------------------------------------------
# Doble de pruebas de ultralytics.YOLO
# ---------------------------------------------------------------------------


class _FakeBoxes:
    def __init__(self, cls, conf, ids, xyxy):
        self.cls = np.asarray(cls, dtype=float)
        self.conf = np.asarray(conf, dtype=float)
        self.id = np.asarray(ids, dtype=float) if ids is not None else None
        self.xyxy = np.asarray(xyxy, dtype=float)


class _FakeMasks:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.uint8)


class _FakeResult:
    def __init__(self, boxes, masks):
        self.boxes = boxes
        self.masks = masks


class _FakeModel:
    """Doble de ``ultralytics.YOLO``: no descarga pesos, cuenta llamadas y
    devuelve siempre el mismo resultado fijo (simula un frame estatico)."""

    def __init__(self, result):
        self._result = result
        self.call_count = 0
        self.calls = []

    def track(self, frame, persist=True, classes=None):
        self.call_count += 1
        self.calls.append({"shape": frame.shape, "persist": persist, "classes": classes})
        return [self._result]


_FRAME_SIZE = 64


def _two_class_result(size=_FRAME_SIZE):
    """Resultado falso con una banana (clase 46, id 1) y OTRA clase (47, id
    2) -- deliberadamente devuelve ambas aunque se le haya pedido
    ``classes=[46]``, para que el test de filtro de clase no dependa de que
    el doble respete ese argumento (ningun modelo real deberia asumirse
    confiable en eso tampoco)."""
    mask_banana = np.zeros((size, size), dtype=np.uint8)
    mask_banana[10:20, 10:20] = 1
    mask_other = np.zeros((size, size), dtype=np.uint8)
    mask_other[40:50, 40:50] = 1
    boxes = _FakeBoxes(
        cls=[46, 47],
        conf=[0.9, 0.8],
        ids=[1, 2],
        xyxy=[[10, 10, 20, 20], [40, 40, 50, 50]],
    )
    masks = _FakeMasks([mask_banana, mask_other])
    return _FakeResult(boxes, masks)


def _blank_frame(size=_FRAME_SIZE):
    return np.zeros((size, size, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# BananaDetector
# ---------------------------------------------------------------------------


def test_only_the_banana_class_survives():
    """No alcanza con `all(i.class_id == 46 for i in out)`: eso pasa igual
    si `detect` devuelve `[]`. Se exige ademas el conteo exacto y que el
    sobreviviente sea especificamente la instancia de clase 46 (track_id=1),
    no la de clase 47 (track_id=2)."""
    fake_model = _FakeModel(_two_class_result())
    det = BananaDetector(infer_size=_FRAME_SIZE, model=fake_model)

    out = det.detect(_blank_frame())

    assert len(out) == 1
    assert all(i.class_id == YOLO_BANANA_CLASS_ID for i in out)
    assert out[0].track_id == 1


def test_skipped_frames_reuse_the_previous_instances():
    """No alcanza con `call_count == 1`: eso pasa igual si los frames
    saltados devuelven `[]`. Se exige ademas que lo devuelto en los frames
    saltados sea IGUAL (mismos track_id, mismas mascaras) a lo que devolvio
    el frame que si corrio inferencia."""
    fake_model = _FakeModel(_two_class_result())
    det = BananaDetector(every_n_frames=3, infer_size=_FRAME_SIZE, model=fake_model)
    frame = _blank_frame()

    first = det.detect(frame)
    second = det.detect(frame)
    third = det.detect(frame)

    assert fake_model.call_count == 1
    assert len(first) == 1  # tras el filtro de clase; si esto fuera 0 el resto no probaria nada
    for skipped in (second, third):
        assert [i.track_id for i in skipped] == [i.track_id for i in first]
        assert len(skipped) == len(first)
        assert all(np.array_equal(a.mask, b.mask) for a, b in zip(skipped, first))


def test_every_n_frames_one_infers_every_frame():
    """Caso por defecto (`every_n_frames=1`): nunca se reutiliza, el modelo
    se llama en cada frame."""
    fake_model = _FakeModel(_two_class_result())
    det = BananaDetector(infer_size=_FRAME_SIZE, model=fake_model)
    frame = _blank_frame()

    det.detect(frame)
    det.detect(frame)
    det.detect(frame)

    assert fake_model.call_count == 3


def test_rescales_bbox_and_mask_from_infer_size_to_original_frame_size():
    """Fix round 1, Finding 1 (reviewer): el resto de este archivo usa
    `infer_size == ancho == alto del frame`, lo que hace `scale_x == scale_y
    == 1.0` -- todo el camino de reescalado (bbox y mascara) es un no-op
    bajo esos tests, y un scale_x/scale_y invertido, un orden de argumentos
    invertido en `cv2.resize`, o un redondeo fuera de lugar pasarian
    desapercibidos. Aqui el frame es NO CUADRADO (80 alto x 120 ancho) con
    `infer_size=40`, asi que scale_x=3.0 != scale_y=2.0: cualquier
    intercambio de ejes cambia el resultado, y se verifica contra
    coordenadas concretas (no solo "cambio de tamano")."""
    original_h, original_w = 80, 120
    infer_size = 40

    mask_infer = np.zeros((infer_size, infer_size), dtype=np.uint8)
    mask_infer[6:14, 4:10] = 1  # filas 6..13, columnas 4..9 en espacio de inferencia

    boxes = _FakeBoxes(cls=[46], conf=[0.9], ids=[1], xyxy=[[4, 6, 10, 14]])
    masks = _FakeMasks([mask_infer])
    fake_model = _FakeModel(_FakeResult(boxes, masks))
    det = BananaDetector(infer_size=infer_size, model=fake_model)
    frame = np.zeros((original_h, original_w, 3), dtype=np.uint8)

    out = det.detect(frame)

    assert len(out) == 1
    inst = out[0]
    # x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y = 4*3, 6*2, 10*3, 14*2
    assert inst.bbox == (12, 12, 30, 28)
    assert inst.mask.shape == (original_h, original_w)
    assert bool(inst.mask[20, 21]) is True  # dentro de la region reescalada
    assert bool(inst.mask[70, 100]) is False  # claramente afuera de ella


def test_missing_track_ids_get_unique_synthetic_ids_not_a_shared_sentinel():
    """Fix round 1, Finding 2 (coordinador, ruling): cuando el tracker
    todavia no produjo ids (`boxes.id is None`), sintetizar un unico
    centinela compartido para todas las detecciones de ese frame fabrica la
    colision que `CentroidReassigner.assign` (Ruling 4) existe para evitar
    -- y aqui no hay reasignacion previa que pueda repararla despues,
    porque las instancias YA llegan indistinguibles. Dos bananos detectados
    sin ids deben recibir dos track_id DISTINTOS (y reconociblemente
    sinteticos: negativos)."""
    size = _FRAME_SIZE
    mask_a = np.zeros((size, size), dtype=np.uint8)
    mask_a[5:10, 5:10] = 1
    mask_b = np.zeros((size, size), dtype=np.uint8)
    mask_b[40:45, 40:45] = 1
    boxes = _FakeBoxes(cls=[46, 46], conf=[0.9, 0.8], ids=None, xyxy=[[5, 5, 10, 10], [40, 40, 45, 45]])
    masks = _FakeMasks([mask_a, mask_b])
    fake_model = _FakeModel(_FakeResult(boxes, masks))
    det = BananaDetector(infer_size=size, model=fake_model)

    out = det.detect(_blank_frame(size))

    assert len(out) == 2
    ids = [i.track_id for i in out]
    assert len(set(ids)) == 2  # ningun id compartido entre las dos detecciones
    assert all(tid < 0 for tid in ids)  # reconociblemente sinteticos, no del tracker


def test_zero_detections_returns_an_empty_result_without_raising():
    """Fix round 1, Finding 3 (coordinador): ultralytics devuelve
    `boxes=None` y `masks=None` cuando no detecto nada en el frame -- el
    caso MAS ORDINARIO del sistema (un frame sin fruta a la vista), no un
    caso raro. Sin guarda esto revienta con `AttributeError` en
    `masks.data`. La Task 14 depende de que este camino devuelva `[]` en
    vez de propagar la excepcion."""
    fake_model = _FakeModel(_FakeResult(boxes=None, masks=None))
    det = BananaDetector(infer_size=_FRAME_SIZE, model=fake_model)

    out = det.detect(_blank_frame())

    assert out == []


# ---------------------------------------------------------------------------
# CentroidReassigner
# ---------------------------------------------------------------------------


def _instance(track_id, cx, cy, size=_FRAME_SIZE, half=3, class_id=46, confidence=0.9):
    mask = np.zeros((size, size), dtype=bool)
    mask[cy - half : cy + half, cx - half : cx + half] = True
    bbox = (cx - half, cy - half, cx + half, cy + half)
    return Instance(track_id=track_id, class_id=class_id, mask=mask, bbox=bbox, confidence=confidence)


def test_centroid_reassigner_recovers_a_dropped_track_id():
    # El objeto esta quieto; el tracker le asigno un id nuevo (7) tras
    # perderlo un frame, pero su centroide practicamente no cambio.
    previous = [_instance(track_id=2, cx=20, cy=20)]
    current = [_instance(track_id=7, cx=21, cy=19)]

    out = CentroidReassigner(30.0).assign(current, previous)

    assert out[0].track_id == 2


def test_centroid_reassigner_keeps_distinct_ids_for_distant_objects():
    """Los ids de entrada son NUEVOS (5, 6) -- no 1 y 2 -- y su orden en la
    lista esta barajado respecto a `previous`. Una implementacion que
    asignara por indice de lista (`current[i] <- previous[i].track_id`) en
    vez de por distancia de centroide fallaria aqui: le daria id=1 (de
    previous[0]) a current[0], que en realidad esta junto a previous[1]."""
    previous = [
        _instance(track_id=1, cx=10, cy=10),
        _instance(track_id=2, cx=50, cy=50),
    ]
    current = [
        _instance(track_id=6, cx=51, cy=49),  # cerca de previous[1] (id 2)
        _instance(track_id=5, cx=9, cy=11),  # cerca de previous[0] (id 1)
    ]

    out = CentroidReassigner(30.0).assign(current, previous)

    assert out[0].track_id == 2
    assert out[1].track_id == 1


def test_centroid_reassigner_does_not_let_two_instances_share_a_track_id():
    """Caso realista (Ruling 4): dos bananos quietos, muy cerca entre si,
    pierden su id el mismo frame. Para `current[1]`, el previo mas cercano
    en distancia pura ES el mismo previo (id 1) que ya reclamo `current[0]`
    (que esta exactamente encima de el). Una asignacion "vecino mas
    cercano" sin exclusion uno a uno le daria id=1 a AMBAS instancias,
    colapsando dos frutos distintos en un solo track_id."""
    previous = [
        _instance(track_id=1, cx=10, cy=10),
        _instance(track_id=2, cx=13, cy=10),
    ]
    current = [
        _instance(track_id=10, cx=10, cy=10),  # distancia 0 a previous[0]
        _instance(track_id=11, cx=11, cy=10),  # distancia 1 a previous[0], 2 a previous[1]
    ]

    out = CentroidReassigner(30.0).assign(current, previous)

    ids = [i.track_id for i in out]
    assert len(set(ids)) == 2
    assert set(ids) == {1, 2}


def test_centroid_reassigner_leaves_distant_instances_untouched():
    """Si no hay ningun previo dentro de `max_distance_px`, la instancia
    conserva su propio track_id (no hay nada que reasignar)."""
    previous = [_instance(track_id=1, cx=10, cy=10)]
    current = [_instance(track_id=9, cx=55, cy=55)]

    out = CentroidReassigner(5.0).assign(current, previous)

    assert out[0].track_id == 9


def test_centroid_reassigner_with_no_previous_instances_is_a_no_op():
    current = [_instance(track_id=3, cx=10, cy=10)]

    out = CentroidReassigner(30.0).assign(current, previous=[])

    assert out[0].track_id == 3


# ---------------------------------------------------------------------------
# masks_are_adjacent
# ---------------------------------------------------------------------------


def _block_mask(size, row0, col0, h=5, w=5):
    mask = np.zeros((size, size), dtype=bool)
    mask[row0 : row0 + h, col0 : col0 + w] = True
    return mask


def test_adjacent_masks_are_detected():
    size = 60
    mask_a = _block_mask(size, row0=5, col0=0)  # columnas 0..4

    mask_touching = _block_mask(size, row0=5, col0=5)  # columnas 5..9, D=1
    assert masks_are_adjacent(mask_a, mask_touching) is True

    mask_far = _block_mask(size, row0=5, col0=50)  # muy lejos
    assert masks_are_adjacent(mask_a, mask_far) is False


def test_adjacent_masks_threshold_is_exact_at_dilate_px():
    """Caso de Review Focus (Ruling 3d): un hueco justo DENTRO de
    `dilate_px` y uno justo FUERA, para que un off-by-one en el radio de
    dilatacion (p.ej. `<=` vs `<`, o un kernel de lado `2*dilate_px` en vez
    de `2*dilate_px + 1`) se note."""
    size = 60
    mask_a = _block_mask(size, row0=5, col0=0)  # borde derecho en columna 4

    # D = dilate_px exactamente (col0 - 4 == 3): debe ser adyacente.
    mask_at_threshold = _block_mask(size, row0=5, col0=7)
    assert masks_are_adjacent(mask_a, mask_at_threshold, dilate_px=3) is True

    # D = dilate_px + 1 (col0 - 4 == 4): un pixel mas alla, ya no.
    mask_past_threshold = _block_mask(size, row0=5, col0=8)
    assert masks_are_adjacent(mask_a, mask_past_threshold, dilate_px=3) is False


def test_adjacent_masks_overlapping_is_trivially_true():
    size = 20
    mask_a = _block_mask(size, row0=0, col0=0, h=10, w=10)
    mask_b = _block_mask(size, row0=5, col0=5, h=10, w=10)
    assert masks_are_adjacent(mask_a, mask_b) is True

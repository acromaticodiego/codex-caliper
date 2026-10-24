"""Deteccion de bananos con YOLOv8-seg: tracking e identidad persistente (Task 12).

Arquitectura del proyecto: "deteccion neuronal, medicion clasica". YOLO dice
QUE hay y DONDE, y mantiene identidad entre frames (``track_id``); toda la
geometria (longitud, calibre, volumen) se mide despues con CV clasica sobre
el lienzo rectificado por homografia (Tasks 6-9). Este modulo es la MITAD
de deteccion -- el unico lugar del proyecto donde aparece YOLO.

Tres piezas, cada una resolviendo un problema real de produccion:

1. ``BananaDetector`` envuelve ``model.track(...)`` (ByteTrack via
   ultralytics), filtra la clase "banana" (id 46 de COCO) y reescala
   mascaras/cajas del tamano de inferencia al tamano del frame original.
   ``every_n_frames`` es el respaldo para equipos sin GPU: los bananos del
   rig estan quietos, asi que correr inferencia cada N frames y reutilizar
   la ultima deteccion en los frames intermedios no pierde nada util.

2. ``CentroidReassigner`` corrige el fallo mas comun de ByteTrack en este
   rig: tras una oclusion breve (una mano cruzando el campo de vision, dos
   bananos que se tapan un instante), el tracker le asigna un id NUEVO a un
   objeto que en realidad no se movio. Como el montaje es una vista cenital
   fija, la cercania de centroide entre frames consecutivos es una señal
   confiable para recuperar la identidad perdida.

3. ``masks_are_adjacent`` detecta mascaras fusionadas o pegadas -- el
   requisito de "nunca reportar una medida sin calibracion valida" tambien
   aplica aqui: si dos frutos distintos producen mascaras que se tocan (o
   YOLO las fusiono en una sola instancia que en realidad son dos), el
   pipeline (Task 14) necesita poder detectarlo para negarse a medir en vez
   de reportar un numero fabricado a partir de una mascara rota.

``ultralytics`` NO esta instalado en este entorno de desarrollo (son ~2.5 GB
con ``torch``, y esta tarea no los necesita: los tests inyectan un doble via
el parametro ``model``). El import vive DENTRO de la rama que construye el
modelo real, nunca a nivel de modulo, para que todo lo demas siga
funcionando sin ``ultralytics`` instalado. La Task 20 es quien instala los
pesos reales y ejercita esa rama por primera vez.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

#: Clase "banana" del dataset COCO -- la unica clase de interes del proyecto.
YOLO_BANANA_CLASS_ID = 46

#: Pesos por defecto cuando no se inyecta un modelo (ver ``BananaDetector``).
DEFAULT_MODEL_PATH = "yolov8n-seg.pt"

#: Lado (en pixeles) del frame reescalado antes de la inferencia.
DEFAULT_INFER_SIZE = 640


@dataclasses.dataclass
class Instance:
    """Una deteccion individual, ya en coordenadas del frame original.

    ``mask`` es booleana y ya fue reescalada del tamano de inferencia al
    tamano del frame de entrada -- quien consume ``Instance`` (Task 13,
    Task 14) nunca ve el redimensionado interno de YOLO. ``bbox`` es
    ``(x1, y1, x2, y2)`` en esas mismas coordenadas.

    No se define un ``__eq__`` especial: el ``__eq__`` por defecto de
    ``dataclass`` compara una tupla de todos los campos, y comparar dos
    arrays de numpy con ``==`` no da un booleano unico (revienta con
    "truth value of an array is ambiguous" en cuanto ``mask`` tiene mas de
    un elemento). Comparar ``Instance`` campo por campo, y ``mask``
    especificamente con ``np.array_equal``.
    """

    track_id: int
    class_id: int
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    confidence: float


class BananaDetector:
    """Envoltura de YOLOv8-seg: tracking persistente, filtro de clase y salto de frames."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        infer_size: int = DEFAULT_INFER_SIZE,
        every_n_frames: int = 1,
        model=None,
    ) -> None:
        """``model`` permite inyectar un doble de pruebas (o unos pesos ya
        cargados); si es ``None`` se construye ``ultralytics.YOLO(model_path)``
        -- import perezoso, ver docstring del modulo."""
        self.model_path = model_path
        self.infer_size = infer_size
        self.every_n_frames = every_n_frames
        if model is not None:
            self.model = model
        else:
            from ultralytics import YOLO  # import perezoso: ultralytics no esta instalado (Task 20)

            self.model = YOLO(model_path)
        self._frame_index = 0
        self._last_instances: list[Instance] = []

    def detect(self, frame: np.ndarray) -> list[Instance]:
        """Detecta y rastrea bananos en ``frame``, o reutiliza la ultima deteccion.

        Reescala ``frame`` a ``infer_size x infer_size`` (cuadrado -- no
        preserva el aspect ratio; simplificacion aceptable porque el rig es
        una vista cenital fija de encuadre conocido, ajustable en la Task 20
        si hiciera falta) y llama a
        ``model.track(frame_resized, persist=True, classes=[YOLO_BANANA_CLASS_ID])``.
        Mascaras y cajas se reescalan de vuelta al tamano de ``frame`` antes
        de devolverlas.

        El filtro de clase se aplica DE NUEVO sobre el resultado del modelo,
        sin confiar en que ``classes=...`` por si solo baste (un doble de
        pruebas, o un modelo real mal configurado, puede devolver otras
        clases igual).

        Con ``every_n_frames > 1``, solo se corre inferencia cuando el
        contador de frames es multiplo de ``every_n_frames``; en los demas
        frames se devuelve la ultima lista de instancias calculada, tal
        cual (mismos ``track_id``, mismas mascaras).
        """
        should_infer = (self._frame_index % self.every_n_frames) == 0
        self._frame_index += 1
        if not should_infer:
            return self._last_instances

        original_h, original_w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (self.infer_size, self.infer_size))
        results = self.model.track(frame_resized, persist=True, classes=[YOLO_BANANA_CLASS_ID])
        instances = self._parse_result(results[0], original_w, original_h)
        self._last_instances = instances
        return instances

    def _parse_result(self, result, original_w: int, original_h: int) -> list[Instance]:
        """Convierte un resultado crudo de ``model.track`` en ``list[Instance]``,
        filtrando la clase banana y reescalando cajas/mascaras a (``original_w``,
        ``original_h``).

        ``boxes`` y ``masks`` llegan ambos en ``None`` cuando ultralytics no
        detecto nada en el frame -- el caso mas ORDINARIO del sistema (un
        frame sin fruta a la vista), no uno raro. Sin esta guarda,
        ``masks.data`` revienta con ``AttributeError`` en el frame mas
        comun que el rig va a ver. La respuesta correcta a "no hay nada" es
        la lista vacia, no una excepcion.
        """
        boxes = result.boxes
        masks = result.masks
        if boxes is None or masks is None:
            return []

        class_ids = np.asarray(boxes.cls)
        confidences = np.asarray(boxes.conf)
        xyxy = np.asarray(boxes.xyxy)
        mask_data = np.asarray(masks.data)

        raw_track_ids = boxes.id
        if raw_track_ids is None:
            # El tracker todavia no produjo ids (pasa, entre otros casos, en
            # el primer frame). Un unico centinela compartido (p.ej. -1 para
            # todas) FABRICARIA la colision que CentroidReassigner.assign
            # existe para evitar (Ruling 4): dos o tres bananos detectados
            # en este frame compartirian un solo track_id, y
            # MeasurementAggregator.push (Task 11) los poolearia como si
            # fueran un solo fruto. A diferencia de una colision que llega
            # DESPUES de la reasignacion por centroide, esta no se puede
            # reparar aguas abajo: dos instancias que ya entran con el mismo
            # id son indistinguibles para `assign`. La spec (S7, etapa 4)
            # cuenta con la reasociacion por centroide para resolver
            # precisamente esto -- y esa reasociacion solo puede funcionar
            # si las instancias LLEGAN distinguibles. Por eso cada instancia
            # sin id de tracker recibe un negativo UNICO (-1, -2, -3, ...);
            # ademas de evitar la colision, el signo negativo deja legible
            # aguas abajo que ese id no vino del tracker.
            track_ids = -1 - np.arange(len(class_ids))
        else:
            track_ids = np.asarray(raw_track_ids)

        scale_x = original_w / self.infer_size
        scale_y = original_h / self.infer_size

        instances: list[Instance] = []
        for idx in range(len(class_ids)):
            class_id = int(class_ids[idx])
            if class_id != YOLO_BANANA_CLASS_ID:
                continue

            x1, y1, x2, y2 = xyxy[idx]
            bbox = (
                int(round(x1 * scale_x)),
                int(round(y1 * scale_y)),
                int(round(x2 * scale_x)),
                int(round(y2 * scale_y)),
            )
            mask_resized = cv2.resize(
                mask_data[idx].astype(np.uint8),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            instances.append(
                Instance(
                    track_id=int(track_ids[idx]),
                    class_id=class_id,
                    mask=mask_resized,
                    bbox=bbox,
                    confidence=float(confidences[idx]),
                )
            )
        return instances


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    """Centroide ``(x, y)`` en pixeles de los pixeles verdaderos de ``mask``.

    Devuelve ``(0.0, 0.0)`` para una mascara vacia en vez de lanzar: no hay
    una posicion correcta que inventar, pero tampoco vale la pena que una
    mascara vacia tumbe la reasignacion de las demas instancias.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.array([0.0, 0.0])
    return np.array([float(xs.mean()), float(ys.mean())])


class CentroidReassigner:
    """Recupera ``track_id`` perdidos por el tracker, por cercania de centroide.

    El tracker interno de YOLO (ByteTrack) ocasionalmente le asigna un id
    nuevo a un objeto que no se movio -- tipicamente tras una oclusion
    breve. Esta clase compara el centroide de cada instancia actual contra
    el centroide de cada instancia de ``previous``: si el par mas cercano
    esta a menos de ``max_distance_px``, la instancia actual hereda ese
    ``track_id`` viejo en vez del que le dio el tracker.

    La asignacion es UNO A UNO: ningun ``track_id`` de ``previous`` puede
    heredarse dos veces en la misma llamada. Sin esa exclusion, dos bananos
    quietos que pierden su id en el mismo frame terminarian con el MISMO
    ``track_id`` -- lo que colapsaria silenciosamente sus buffers
    temporales en ``MeasurementAggregator.push`` (Task 11), pooleando dos
    frutos distintos como si fueran uno solo, sin lanzar ningun error.
    """

    def __init__(self, max_distance_px: float) -> None:
        self.max_distance_px = max_distance_px

    def assign(self, instances: list[Instance], previous: list[Instance]) -> list[Instance]:
        """Devuelve ``instances`` con los ``track_id`` corregidos contra ``previous``.

        Candidatos: todo par (instancia actual, instancia previa) cuya
        distancia de centroide sea ESTRICTAMENTE menor que
        ``max_distance_px``. Se procesan en orden de distancia creciente
        (el par mas cercano global primero) y cada instancia actual, y cada
        ``track_id`` previo, se consume como maximo una vez -- de ahi la
        exclusion uno a uno.
        """
        if not previous:
            return list(instances)

        current_centroids = [_mask_centroid(inst.mask) for inst in instances]
        previous_centroids = [_mask_centroid(prev.mask) for prev in previous]

        candidates = []
        for i, current_centroid in enumerate(current_centroids):
            for j, previous_centroid in enumerate(previous_centroids):
                distance = float(np.linalg.norm(current_centroid - previous_centroid))
                if distance < self.max_distance_px:
                    candidates.append((distance, i, j))
        candidates.sort(key=lambda item: item[0])

        reassigned_id_by_index: dict[int, int] = {}
        used_previous_indices: set[int] = set()
        for _distance, i, j in candidates:
            if i in reassigned_id_by_index or j in used_previous_indices:
                continue
            reassigned_id_by_index[i] = previous[j].track_id
            used_previous_indices.add(j)

        out = []
        for i, inst in enumerate(instances):
            new_track_id = reassigned_id_by_index.get(i, inst.track_id)
            out.append(dataclasses.replace(inst, track_id=new_track_id))
        return out


def masks_are_adjacent(a: np.ndarray, b: np.ndarray, dilate_px: int = 3) -> bool:
    """``True`` si ``a`` y ``b`` se tocan, o sus pixeles mas cercanos estan
    separados por a lo sumo ``dilate_px`` pixeles (distancia de Chebyshev:
    el kernel de dilatacion es cuadrado, no un disco).

    Implementacion: dilata ``a`` con un kernel cuadrado de lado
    ``2 * dilate_px + 1`` y comprueba si el resultado intersecta a ``b`` sin
    dilatar. Esto fija el umbral exactamente en ``dilate_px``: dos mascaras
    cuyo par de pixeles mas cercanos esta a distancia ``dilate_px`` cuentan
    como adyacentes; a ``dilate_px + 1`` ya no.

    Nota de calibracion para quien use el valor por defecto (Task 14, que
    decide el valor operativo): con el ``canvas_mm_per_px`` por defecto de
    ``AppConfig`` (0.2 mm/px), ``dilate_px=3`` equivale a 0.6 mm de
    separacion maxima -- posiblemente demasiado estricto para separar dos
    bananos reales que se tocan apenas. La firma queda en pixeles a
    proposito (esta funcion es una operacion de espacio de mascara; no
    conoce la escala del lienzo), pero el valor operativo real debe
    escogerse con esa cifra en mente, no a ciegas.
    """
    kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=np.uint8)
    dilated_a = cv2.dilate(a.astype(np.uint8), kernel)
    return bool(np.any((dilated_a > 0) & (np.asarray(b) > 0)))

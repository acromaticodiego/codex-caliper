"""Pruebas de Task 14: orquestacion del pipeline por cuadro.

Esta es la primera tarea donde TODOS los modulos se conectan, asi que las
pruebas de aqui no verifican una funcion aislada: verifican el ORDEN y las
UNIDADES de la cadena completa. Dos defectos del Task Brief se corrigen aqui
(ver el reporte de la tarea):

1. El brief comparaba la longitud agregada contra ``gt.outer_length_mm`` con
   la correccion de paralaje ACTIVA. Sobre una escena sintetica -- que es
   PLANA, altura cero -- esa correccion no es una correccion sino error puro:
   encoge toda magnitud por ``2*H_ef / (2*H_ef + calibre_crudo)``. Por eso
   ``Pipeline`` tiene ``correct_parallax`` (que la spec S7 etapa 7 y S13
   piden de forma independiente, para publicar el antes/despues) y hay DOS
   pruebas end-to-end: una con la correccion apagada contra la verdad cruda,
   y otra con la correccion encendida que fija el factor EXACTO de forma
   cerrada -- no "aproximadamente 6%", que ya se demostro (Task 9) que no
   discrimina.
2. El brief nunca expone ``PlaneCalibration.state``: con 3 marcadores la
   spec (S7 etapa 2) exige seguir midiendo Y avisarlo. Hay una prueba para
   eso.

Las dos pruebas del brief que asertan sobre ``results == []`` son vacuas por
si solas (un pipeline que no mide NUNCA las pasa igual), asi que cada una
lleva adosada una asercion positiva sobre el MISMO pipeline: la misma
instancia tiene que medir la escena sana.
"""

from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np
import pytest

from codex_caliper.aruco_tools import CalibrationState, detect_markers
from codex_caliper.codex import CODEX_MIN_CALIPER_MM, CODEX_MIN_LENGTH_MM
from codex_caliper.config import AppConfig
from codex_caliper.detection import BananaDetector, YOLO_BANANA_CLASS_ID
from codex_caliper.geometry.measures import measure_mask
from codex_caliper.geometry.outline import smoothed_arc_length_mm
from codex_caliper.homography import DegenerateMarkerLayout, canvas_size_for
from codex_caliper.intrinsics import Intrinsics
from codex_caliper.pipeline import (
    ADJACENCY_DILATE_PX,
    CLOSING_BRIDGE_PX,
    Pipeline,
)
from codex_caliper.segmentation import refine_mask
from tests.fixtures.synthetic import make_banana, make_banana_scene

_MM_PER_PX = 0.2
_MARKER_THICKNESS_MM = 3.0

# Umbral de saturacion con el que el doble de YOLO separa fruta de fondo.
# El fondo blanco y la sombra gris neutra de la escena estan en saturacion 0;
# el amarillo de la fruta, en 255.
_FAKE_SATURATION_KEY = 80

# Minimo de pixeles para que un componente conexo cuente como una instancia
# (equivale al umbral de confianza de un detector real: descarta motas).
_FAKE_MIN_INSTANCE_PX = 200

# Densidad de la pulpa de banano, g/cm3 -- solo para fijar la aritmetica del
# peso, no es una medicion de esta tarea.
_DENSITY_G_PER_CM3 = 0.94

# ---------------------------------------------------------------------------
# Geometria de los bananos sinteticos
# ---------------------------------------------------------------------------

# Fixture de punta afilada heredado de Task 7 (``tests/test_outline.py``): es
# la UNICA geometria del fixture sobre la que ``GroundTruth.outer_length_mm``
# -- el riel puro ``(R + w/2)*theta``, que EXCLUYE las tapas -- es comparable
# contra una cadena de contorno medida, que si incluye material de tapa. Con
# un calibre realista (w=36) el error se queda en +15..+23% por construccion,
# no por el algoritmo. ``pedicel_mm=0``: la verdad analitica describe la
# capsula pura.
_SHARP_BANANA = {
    "radius_mm": 220.0,
    "arc_rad": 4.0,
    "width_mm": 12.0,
    "blossom_taper_mm": 11.9,
    "pedicel_mm": 0.0,
}
# Marcadores en las cuatro esquinas de un rectangulo que encierra al banano
# con holgura (la fruta ocupa [20, 452] x [20, 343.5] mm en la escena).
_SHARP_LAYOUT = {
    1: (-10.0, -10.0),
    2: (470.0, -10.0),
    3: (470.0, 360.0),
    4: (-10.0, 360.0),
}

# Banano chico y grueso, para las pruebas que no comparan contra la verdad
# analitica (multiplicidad, peso, adyacencia, acumulacion temporal): 189 mm
# de riel exterior y 30 mm de calibre cumplen los dos minimos del CODEX.
_PLUMP_BANANA = {"radius_mm": 120.0, "arc_rad": 1.4, "width_mm": 30.0}
_PLUMP_LAYOUT = {
    1: (-10.0, -10.0),
    2: (490.0, -10.0),
    3: (490.0, 110.0),
    4: (-10.0, 110.0),
}

# Banano cuyo calibre CRUDO queda justo ENCIMA del minimo CODEX (27 mm) y
# cuyo calibre CORREGIDO queda justo por DEBAJO: el unico caso que distingue
# "el veredicto se evalua sobre el agregado corregido" de "se evalua sobre la
# medida cruda".
_BORDERLINE_BANANA = {"radius_mm": 120.0, "arc_rad": 1.4, "width_mm": 27.5}
_BORDERLINE_LAYOUT = {
    1: (-10.0, -10.0),
    2: (250.0, -10.0),
    3: (250.0, 110.0),
    4: (-10.0, 110.0),
}

# Mismo banano que ``_BORDERLINE_BANANA``, con los marcadores APRETADOS de
# modo que el lienzo calibrado corte la fruta por el borde izquierdo.
_CLIPPED_LAYOUT = {
    1: (40.0, 20.0),
    2: (210.0, 20.0),
    3: (210.0, 90.0),
    4: (40.0, 90.0),
}


# ---------------------------------------------------------------------------
# Doble de ultralytics y detector que registra lo que recibe
# ---------------------------------------------------------------------------


class _FakeBoxes:
    def __init__(self, cls, conf, xyxy, ids) -> None:
        self.cls = cls
        self.conf = conf
        self.xyxy = xyxy
        self.id = ids


class _FakeMasks:
    def __init__(self, data) -> None:
        self.data = data


class _FakeResult:
    def __init__(self, boxes, masks) -> None:
        self.boxes = boxes
        self.masks = masks


class _FakeYoloModel:
    """Doble de ``ultralytics.YOLO`` que segmenta por saturacion.

    ``ultralytics`` y ``torch`` no estan instalados (ni deben estarlo: ~2.5 GB
    que esta tarea no necesita), y ``BananaDetector`` existe con un parametro
    ``model=`` justamente para inyectar esto.

    Es deliberadamente BASADO EN CONTENIDO -- deriva las instancias de los
    pixeles del cuadro que recibe, no de una lista de posiciones conocidas --
    para que las pruebas end-to-end dependan de verdad de que el pipeline le
    entregue el LIENZO RECTIFICADO: alimentado con el cuadro crudo, el doble
    devuelve mascaras en las coordenadas equivocadas y la medida se va.

    Devuelve ``boxes``/``masks`` en ``None`` cuando no hay nada, igual que
    ultralytics en un cuadro vacio.

    ``id_offset`` (mutable, default 0) se suma a cada id crudo. Por defecto
    los ids salen deterministas y estables, derivados del orden de barrido
    de ``connectedComponents`` sobre una escena quieta -- pero eso es
    precisamente lo que una prueba sobre perdida de identidad NO debe dejar
    fijo: un tracker real (ByteTrack) no promete el mismo id crudo despues
    de una oclusion, y una prueba cuyo doble si lo promete pasaria igual con
    un pipeline que tira la memoria de centroides en el cuadro vacio (los
    ids crudos coincidirian con los de antes por construccion del doble, no
    porque ``CentroidReassigner`` los haya recuperado). Subir ``id_offset``
    entre llamadas simula esa reidentificacion con ids nuevos.
    """

    def __init__(self, id_offset: int = 0) -> None:
        self.id_offset = id_offset

    def track(self, frame, persist=True, classes=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        binary = (hsv[..., 1] > _FAKE_SATURATION_KEY).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary)

        masks, boxes, ids = [], [], []
        for label in range(1, num_labels):
            component = labels == label
            if int(np.count_nonzero(component)) < _FAKE_MIN_INSTANCE_PX:
                continue
            ys, xs = np.nonzero(component)
            masks.append(component.astype(np.float32))
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
            # Los labels de ``connectedComponents`` salen en orden de barrido
            # (arriba-izquierda primero), asi que SIN ``id_offset`` los ids
            # son deterministas y estables entre cuadros sobre una escena
            # quieta -- ver la nota de ``id_offset`` arriba sobre por que
            # eso no basta para probar perdida de identidad.
            ids.append(len(ids) + 1 + self.id_offset)

        if not masks:
            return [_FakeResult(None, None)]
        return [
            _FakeResult(
                _FakeBoxes(
                    np.full(len(ids), YOLO_BANANA_CLASS_ID),
                    np.full(len(ids), 0.9),
                    np.asarray(boxes, dtype=np.float64),
                    np.asarray(ids, dtype=np.int64),
                ),
                _FakeMasks(np.stack(masks)),
            )
        ]


class _RecordingDetector:
    """Envuelve un ``BananaDetector`` y guarda cada cuadro que le entregan.

    El contrato que protege: ``Instance.mask`` sale con la forma del cuadro
    que se le pasa a ``detect()``, y Task 13 (``refine_mask``) depende de que
    esa forma sea la del LIENZO RECTIFICADO. Registrar el cuadro es la unica
    forma directa de verificar sobre que lo corrio el pipeline.
    """

    def __init__(self, inner: BananaDetector) -> None:
        self.inner = inner
        self.frames: list[np.ndarray] = []

    def detect(self, frame: np.ndarray):
        self.frames.append(frame)
        return self.inner.detect(frame)


# ---------------------------------------------------------------------------
# Helpers de montaje
# ---------------------------------------------------------------------------


def _config_for(layout: dict[int, tuple[float, float]]) -> AppConfig:
    return AppConfig(
        marker_thickness_mm=_MARKER_THICKNESS_MM,
        canvas_mm_per_px=_MM_PER_PX,
        marker_layout=layout,
    )


def _identity_intrinsics(scene: np.ndarray) -> Intrinsics:
    """Intrinsecos sin distorsion a la resolucion exacta de ``scene``.

    ``undistort`` exige que la resolucion coincida (``require_compatible``), y
    con ``dist_coeffs`` en cero el remapeo es la identidad: la escena
    sintetica ya es una vista cenital sin distorsion de lente.
    """
    height, width = scene.shape[:2]
    return Intrinsics(
        camera_matrix=np.array(
            [[1000.0, 0.0, width / 2.0], [0.0, 1000.0, height / 2.0], [0.0, 0.0, 1.0]]
        ),
        dist_coeffs=np.zeros(5),
        rms_px=0.1,
        image_size=(width, height),
    )


def _build(
    scene: np.ndarray,
    cfg: AppConfig,
    *,
    density_g_per_cm3: float | None = None,
    correct_parallax: bool = True,
) -> tuple[Pipeline, _RecordingDetector]:
    """Arma un ``Pipeline`` sobre ``scene`` y devuelve tambien su detector.

    ``infer_size`` se fija al lado mayor del lienzo en vez de los 640 px de
    produccion: el doble de YOLO separa la fruta por COLOR, asi que reducir
    el cuadro a 640 px lo haria perder un hueco de 1.6 mm entre dos frutos
    (la prueba de adyacencia). Es una propiedad del doble, no del pipeline --
    ``BananaDetector`` reescala mascaras y cajas de vuelta al tamano del
    cuadro en cualquier caso, cosa que Task 12 ya verifica por su cuenta.
    """
    detector = _RecordingDetector(
        BananaDetector(infer_size=max(canvas_size_for(cfg)), model=_FakeYoloModel())
    )
    pipeline = Pipeline(
        cfg,
        _identity_intrinsics(scene),
        detector,
        density_g_per_cm3=density_g_per_cm3,
        correct_parallax=correct_parallax,
    )
    return pipeline, detector


def _hide_markers(scene: np.ndarray, cfg: AppConfig, ids: list[int]) -> np.ndarray:
    """Copia de ``scene`` con los marcadores ``ids`` pintados de blanco.

    Tapar marcadores sobre la MISMA escena (en vez de generar otra con menos
    marcadores) mantiene la resolucion identica, asi que el mismo
    ``Pipeline`` -- mismos intrinsecos, mismo lienzo -- puede procesar la
    escena sana y la degradada. Eso es lo que permite que cada prueba de
    "no hay resultados" lleve adosada la asercion positiva que la vuelve no
    vacua.
    """
    out = scene.copy()
    corners_by_id = {d.id: d.corners for d in detect_markers(scene, cfg)}
    for marker_id in ids:
        corners = corners_by_id[marker_id]
        x0 = max(0, int(math.floor(corners[:, 0].min())) - 4)
        y0 = max(0, int(math.floor(corners[:, 1].min())) - 4)
        x1 = int(math.ceil(corners[:, 0].max())) + 5
        y1 = int(math.ceil(corners[:, 1].max())) + 5
        out[y0:y1, x0:x1] = 255
    return out


def _hide_fruit(scene: np.ndarray) -> np.ndarray:
    """Copia de ``scene`` sin fruta: todo pixel saturado pasa a blanco.

    Deja intactos los marcadores (sin saturacion) y la sombra gris neutra, de
    modo que la calibracion sigue siendo valida y lo unico que desaparece son
    las detecciones -- un cuadro perdido, no un cuadro sin calibrar.
    """
    out = scene.copy()
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    out[hsv[..., 1] > _FAKE_SATURATION_KEY] = 255
    return out


def _parallax_factor(effective_height_mm: float, raw_caliper_mm: float) -> float:
    """Forma cerrada de Task 9, escrita a mano aqui a proposito.

    Reimplementar la expresion en la prueba -- en vez de llamar a
    ``parallax.scale_factor`` o ``correct_measures`` -- es lo que hace que la
    prueba detecte una inversion del factor (multiplicar donde habia que
    dividir): si la prueba reusara la funcion del codigo, el mismo error
    aparecia en los dos lados de la igualdad y se cancelaria.
    """
    return 2.0 * effective_height_mm / (2.0 * effective_height_mm + raw_caliper_mm)


# ---------------------------------------------------------------------------
# Escenas (module-scoped: generarlas cuesta 0.2-0.5 s cada una)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sharp_scene() -> np.ndarray:
    return make_banana_scene([_SHARP_BANANA], _MM_PER_PX, markers=_SHARP_LAYOUT)


@pytest.fixture(scope="module")
def sharp_truth():
    _mask, gt = make_banana(mm_per_px=_MM_PER_PX, **_SHARP_BANANA)
    return gt


@pytest.fixture(scope="module")
def sharp_mask() -> np.ndarray:
    """La mascara PRISTINA de ``_SHARP_BANANA``: sin escena, sin marcadores,
    sin rectificacion ni ``refine_mask`` -- la unica entrada que aisla el
    sesgo de ``measure_mask`` (fixture de Task 7 + recorte de pedicelo de
    Task 8) del sesgo que aporta la cadena ArUco/homografia/deteccion. Ver
    ``test_end_to_end_error_decomposes_into_pristine_and_warp_chain_bias``.
    """
    mask, _gt = make_banana(mm_per_px=_MM_PER_PX, **_SHARP_BANANA)
    return mask


@pytest.fixture(scope="module")
def sharp_cfg() -> AppConfig:
    return _config_for(_SHARP_LAYOUT)


@pytest.fixture(scope="module")
def plump_scene() -> np.ndarray:
    """Dos bananos separados por los 30 mm por defecto del fixture."""
    return make_banana_scene([_PLUMP_BANANA, _PLUMP_BANANA], _MM_PER_PX, markers=_PLUMP_LAYOUT)


@pytest.fixture(scope="module")
def plump_cfg() -> AppConfig:
    return _config_for(_PLUMP_LAYOUT)


@pytest.fixture(scope="module")
def touching_scene_and_cfg() -> tuple[np.ndarray, AppConfig]:
    """Dos bananos separados exactamente por la distancia de puenteo medida.

    ``refine_mask`` cierra con un kernel de 9x9 px, que puentea un hueco de
    hasta 8 px (1.6 mm a 0.2 mm/px) -- medido, ver ``CLOSING_BRIDGE_PX``. A
    esa separacion la segmentacion FUSIONA los dos frutos en un componente
    conexo y "el componente mas grande" devuelve una quimera que el pipeline
    mediria como un solo fruto. Es exactamente el caso que el chequeo de
    adyacencia tiene que atrapar ANTES de refinar.

    El desplazamiento horizontal se calcula de modo que los dos extremos que
    se enfrentan (los dos a la misma altura, el punto de maximo x de uno y el
    de minimo x del otro) queden a ``CLOSING_BRIDGE_PX`` pixeles: el ancho de
    la mascara menos sus dos margenes de 20 mm, mas el hueco.
    """
    mask, _gt = make_banana(mm_per_px=_MM_PER_PX, **_PLUMP_BANANA)
    mask_width_mm = mask.shape[1] * _MM_PER_PX
    gap_mm = CLOSING_BRIDGE_PX * _MM_PER_PX
    offset_mm = mask_width_mm - 2 * 20.0 + gap_mm

    layout = {
        1: (-10.0, -10.0),
        2: (offset_mm + 230.0, -10.0),
        3: (offset_mm + 230.0, 110.0),
        4: (-10.0, 110.0),
    }
    scene = make_banana_scene(
        [
            dict(_PLUMP_BANANA, position_mm=(0.0, 0.0)),
            dict(_PLUMP_BANANA, position_mm=(offset_mm, 0.0)),
        ],
        _MM_PER_PX,
        markers=layout,
    )
    return scene, _config_for(layout)


@pytest.fixture(scope="module")
def borderline_scene() -> np.ndarray:
    return make_banana_scene([_BORDERLINE_BANANA], _MM_PER_PX, markers=_BORDERLINE_LAYOUT)


@pytest.fixture(scope="module")
def borderline_cfg() -> AppConfig:
    return _config_for(_BORDERLINE_LAYOUT)


@pytest.fixture(scope="module")
def markers_only_scene() -> np.ndarray:
    return make_banana_scene([], _MM_PER_PX, markers=_PLUMP_LAYOUT)


# ---------------------------------------------------------------------------
# (1) End-to-end SIN correccion de paralaje, contra la verdad cruda
# ---------------------------------------------------------------------------


def test_end_to_end_without_parallax_correction_recovers_outer_length_ground_truth(
    sharp_scene, sharp_truth, sharp_cfg
):
    """La cadena completa, sobre una escena plana, recupera el riel exterior.

    Esta es la version corregida de la prueba del brief: la escena sintetica
    es PLANA, asi que la unica comparacion honesta contra
    ``gt.outer_length_mm`` es con la correccion de paralaje APAGADA (sobre
    una altura de objeto real de cero, el factor de paralaje es error puro,
    no correccion -- ver el docstring del modulo).

    No aserta solo la longitud: tambien exige calibracion ``OK`` sin avisos,
    exactamente una instancia, y que la mascara devuelta tenga la forma del
    LIENZO. Una implementacion que deje de medir no puede pasar esto.
    """
    pipeline, _detector = _build(sharp_scene, sharp_cfg, correct_parallax=False)
    r = pipeline.process(sharp_scene)

    assert r.warnings == []
    assert r.plane is not None
    assert r.plane.state is CalibrationState.OK
    assert r.rectified is not None
    assert r.rectified.shape[:2] == tuple(reversed(canvas_size_for(sharp_cfg)))
    assert r.raw_undistorted.shape == sharp_scene.shape
    assert len(r.results) == 1

    result = r.results[0]
    assert result.aggregate.length_mm == pytest.approx(sharp_truth.outer_length_mm, rel=0.03)
    assert result.aggregate.caliper_mm == pytest.approx(sharp_truth.caliber_mm, abs=0.5)
    assert result.aggregate.samples == 1
    assert result.aggregate.volume_mm3 > 0.0

    # La mascara reportada vive en el lienzo rectificado y describe fruta real.
    assert result.mask.dtype == bool
    assert result.mask.shape == r.rectified.shape[:2]
    assert np.count_nonzero(result.mask) > 0


# ---------------------------------------------------------------------------
# Geometria del overlay en vivo en ``BananaResult`` (encargo posterior al
# plan de 20 tareas): el pipeline tiene que entregar la geometria de
# PIXEL que ``measure_mask`` ya calculo, no descartarla ni recalcularla.
# ---------------------------------------------------------------------------


def test_banana_result_carries_the_pixel_geometry_measure_mask_computed(
    sharp_scene, sharp_cfg
):
    """``BananaResult.outer_curve_px`` tiene que ser literalmente la curva
    que produjo ``aggregate.length_mm`` de este mismo cuadro, no una
    aproximacion ni un placeholder vacio.

    Con una sola muestra (``aggregate.samples == 1``) la mediana del
    agregado es exactamente la medida cruda de este cuadro, asi que
    recalcular la longitud suavizada SOBRE ``outer_curve_px`` tiene que
    reproducir ``aggregate.length_mm`` al bit (misma funcion,
    ``smoothed_arc_length_mm``, sobre los mismos pixeles) -- con
    ``correct_parallax=False`` para que nada reescale el agregado lejos de
    la medida cruda que esos pixeles describen. Una implementacion que
    dejara estos campos vacios (regresion del seam Task 14 -> overlay) o
    que les asignara la geometria SIN RECORTAR (``full_outer`` en vez de
    ``outer`` dentro de ``measure_mask``) fallaria esta prueba: vacio no
    tiene puntos que medir, y sin recortar mide mas que el fruto.
    """
    pipeline, _detector = _build(sharp_scene, sharp_cfg, correct_parallax=False)
    r = pipeline.process(sharp_scene)
    assert len(r.results) == 1
    result = r.results[0]
    mm_per_px = r.plane.mm_per_px

    assert len(result.contour_px) >= 2
    assert len(result.axis_path_px) >= 2
    assert len(result.outer_curve_px) >= 2

    recomputed_length_mm = smoothed_arc_length_mm(result.outer_curve_px, mm_per_px)
    assert recomputed_length_mm == pytest.approx(result.aggregate.length_mm, rel=1e-9)

    # La curva Codex (recortada) nunca puede ser mas larga que el contorno
    # ENTERO de la mascara (que incluye las dos cadenas, exterior e
    # interior, mas el pedicelo si lo hay): es un sub-tramo de una de las
    # dos, nunca el total.
    assert len(result.outer_curve_px) < len(result.contour_px)


def test_pixel_geometry_is_not_rescaled_by_the_parallax_correction(sharp_scene, sharp_cfg):
    """``contour_px``/``axis_path_px``/``axis_tips_px``/``outer_curve_px``
    son posiciones de PIXEL del lienzo rectificado -- ``parallax.py`` dice
    explicitamente que no los toca, porque el paralaje corrige magnitudes
    FISICAS, no la grilla de pixeles con la que ya se dibujo la escena
    (ver docstring de ese modulo). Esta prueba lo verifica desde afuera de
    ``parallax.py``: dos pipelines sobre la MISMA escena, uno con la
    correccion encendida y otro apagada, tienen que coincidir PIXEL A PIXEL
    en la geometria pero DIFERIR en ``aggregate.length_mm`` -- si alguien
    reescalara los pixeles igual que los escalares, las dos geometrias
    dejarian de coincidir.
    """
    pipeline_corrected, _ = _build(sharp_scene, sharp_cfg, correct_parallax=True)
    pipeline_raw, _ = _build(sharp_scene, sharp_cfg, correct_parallax=False)

    corrected = pipeline_corrected.process(sharp_scene).results[0]
    raw = pipeline_raw.process(sharp_scene).results[0]

    assert corrected.aggregate.length_mm != pytest.approx(raw.aggregate.length_mm)
    assert np.array_equal(corrected.contour_px, raw.contour_px)
    assert np.array_equal(corrected.axis_path_px, raw.axis_path_px)
    assert np.array_equal(corrected.axis_tips_px[0], raw.axis_tips_px[0])
    assert np.array_equal(corrected.outer_curve_px, raw.outer_curve_px)


# ---------------------------------------------------------------------------
# ``FrameResult.markers`` (encargo posterior al plan de 20 tareas): el
# pipeline ya calcula ``detect_markers`` y lo tiraba al salir de
# ``process``, asi que la vista en vivo nunca podia mostrar los
# marcadores que sostienen la calibracion. Se exponen en TODOS los
# caminos de salida, incluido el ``BLOCKED``.
# ---------------------------------------------------------------------------


def test_frame_result_exposes_the_markers_it_already_detected(sharp_scene, sharp_cfg):
    pipeline, _ = _build(sharp_scene, sharp_cfg, correct_parallax=False)
    r = pipeline.process(sharp_scene)
    assert {d.id for d in r.markers} == set(sharp_cfg.marker_ids)


def test_markers_survive_a_degraded_frame(sharp_scene, sharp_cfg):
    degraded_scene = _hide_markers(sharp_scene, sharp_cfg, [3])
    pipeline, _ = _build(degraded_scene, sharp_cfg, correct_parallax=False)
    r = pipeline.process(degraded_scene)
    assert {d.id for d in r.markers} == {1, 2, 4}


def test_markers_survive_even_a_blocked_frame_with_no_results(sharp_scene, sharp_cfg):
    # El caso que mas importa para el operador/espectador: la calibracion
    # fallo y ``results`` esta vacio, pero los marcadores que SI siguen
    # visibles tienen que seguir llegando -- es exactamente la informacion
    # que explica POR QUE bloqueo, no solo QUE bloqueo.
    blocked_scene = _hide_markers(sharp_scene, sharp_cfg, [3, 4])
    pipeline, _ = _build(blocked_scene, sharp_cfg, correct_parallax=False)
    r = pipeline.process(blocked_scene)
    assert r.results == []
    assert {d.id for d in r.markers} == {1, 2}


def test_end_to_end_error_decomposes_into_pristine_and_warp_chain_bias(
    sharp_scene, sharp_mask, sharp_truth, sharp_cfg
):
    """El +0.32% de la prueba de arriba NO es la exactitud del sistema: son
    DOS sesgos de signo opuesto anulandose, y sin esta prueba esa
    cancelacion queda escondida dentro de un solo numero.

    Mecanismo (encontrado en revision, verificar antes de volver a
    derivarlo): ``trim_pedicel`` (Task 8) recorta ~13 mm del afinamiento
    floral de ``_SHARP_BANANA`` AUNQUE ``pedicel_mm=0.0`` -- con
    ``blossom_taper_mm=11.9`` sobre ``width_mm=12`` la caida de ancho es
    abrupta en relacion al cuerpo, y el criterio de derivada normalizada de
    ``trim_pedicel`` la dispara igual. Ese recorte cancela casi todo el
    sesgo conocido del fixture de Task 7 (+15..23% en ``w`` grande, mucho
    menor aqui por ser ``w=12``): ``measure_mask`` sobre la mascara
    PRISTINA (sin escena, sin marcadores, sin rectificacion, sin
    ``refine_mask``) mide apenas **-0.04%**, muy por debajo del ~1.4% que
    el presupuesto de la tarea asumia para ese termino solo. El pipeline
    COMPLETO le suma encima el sesgo de la cadena ArUco -> homografia ->
    rectificacion -> deteccion -> ``refine_mask``, y el resultado neto
    (+0.32%) sigue siendo pequeño -- pero por composicion de dos terminos
    pequeños de signo distinto, no porque cada eslabon sea exacto por si
    solo.

    Por que esto necesita DOS aserciones, no una: pinear solo el +0.32%
    total deja ciego a un cambio que empeorara la cadena de warp un 1.4%
    mientras el recorte de pedicelo se volviera 1.4% MENOS agresivo (o
    viceversa) -- las dos prueba de arriba seguiria sin moverse, porque los
    dos terminos se compensarian de nuevo, por una coincidencia distinta de
    parametros. Pineando el termino PRISTINO por separado (que no pasa por
    la cadena de warp en absoluto), un cambio en la agresividad del recorte
    se ve aqui sin importar que haga la cadena de warp; pineando el
    COMPLETO por separado con un margen mas angosto que ``rel=0.03``, un
    cambio en la cadena de warp se ve sin importar que haga el recorte.

    No tratar el +0.32% como la precision del sistema en ningun reporte:
    el objetivo honesto declarado es +-1 mm, con el recorte de pedicelo
    dominando a +-3 mm sobre fruta con cabo real (spec S7.b, HANDOFF). Este
    numero es una coincidencia de los parametros de ESTE fixture sintetico.
    """
    pristine = measure_mask(sharp_mask, _MM_PER_PX, sharp_cfg)
    assert pristine is not None
    # Termino PRISTINO: fixture de Task 7 + recorte de pedicelo de Task 8,
    # sin ningun eslabon de la cadena de adquisicion. ~1 mm de margen
    # (~0.11% de 904 mm) alcanza para el -0.33 mm medido y para romperse si
    # el recorte se volviera tan agresivo -- o tan laxo -- como predecia el
    # presupuesto original (~1.4%, ~12.7 mm).
    assert pristine.codex_length_mm == pytest.approx(sharp_truth.outer_length_mm, abs=1.0)

    pipeline, _detector = _build(sharp_scene, sharp_cfg, correct_parallax=False)
    full = pipeline.process(sharp_scene).results[0].aggregate
    # Termino COMPLETO: le suma a lo de arriba la cadena ArUco/homografia/
    # rectificacion/deteccion/``refine_mask``. 4 mm (~0.44%) es mas angosto
    # que el ``rel=0.03`` (~27 mm) de la prueba de arriba a proposito: ese
    # margen amplio existe para la interfaz publicada, no para esta prueba,
    # que quiere notar un movimiento de la cadena de warp que el margen
    # amplio absorberia sin pestañear.
    assert full.length_mm == pytest.approx(sharp_truth.outer_length_mm, abs=4.0)


# ---------------------------------------------------------------------------
# (2) End-to-end CON correccion, fijando el factor exacto
# ---------------------------------------------------------------------------


def test_parallax_correction_applies_the_exact_closed_form_factor(sharp_scene, sharp_cfg):
    """Con la correccion encendida, cada magnitud escala por el factor EXACTO.

    No "aproximadamente 6%": Task 9 ya demostro que esa forma de asercion no
    discrimina (un factor de una sola pasada, 1.0723, entraba en ``rel=0.02``
    de 1.065 y pasaba feliz). Lo que si discrimina es el pin exacto, a
    ``rel=1e-12``: la longitud corregida tiene que ser la longitud cruda por
    ``2*H_ef / (2*H_ef + calibre_crudo)``, con el area al cuadrado y el
    volumen al cubo.

    Atrapa tres roturas distintas: saltarse ``correct_measures`` (razon 1.0),
    invertir el factor (razon ``1/f``, un 4.6% al otro lado), y empujar al
    agregador ANTES de corregir (razon 1.0 tambien).
    """
    raw_pipeline, _ = _build(sharp_scene, sharp_cfg, correct_parallax=False)
    raw = raw_pipeline.process(sharp_scene).results[0].aggregate

    corrected_pipeline, _ = _build(sharp_scene, sharp_cfg, correct_parallax=True)
    corrected = corrected_pipeline.process(sharp_scene).results[0].aggregate

    factor = _parallax_factor(sharp_cfg.effective_height_mm, raw.caliper_mm)

    # El factor encoge (el objeto esta MAS CERCA de la camara que el plano de
    # calibracion), y lo hace de forma apreciable: si fuera ~1.0 el pin de
    # abajo no distinguiria nada.
    assert 0.9 < factor < 1.0
    assert factor == pytest.approx(0.9773060029282578, rel=1e-12)

    assert corrected.length_mm == pytest.approx(raw.length_mm * factor, rel=1e-12)
    assert corrected.caliper_mm == pytest.approx(raw.caliper_mm * factor, rel=1e-12)
    assert corrected.volume_mm3 == pytest.approx(raw.volume_mm3 * factor**3, rel=1e-12)


# ---------------------------------------------------------------------------
# (3) El detector corre sobre el lienzo rectificado, no sobre el cuadro crudo
# ---------------------------------------------------------------------------


def test_the_detector_runs_on_the_rectified_canvas_not_the_raw_frame(sharp_scene, sharp_cfg):
    """Contrato de Task 12/13: ``Instance.mask`` sale con la forma del cuadro
    que recibe ``detect()``, y ``refine_mask`` depende de que esa forma sea la
    del lienzo rectificado.

    La escena cruda y el lienzo tienen tamanos DISTINTOS a proposito (la
    escena lleva 20 mm de margen mas alla de los marcadores), asi que la
    asercion no puede pasar por casualidad.
    """
    pipeline, detector = _build(sharp_scene, sharp_cfg, correct_parallax=False)
    r = pipeline.process(sharp_scene)

    assert len(detector.frames) == 1
    assert detector.frames[0].shape == r.rectified.shape
    assert detector.frames[0].shape != r.raw_undistorted.shape


# ---------------------------------------------------------------------------
# (4) 3 de 4 marcadores: sigue midiendo Y lo avisa
# ---------------------------------------------------------------------------


def test_three_of_four_markers_keeps_measuring_and_warns_about_degraded_calibration(
    sharp_scene, sharp_truth, sharp_cfg
):
    """Spec S7 etapa 2: con 3 marcadores la calibracion es posible sobre 12
    correspondencias y el sistema DEBE seguir midiendo, declarandolo.

    El brief nunca expone ``PlaneCalibration.state``, que existe desde Task 5
    precisamente para esto. Degradar en silencio seria reportar un numero sin
    decir que su calibracion perdio redundancia.
    """
    degraded_scene = _hide_markers(sharp_scene, sharp_cfg, [3])
    pipeline, _ = _build(degraded_scene, sharp_cfg, correct_parallax=False)
    r = pipeline.process(degraded_scene)

    assert r.plane is not None
    assert r.plane.state is CalibrationState.DEGRADED

    assert any("3/4" in w for w in r.warnings), r.warnings
    assert any("degraded" in w.lower() for w in r.warnings), r.warnings

    # Y lo esencial: sigue midiendo, con la misma exactitud.
    assert len(r.results) == 1
    assert r.results[0].aggregate.length_mm == pytest.approx(
        sharp_truth.outer_length_mm, rel=0.03
    )


# ---------------------------------------------------------------------------
# (5) Calibracion bloqueada: nada de medidas, y un aviso
# ---------------------------------------------------------------------------


def test_no_results_and_a_warning_when_calibration_is_blocked(sharp_scene, sharp_cfg):
    """Con 2 marcadores no hay geometria suficiente: "sin datos" antes que un
    numero falso.

    La version del brief solo aserta ``results == []``, que un pipeline que no
    mide nunca pasa igual. Por eso la segunda mitad: el MISMO pipeline, sobre
    la escena sana, tiene que medir. Ninguna implementacion que deje de medir
    pasa las dos mitades a la vez.
    """
    blocked_scene = _hide_markers(sharp_scene, sharp_cfg, [3, 4])
    pipeline, _ = _build(blocked_scene, sharp_cfg, correct_parallax=False)

    blocked = pipeline.process(blocked_scene)
    assert blocked.results == []
    assert blocked.plane is None
    assert blocked.rectified is None
    assert any("calibration" in w.lower() for w in blocked.warnings), blocked.warnings

    healthy = pipeline.process(sharp_scene)
    assert len(healthy.results) == 1
    assert healthy.results[0].aggregate.length_mm > CODEX_MIN_LENGTH_MM


def test_a_degenerate_marker_layout_propagates_instead_of_becoming_a_warning():
    """Un ``marker_layout`` colineal es un error de CONFIGURACION persistente,
    no un estado transitorio del cuadro.

    Task 5 decidio lanzar ``DegenerateMarkerLayout`` en vez de devolver
    ``None`` exactamente para que no se confunda con ``BLOCKED``. Si el
    pipeline lo convirtiera en un aviso mas, el operador quedaria mirando
    "calibracion bloqueada" para siempre, buscando un problema de marcadores
    que no existe. Esta prueba es la guardia contra un ``except`` generico
    (el proyecto ya produjo uno, en Task 8).

    Centros colineales SOLOS no alcanzan: cada marcador aporta 4 esquinas
    que abarcan ``marker_length_mm`` (24 mm por defecto), asi que los 16
    puntos de mundo no son colineales aunque sus 4 centros lo sean --
    verificado a mano: con el layout de abajo y ``marker_length_mm=24.0``
    ``compute_plane_calibration`` devuelve una calibracion ``OK`` valida
    (residual ~0.014 mm), no lanza nada. Esto ya se habia resuelto en Task 5
    (el reviewer lo afirmo, el implementador lo disputo, el re-reviewer lo
    verifico) y esta en el ledger del proyecto. Lo que si colapsa los 16
    puntos a una recta es ``marker_length_mm=0.0`` -- cada "esquina" coincide
    con el centro de su marcador -- y es lo que esta prueba usa.
    """
    collinear = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (200.0, 0.0), 4: (300.0, 0.0)}
    scene = make_banana_scene([], _MM_PER_PX, markers=collinear)
    cfg = dataclasses.replace(_config_for(collinear), marker_length_mm=0.0)
    pipeline, _ = _build(scene, cfg)

    with pytest.raises(DegenerateMarkerLayout):
        pipeline.process(scene)


# ---------------------------------------------------------------------------
# (6) Adyacencia: frutos demasiado juntos
# ---------------------------------------------------------------------------


def _two_square_blobs_bgr(gap_px: int, blob_size_px: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Lienzo BGR con dos manchas SATURADAS cuadradas, cara a cara, separadas
    exactamente ``gap_px`` pixeles, mas una mascara gruesa que cubre a las
    dos (para que ``refine_mask`` no las recorte por su propio bbox).

    Fondo blanco (``saturacion 0``), manchas amarillo puro ``(0, 255, 255)``
    en BGR (``saturacion 255`` en HSV) -- la misma separacion de canal que usa
    el resto del modulo para distinguir fruta de fondo.
    """
    height = blob_size_px + 40
    width = 2 * blob_size_px + gap_px + 40
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y0 = 20
    x0_left = 20
    x0_right = x0_left + blob_size_px + gap_px
    canvas[y0 : y0 + blob_size_px, x0_left : x0_left + blob_size_px] = (0, 255, 255)
    canvas[y0 : y0 + blob_size_px, x0_right : x0_right + blob_size_px] = (0, 255, 255)

    coarse_mask = np.zeros((height, width), dtype=bool)
    coarse_mask[y0 : y0 + blob_size_px, x0_left : x0_right + blob_size_px] = True
    return canvas, coarse_mask


def test_closing_bridge_px_matches_refine_masks_actual_bridging_distance():
    """``CLOSING_BRIDGE_PX`` tiene que seguir siendo la distancia de puenteo
    REAL de ``refine_mask``, no una constante copiada de un script aparte.

    La prueba anterior (``test_the_adjacency_threshold_exceeds_...`` abajo)
    solo compara ``ADJACENCY_DILATE_PX > CLOSING_BRIDGE_PX`` -- dos
    constantes de este modulo entre si. Eso queda en VERDE aunque las dos
    deriven mal, o aunque alguien cambie ``_CLOSE_KERNEL_PX`` en
    ``segmentation.py`` sin tocar esta constante: nada ejercita el
    ``refine_mask`` real. Esta prueba si lo hace.

    A un hueco de EXACTAMENTE ``CLOSING_BRIDGE_PX`` pixeles, el cierre
    puentea las dos manchas y ``refine_mask`` (que se queda con el
    componente conexo MAS GRANDE) debe devolver las DOS fusionadas en una
    sola mascara -- su area tiene que acercarse a la SUMA de las dos. A
    ``CLOSING_BRIDGE_PX + 1`` (ya no puenteable segun la constante),
    ``refine_mask`` debe quedarse con UNA sola -- su area tiene que
    acercarse a la MITAD. Si ``CLOSING_BRIDGE_PX`` estuviera mal (por
    ejemplo, un ``_CLOSE_KERNEL_PX`` distinto en produccion), uno de los dos
    huecos no haria lo que su nombre promete y la comparacion de abajo
    fallaria.
    """
    bridged_bgr, bridged_coarse = _two_square_blobs_bgr(CLOSING_BRIDGE_PX)
    separated_bgr, separated_coarse = _two_square_blobs_bgr(CLOSING_BRIDGE_PX + 1)

    bridged = refine_mask(bridged_bgr, bridged_coarse)
    separated = refine_mask(separated_bgr, separated_coarse)

    assert bridged is not None
    assert separated is not None

    bridged_px = int(np.count_nonzero(bridged))
    separated_px = int(np.count_nonzero(separated))

    # Fusionadas (gap == CLOSING_BRIDGE_PX): el area cubre las DOS manchas.
    # Separadas (gap == CLOSING_BRIDGE_PX + 1): el area cubre solo UNA.
    # El factor 1.5 deja margen de sobra entre "el doble" y "la mitad" sin
    # acercarse a ninguno de los dos, asi que no puede pasar por casualidad.
    assert bridged_px > separated_px * 1.5


def test_the_adjacency_threshold_exceeds_the_measured_closing_bridge():
    """El umbral de adyacencia se DERIVA, no se adivina.

    ``refine_mask`` cierra con un kernel de 9x9 px que puentea un hueco de
    hasta ``CLOSING_BRIDGE_PX`` (8 px = 1.6 mm, medido -- ver la prueba de
    arriba, que ejercita ``refine_mask`` de verdad en vez de comparar solo
    estas dos constantes entre si). Por debajo de eso la segmentacion
    fusiona dos frutos y "el componente conexo mas grande" devuelve una
    quimera. El umbral tiene que ser ESTRICTAMENTE mayor, o existe una
    banda de separaciones en la que el pipeline mide una quimera sin aviso.
    """
    assert ADJACENCY_DILATE_PX > CLOSING_BRIDGE_PX


def test_bananas_within_the_closing_bridge_are_refused_with_a_warning(touching_scene_and_cfg):
    """Dos frutos a la distancia de puenteo: no se mide ninguno, y se avisa."""
    scene, cfg = touching_scene_and_cfg
    pipeline, _ = _build(scene, cfg, correct_parallax=False)
    r = pipeline.process(scene)

    assert r.plane is not None  # la calibracion esta bien; el problema es la fruta
    assert r.results == []
    assert any("separate" in w.lower() for w in r.warnings), r.warnings


def test_the_adjacency_warning_the_pipeline_writes_is_the_one_the_window_shows(
    touching_scene_and_cfg, qtbot
):
    """C2 (CRITICO, ronda de correcciones finales): la costura productor ->
    consumidor de ``FrameResult.warnings``.

    ``warnings`` no tenia NINGUN consumidor en todo el proyecto:
    ``MainWindow.update_frame_result`` leia ``plane`` y ``results`` y pasaba
    de largo. Los cinco mensajes de operador que este modulo escribe eran
    inalcanzables, entre ellos el de esta prueba -- y la spec S7 etapa 4 no
    lo deja opcional: con dos mascaras fusionadas el sistema tiene que
    negarse a medir *y mostrar el aviso*, porque "una medida falsa es peor
    que ninguna".

    Esta prueba vive aqui, junto al fixture que produce el caso real, porque
    lo que fija no es "la ventana muestra algo": es que la REDACCION EXACTA
    que ``Pipeline.process`` compone llegue intacta a lo que el operador lee.
    Una prueba del lado de la interfaz con un ``warnings`` escrito a mano (la
    hay, en ``test_ui_layout.py``) rechaza la implementacion que ignora el
    campo, pero no detecta que alguien reescriba el mensaje de este modulo y
    deje al operador sin la frase de accion.

    La implementacion equivocada que RECHAZA: cualquiera en la que el aviso
    del pipeline no sea literalmente el texto mostrado -- se afirma la
    inclusion del mensaje COMPLETO, carácter por carácter, no una subcadena
    elegida a mano.
    """
    from codex_caliper.ui.main_window import MainWindow

    scene, cfg = touching_scene_and_cfg
    pipeline, _ = _build(scene, cfg, correct_parallax=False)
    result = pipeline.process(scene)

    adjacency_warnings = [w for w in result.warnings if "separate" in w.lower()]
    assert adjacency_warnings, result.warnings

    window = MainWindow(intrinsics=_identity_intrinsics(scene))
    qtbot.addWidget(window)
    window.update_frame_result(result)

    shown = window.notice_text()
    for warning in adjacency_warnings:
        assert warning in shown, f"el aviso del pipeline no llego a la interfaz: {warning!r}"
    # La frase de ACCION es la mitad util del mensaje: un canal que mostrara
    # un generico "a warning occurred" dejaria al operador sin saber que hacer.
    assert "Separate them on the tray." in shown
    # Y el panel no puede decir "sin fruta" como si no pasara nada: el cuadro
    # no es medible, y la razon ahora se puede leer.
    assert result.results == []

    window.close()


def test_two_well_separated_bananas_are_both_measured(plump_scene, plump_cfg):
    """La contraparte positiva de la adyacencia: a 30 mm de separacion -- la
    distancia a la que el fixture y el rig real colocan dos frutos -- los dos
    se miden, con identidades distintas y sin ningun aviso.

    Sin esta prueba, el chequeo de adyacencia podria estar disparando SIEMPRE
    y las pruebas de "no hay resultados" seguirian verdes.
    """
    pipeline, _ = _build(plump_scene, plump_cfg, correct_parallax=False)
    r = pipeline.process(plump_scene)

    assert r.warnings == []
    assert len(r.results) == 2
    assert len({result.track_id for result in r.results}) == 2
    for result in r.results:
        assert result.aggregate.length_mm > CODEX_MIN_LENGTH_MM
        assert result.aggregate.caliper_mm == pytest.approx(
            _PLUMP_BANANA["width_mm"], abs=0.5
        )
        assert result.verdict.passes


# ---------------------------------------------------------------------------
# (7) Cuadros sin fruta / mascaras no medibles
# ---------------------------------------------------------------------------


def test_zero_detections_returns_an_empty_result_without_raising(
    markers_only_scene, plump_cfg
):
    """Un cuadro calibrado pero sin fruta es el caso MAS ORDINARIO del rig, no
    un error: lista vacia y NINGUN aviso.

    Fortalecida respecto del brief: ademas de ``results == []`` exige que la
    calibracion siga siendo valida (``plane is not None``), que es lo que
    distingue "no hay fruta" de "no puedo medir".
    """
    pipeline, _ = _build(markers_only_scene, plump_cfg)
    r = pipeline.process(markers_only_scene)

    assert r.results == []
    assert r.warnings == []
    assert r.plane is not None
    assert r.rectified is not None


def test_a_fruit_clipped_by_the_canvas_border_is_not_measured_but_is_reported(
    borderline_scene,
):
    """Fruta que sale del area calibrada: ``measure_mask`` devuelve ``None``
    (toca el borde) y el pipeline lo dice en vez de reportar una longitud
    truncada como si fuera buena.
    """
    cfg = _config_for(_CLIPPED_LAYOUT)
    scene = make_banana_scene([_BORDERLINE_BANANA], _MM_PER_PX, markers=_CLIPPED_LAYOUT)
    pipeline, _ = _build(scene, cfg)
    r = pipeline.process(scene)

    assert r.plane is not None
    assert r.results == []
    assert any("border" in w.lower() for w in r.warnings), r.warnings


# ---------------------------------------------------------------------------
# (8) Peso
# ---------------------------------------------------------------------------


def test_weight_is_none_until_a_density_is_supplied(plump_scene, plump_cfg):
    """Sin densidad no hay peso -- pero si hay medida: el volumen esta ahi."""
    pipeline, _ = _build(plump_scene, plump_cfg, density_g_per_cm3=None)
    r = pipeline.process(plump_scene)

    assert len(r.results) == 2
    for result in r.results:
        assert result.estimated_weight_g is None
        assert result.aggregate.volume_mm3 > 0.0


def test_weight_is_the_volume_in_cubic_centimetres_times_the_density(plump_scene, plump_cfg):
    """La conversion mm3 -> cm3 (dividir por 1000) fijada EXACTAMENTE.

    La densidad esta en g/cm3 y el volumen en mm3: omitir o equivocar esa
    division es el modo de fallo recurrente de este proyecto, y la unica
    prueba de peso del brief cubre solo la rama ``None``. Por eso van las dos
    aserciones: el pin exacto (que atrapa un factor 1000 en cualquier
    direccion) y una banda de plausibilidad fisica (que lo atrapa tambien si
    alguien "arregla" el pin copiando la formula equivocada).
    """
    pipeline, _ = _build(plump_scene, plump_cfg, density_g_per_cm3=_DENSITY_G_PER_CM3)
    r = pipeline.process(plump_scene)

    assert len(r.results) == 2
    for result in r.results:
        expected_g = result.aggregate.volume_mm3 / 1000.0 * _DENSITY_G_PER_CM3
        assert result.estimated_weight_g == pytest.approx(expected_g, rel=1e-12)
        # Un banano de este tamano pesa entre 50 y 400 g. Un factor 1000 en
        # cualquier direccion sale de esta banda por tres ordenes de magnitud.
        assert 50.0 < result.estimated_weight_g < 400.0


# ---------------------------------------------------------------------------
# (9) Veredicto CODEX sobre el agregado CORREGIDO
# ---------------------------------------------------------------------------


def test_the_verdict_is_computed_from_the_parallax_corrected_aggregate(
    borderline_scene, borderline_cfg
):
    """El calibre de este banano cruza el minimo CODEX al corregirse.

    Crudo mide ~27.6 mm (por encima de los 27.0 mm del estandar); corregido
    por paralaje cae a ~26.2 mm (por debajo). Las dos corridas tienen que dar
    veredictos OPUESTOS sobre el calibre -- es el unico caso que distingue
    "se evalua el agregado corregido" de "se evalua la medida cruda", porque
    en cualquier otro banano los dos veredictos coinciden.
    """
    raw_pipeline, _ = _build(borderline_scene, borderline_cfg, correct_parallax=False)
    raw = raw_pipeline.process(borderline_scene).results[0]

    corrected_pipeline, _ = _build(borderline_scene, borderline_cfg, correct_parallax=True)
    corrected = corrected_pipeline.process(borderline_scene).results[0]

    # El fixture tiene que seguir bracketeando el umbral: si una mejora aguas
    # arriba mueve el calibre, esta prueba deja de discriminar y hay que
    # reajustar ``_BORDERLINE_BANANA``, no la asercion.
    assert raw.aggregate.caliper_mm > CODEX_MIN_CALIPER_MM
    assert corrected.aggregate.caliper_mm < CODEX_MIN_CALIPER_MM

    assert raw.verdict.caliper_ok is True
    assert corrected.verdict.caliper_ok is False
    assert corrected.verdict.length_ok is True
    assert corrected.verdict.passes is False


# ---------------------------------------------------------------------------
# (10) El acumulador temporal no se reinicia por un cuadro perdido
# ---------------------------------------------------------------------------


def test_the_aggregator_is_not_reset_by_a_frame_without_detections(plump_scene, plump_cfg):
    """Spec S7 etapa 4: la reasociacion por centroide existe para que perder
    un ``track_id`` NO reinicie el acumulador.

    Tres cuadros buenos, uno perdido (misma escena, fruta borrada: la
    calibracion sigue intacta), y un cuarto cuadro bueno con ids CRUDOS
    nuevos (``model.id_offset``, ver el doble de YOLO). Si el cuadro
    perdido reiniciara los buffers, el ultimo agregado volveria a
    ``samples == 1`` -- y si el pipeline tirara su memoria de centroides en
    ese cuadro vacio, los ids crudos nuevos no tendrian contra que
    reasociarse y saldrian como ``track_id`` tal cual, distintos de los de
    antes del hueco.

    La version anterior de esta prueba NO distinguia eso: el doble de YOLO
    deriva sus ids del orden de barrido de ``connectedComponents`` sobre
    una escena quieta, asi que sin ``id_offset`` los ids crudos salen
    identicos en todos los cuadros por construccion, pasen por
    ``CentroidReassigner`` o no. Un pipeline que pisara
    ``self._previous_instances`` con la lista vacia del cuadro perdido
    pasaba esta prueba igual, porque los ids crudos del cuadro siguiente
    coincidian con los track_id anteriores por casualidad del doble, no
    porque la reasociacion los hubiera recuperado. Subir ``id_offset``
    despues del hueco fuerza ids crudos DISTINTOS, asi que la unica forma
    de recuperar la identidad es que la memoria de centroides sobreviva al
    cuadro vacio.
    """
    pipeline, detector = _build(plump_scene, plump_cfg)
    model = detector.inner.model

    for expected_samples in (1, 2, 3):
        r = pipeline.process(plump_scene)
        assert [result.aggregate.samples for result in r.results] == [expected_samples] * 2
    track_ids_before_gap = sorted(result.track_id for result in r.results)

    dropped = pipeline.process(_hide_fruit(plump_scene))
    assert dropped.results == []
    assert dropped.plane is not None

    model.id_offset += 1000
    r = pipeline.process(plump_scene)
    assert [result.aggregate.samples for result in r.results] == [4, 4]
    # Los ids CRUDOS que entrego YOLO para este cuadro son nuevos
    # (desplazados por ``id_offset``); que el pipeline haya conservado el
    # MISMO ``track_id`` que antes del hueco solo puede venir de
    # ``CentroidReassigner`` reasociando por cercania de centroide, no de
    # una coincidencia de ids crudos.
    assert sorted(result.track_id for result in r.results) == track_ids_before_gap


def test_the_aggregate_window_follows_the_configured_temporal_window(plump_cfg):
    """``Pipeline`` no inventa su propia ventana temporal: usa la de la config."""
    scene = make_banana_scene([_PLUMP_BANANA], _MM_PER_PX, markers=_PLUMP_LAYOUT)
    cfg = AppConfig(
        marker_thickness_mm=_MARKER_THICKNESS_MM,
        canvas_mm_per_px=_MM_PER_PX,
        marker_layout=_PLUMP_LAYOUT,
        temporal_window=2,
    )
    pipeline, _ = _build(scene, cfg)

    assert pipeline.process(scene).results[0].aggregate.samples == 1
    assert pipeline.process(scene).results[0].aggregate.samples == 2
    # La ventana esta llena: un tercer cuadro no la agranda.
    assert pipeline.process(scene).results[0].aggregate.samples == 2

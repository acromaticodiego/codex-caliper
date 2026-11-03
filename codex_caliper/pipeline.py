"""Orquestacion del pipeline por cuadro (Task 14).

Primera tarea donde TODOS los modulos se conectan. No agrega geometria ni
heuristica propia: fija el ORDEN y las UNIDADES de la cadena, y decide que
hacer cuando un eslabon dice "no puedo". Un cuadro recorre, en este orden:

    undistort -> detect_markers -> compute_plane_calibration -> rectify
    -> detector.detect (sobre el LIENZO) -> CentroidReassigner
    -> masks_are_adjacent por pares -> refine_mask -> measure_mask
    -> correct_measures -> aggregator.push -> codex.evaluate -> peso

La regla que gobierna todo el modulo: **nunca reportar una medida sin
calibracion valida; "sin datos" antes que un numero falso.** Por eso los
problemas se ACUMULAN en ``FrameResult.warnings`` en vez de lanzarse (un
cuadro malo no debe tumbar un sistema en vivo) y por eso cada rama que no
puede medir devuelve una lista de resultados vacia, nunca una medida
parcial.

El detector corre sobre el LIENZO RECTIFICADO, no sobre el cuadro crudo
-------------------------------------------------------------------------
``Instance.mask`` (Task 12) sale con la forma del cuadro que se le pasa a
``detect()``, y ``refine_mask`` (Task 13) usa esa mascara como recorte
DENTRO del lienzo rectificado que recibe como primer argumento. Si el
detector corriera sobre el cuadro crudo, las dos cosas estarian en sistemas
de coordenadas distintos: ``refine_mask`` recortaria la region equivocada
del lienzo y devolveria una mascara plausible pero desplazada. No hay nada
que lance una excepcion en ese camino -- es exactamente el tipo de error
silencioso que este proyecto ya repitio -- asi que el orden de arriba no es
una preferencia de estilo.

Que NO se captura (y por que)
------------------------------
``DegenerateMarkerLayout`` (Task 5) se deja PROPAGAR. ``BLOCKED`` es un
estado transitorio que el pipeline maneja cada cuadro: el marcador tapado
reaparece y el sistema se recalibra solo. Un ``marker_layout`` degenerado
(colineal, tipicamente un error de tipeo en la config) no se arregla nunca
por si solo, y convertirlo en un aviso mas dejaria al operador mirando
"calibracion bloqueada" para siempre, buscando un problema de marcadores que
no existe. Tampoco hay ningun ``except`` generico en este modulo: un error
de programacion aguas arriba tiene que verse, no convertirse en un ``None``
silencioso (el proyecto ya produjo ese defecto exacto en Task 8).

Degradacion elegante con 3 marcadores (spec S7 etapa 2)
--------------------------------------------------------
Con 3 marcadores quedan 12 correspondencias para 8 incognitas: la
homografia sigue siendo ajustable, asi que el sistema SIGUE MIDIENDO -- y lo
declara en ``warnings``. ``PlaneCalibration.state`` carga ese
``DEGRADED`` desde Task 5; no exponerlo seria reportar un numero sin decir
que su calibracion perdio toda la redundancia.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from codex_caliper.aggregation import MeasurementAggregator, TrackAggregate
from codex_caliper.aruco_tools import CalibrationState, MarkerDetection, detect_markers
from codex_caliper.codex import CodexVerdict, evaluate
from codex_caliper.config import AppConfig
from codex_caliper.detection import CentroidReassigner, Instance, masks_are_adjacent
from codex_caliper.geometry.measures import measure_mask
from codex_caliper.homography import (
    PlaneCalibration,
    canvas_size_for,
    compute_plane_calibration,
    rectify,
)
from codex_caliper.intrinsics import Intrinsics, undistort
from codex_caliper.parallax import correct_measures
from codex_caliper.segmentation import refine_mask

#: Cantidad de marcadores del rig. Aparece en los avisos ("3/4 ArUcos"), no
#: en ningun calculo: ``calibration_state`` ya decide el estado por su cuenta.
MARKER_COUNT = 4

#: Esquinas que aporta cada marcador ArUco a la homografia (Task 5).
_CORNERS_PER_MARKER = 4

#: Hueco maximo, en pixeles, que el CIERRE morfologico de ``refine_mask``
#: (kernel cuadrado de 9x9 px, Task 13) puede puentear entre dos manchas.
#:
#: MEDIDO, no asumido: con dos blobs separados por ``g`` pixeles y la misma
#: secuencia apertura(3)+cierre(9) que aplica ``refine_mask``, el resultado
#: queda en UN componente conexo para todo ``g <= 8`` y en DOS desde ``g = 9``
#: -- identico para contactos cara a cara anchos y para barras finas, y nunca
#: puentea un contacto en diagonal (ambos kernels son cuadrados, asi que la
#: distancia relevante es la de Chebyshev, la misma que usa
#: ``masks_are_adjacent``). A la escala fija del lienzo (0.2 mm/px) son
#: 1.6 mm.
CLOSING_BRIDGE_PX = 8

#: Separacion maxima, en pixeles del lienzo, a la que dos mascaras se
#: consideran "el mismo cuerpo" y el cuadro se declara no medible.
#:
#: Se DERIVA de ``CLOSING_BRIDGE_PX``, no se elige: por debajo de esa
#: separacion la propia segmentacion fusiona los dos frutos en un unico
#: componente conexo y "el componente mas grande" de ``refine_mask`` devuelve
#: una QUIMERA que el pipeline mediria como un solo banano -- un numero
#: plausible y falso, sin ninguna excepcion de por medio. El umbral tiene que
#: ser ESTRICTAMENTE mayor que la distancia de puenteo, o queda una banda de
#: separaciones en la que eso pasa sin aviso.
#:
#: 10 px = 2.0 mm: dos pixeles de margen sobre el minimo estricto (9 px), un
#: numero redondo en milimetros a la escala del lienzo, y muy por debajo de
#: los 30 mm a los que el fixture sintetico y el rig real separan dos frutos
#: -- asi que no puede disparar sobre fruta bien colocada. El default de
#: ``masks_are_adjacent`` (3 px = 0.6 mm, "solo si se tocan") queda POR DEBAJO
#: de la distancia de puenteo y por lo tanto no sirve aqui; Task 12 ya
#: advertia que el valor operativo lo decide esta tarea.
ADJACENCY_DILATE_PX = 10

#: Tolerancia de la reasociacion por centroide, en milimetros.
#:
#: El rig es una vista cenital fija y la fruta se coloca a mano y despues se
#: queda quieta: lo unico que esta tolerancia tiene que cubrir es el temblor
#: del centroide entre cuadros consecutivos por ruido de mascara, no
#: movimiento real. 10 mm es un orden de magnitud mas que ese temblor y un
#: tercio de la separacion minima entre dos frutos del plato (30 mm), asi que
#: no puede trasladar la identidad de un fruto al de al lado.
REASSIGN_MAX_DISTANCE_MM = 10.0

#: Milimetros cubicos por centimetro cubico. La densidad llega en g/cm3 y el
#: volumen en mm3: esta division es la conversion de unidades del peso, y
#: omitirla lo multiplica por mil.
_MM3_PER_CM3 = 1000.0


@dataclasses.dataclass(frozen=True)
class BananaResult:
    """Un fruto medido en un cuadro, con su veredicto y su peso estimado.

    ``aggregate`` es el resumen TEMPORAL del ``track_id`` (mediana e
    incertidumbre sobre la ventana, Task 11), no la medida de este cuadro
    suelto: el veredicto y el peso se derivan de ese agregado, no de la
    muestra individual. ``mask`` es la mascara refinada, booleana y en
    coordenadas del lienzo rectificado (la misma forma que
    ``FrameResult.rectified``), lista para dibujar encima.

    Geometria para el overlay en vivo (encargo posterior al plan de 20
    tareas, "haz visible el metodo en el video")
    -----------------------------------------------------------------------
    ``contour_px``, ``axis_path_px``, ``axis_tips_px`` y ``outer_curve_px``
    son exactamente los campos del mismo nombre de ``BananaMeasures``
    (Task 8, ``geometry/measures.py``) despues de pasar por
    ``correct_parallax`` -- sin cambios, porque esos cuatro son PIXELES del
    lienzo, no magnitudes fisicas (ver docstring de ``parallax.py``). Viajan
    hasta aqui, y no se quedan solo en ``BananaMeasures``, porque
    ``codex_caliper.ui.overlay`` -- que corre en el hilo de la GUI, por
    cuadro -- importa ``BananaResult``, no ``BananaMeasures``: este modulo
    (Task 14) es el unico punto de union entre la cadena de medicion y la
    interfaz, y el overlay no tiene ninguna razon para conocer el tipo
    intermedio. Recalcularlos en el overlay en vez de cargarlos hasta aqui
    costaria ~11 ms/banano (``medial_axis_path``, ver ``medial.py``) EN EL
    HILO DE LA GUI, cada cuadro -- exactamente el costo que este seam evita.
    """

    track_id: int
    aggregate: TrackAggregate
    verdict: CodexVerdict
    estimated_weight_g: float | None
    mask: np.ndarray
    contour_px: np.ndarray
    axis_path_px: np.ndarray
    axis_tips_px: tuple[np.ndarray, np.ndarray]
    outer_curve_px: np.ndarray


@dataclasses.dataclass
class FrameResult:
    """Todo lo que el pipeline produjo para un cuadro.

    ``rectified`` es ``None`` -- y no un lienzo en negro -- cuando la
    calibracion quedo ``BLOCKED``: sin homografia no existe el lienzo
    metrico, y devolver una imagen vacia invitaria a dibujar sobre ella como
    si significara algo. ``raw_undistorted`` siempre esta (la correccion de
    distorsion no depende de los marcadores), asi que la interfaz siempre
    tiene algo que mostrar. ``results`` esta vacio en todo cuadro no medible;
    ``warnings`` dice por que, en ingles, para mostrarse tal cual.

    ``markers`` son las ``MarkerDetection`` que ``detect_markers`` ya
    calculo este mismo cuadro (encargo posterior al plan de 20 tareas): el
    pipeline las calculaba y las tiraba al salir de ``process``, asi que
    ``draw_marker_overlay`` (Task 17) solo podia dibujarse en pantallas que
    vuelven a llamar a ``detect_markers`` por su cuenta (``calibration_view``)
    -- nunca en la vista de video en vivo, que es donde la spec (S13) dice
    que el video se publica primero. Se exponen SIEMPRE, incluso en el
    cuadro ``BLOCKED`` de abajo: ahi es exactamente donde el operador (y el
    espectador del video) necesita ver CUALES marcadores siguen visibles,
    no solo que la calibracion fallo.
    """

    rectified: np.ndarray | None
    raw_undistorted: np.ndarray
    plane: PlaneCalibration | None
    results: list[BananaResult]
    warnings: list[str]
    markers: list[MarkerDetection]


class Pipeline:
    """Procesa cuadros sueltos y mantiene el estado que cruza cuadros.

    El estado que persiste entre cuadros es el minimo: el acumulador temporal
    por ``track_id`` y la lista de instancias del cuadro anterior (que
    necesita ``CentroidReassigner``). El resto -- marcadores, homografia,
    lienzo -- se recalcula cada cuadro, igual que decidio Task 5: los cuatro
    marcadores estan siempre a la vista, recalcular no cuesta nada, y el
    sistema se autorrepara si alguien golpea el soporte de la camara.

    ``correct_parallax`` permite apagar la correccion de paralaje de Task 9.
    No es una comodidad de pruebas: la spec (S7 etapa 7, S13) pide poder
    mostrar el antes/despues de esa correccion, que es la mas grande de la
    cadena (~6.5% sobre un banano real). Apagarla tambien es lo unico
    honesto sobre una escena PLANA, donde la altura del objeto es cero y el
    factor no corrige nada: solo resta.
    """

    def __init__(
        self,
        cfg: AppConfig,
        intrinsics: Intrinsics,
        detector,
        density_g_per_cm3: float | None = None,
        *,
        correct_parallax: bool = True,
    ) -> None:
        self.cfg = cfg
        self.intrinsics = intrinsics
        self.detector = detector
        self.density_g_per_cm3 = density_g_per_cm3
        self.correct_parallax = correct_parallax

        self._canvas_size_px = canvas_size_for(cfg)
        self._reassigner = CentroidReassigner(
            REASSIGN_MAX_DISTANCE_MM / cfg.canvas_mm_per_px
        )
        self._aggregator = MeasurementAggregator(window=cfg.temporal_window)
        self._previous_instances: list[Instance] = []

    def process(self, frame: np.ndarray) -> FrameResult:
        """Procesa un cuadro entero y devuelve su ``FrameResult``.

        No lanza por nada que un cuadro en vivo pueda traer (marcadores
        tapados, fruta pegada, mascara degenerada, fruta fuera del area
        calibrada): todo eso sale como aviso. Si lanza
        ``DegenerateMarkerLayout`` (config colineal) o ``IntrinsicsMismatch``
        (resolucion distinta de la calibrada) -- los dos, errores de
        configuracion que no se arreglan en el cuadro siguiente.
        """
        warnings: list[str] = []

        raw_undistorted = undistort(frame, self.intrinsics)
        detections = detect_markers(raw_undistorted, self.cfg)

        # ``DegenerateMarkerLayout`` se propaga a proposito -- ver docstring
        # del modulo.
        plane = compute_plane_calibration(detections, self.cfg)
        if plane is None:
            warnings.append(
                f"Calibration blocked: only {len(detections)}/{MARKER_COUNT} ArUco markers "
                "are visible, which is not enough geometry to measure. Clear whatever is "
                "covering them."
            )
            return FrameResult(
                rectified=None,
                raw_undistorted=raw_undistorted,
                plane=None,
                results=[],
                warnings=warnings,
                markers=detections,
            )

        if plane.state is CalibrationState.DEGRADED:
            warnings.append(
                f"Degraded calibration: {len(detections)}/{MARKER_COUNT} ArUco markers "
                f"visible. Still measuring, on "
                f"{len(detections) * _CORNERS_PER_MARKER} corner correspondences instead "
                f"of {MARKER_COUNT * _CORNERS_PER_MARKER}."
            )

        rectified = rectify(raw_undistorted, plane, self._canvas_size_px)

        # El detector corre sobre el LIENZO, nunca sobre el cuadro crudo:
        # ``refine_mask`` usa estas mascaras como recorte dentro de este
        # mismo lienzo (ver docstring del modulo).
        instances = self._reassigner.assign(
            self.detector.detect(rectified), self._previous_instances
        )
        # Solo se refresca la memoria de centroides si este cuadro produjo
        # ALGUNA instancia. Un cuadro sin detecciones (fruta tapada un
        # instante, una oclusion) no es evidencia de que la fruta se fue --
        # es exactamente el hueco que ``CentroidReassigner`` existe para
        # puentear (spec S7 etapa 4). Sobrescribir con la lista vacia
        # tiraria esa memoria justo en el hueco que debia cubrir: el
        # siguiente cuadro bueno no tendria nada contra que reasociarse y
        # YOLO le asignaria ids nuevos, lo que en la practica reinicia el
        # acumulador temporal (Ruling 7) por un mecanismo distinto al que
        # esa regla nombra explicitamente (``forget()``), pero con el mismo
        # efecto observable que prohibe.
        #
        # Los otros dos caminos que retornan temprano no necesitan el mismo
        # cuidado, por razones distintas: un cuadro ``BLOCKED`` retorna
        # ANTES de llegar aqui, asi que nunca pisa la memoria (la deja tal
        # cual la dejo el ultimo cuadro medible). Un cuadro con fruta
        # adyacente SI llega aqui con ``instances`` no vacio -- hay
        # detecciones, solo que demasiado juntas para medir por separado --
        # asi que la condicion de abajo ya las conserva igual que a
        # cualquier otro cuadro medible.
        if instances:
            self._previous_instances = instances

        adjacent_pair = _first_adjacent_pair(instances)
        if adjacent_pair is not None:
            first, second = adjacent_pair
            warnings.append(
                f"Fruits {first} and {second} are closer than "
                f"{ADJACENCY_DILATE_PX * self.cfg.canvas_mm_per_px:.1f} mm, so they cannot "
                "be measured separately. Separate them on the tray."
            )
            return FrameResult(
                rectified=rectified,
                raw_undistorted=raw_undistorted,
                plane=plane,
                results=[],
                warnings=warnings,
                markers=detections,
            )

        results: list[BananaResult] = []
        for instance in instances:
            refined = refine_mask(rectified, instance.mask)
            if refined is None:
                warnings.append(
                    f"Fruit {instance.track_id}: the outline could not be refined, so it "
                    "was not measured on this frame."
                )
                continue

            measures = measure_mask(refined, plane.mm_per_px, self.cfg)
            if measures is None:
                warnings.append(
                    f"Fruit {instance.track_id}: its outline touches the border of the "
                    "calibrated area, or is too degenerate to measure, so it was not "
                    "measured on this frame."
                )
                continue

            if self.correct_parallax:
                measures = correct_measures(measures, self.cfg)

            # El agregado se alimenta de la medida YA corregida: el veredicto
            # y el peso se leen del agregado, asi que corregir despues de
            # acumular dejaria la ventana en unidades crudas.
            aggregate = self._aggregator.push(instance.track_id, measures)
            results.append(
                BananaResult(
                    track_id=instance.track_id,
                    aggregate=aggregate,
                    verdict=evaluate(aggregate.length_mm, aggregate.caliper_mm),
                    estimated_weight_g=self._weight_g(aggregate.volume_mm3),
                    mask=refined,
                    contour_px=measures.contour_px,
                    axis_path_px=measures.axis_path_px,
                    axis_tips_px=measures.axis_tips_px,
                    outer_curve_px=measures.outer_curve_px,
                )
            )

        return FrameResult(
            rectified=rectified,
            raw_undistorted=raw_undistorted,
            plane=plane,
            results=results,
            warnings=warnings,
            markers=detections,
        )

    def _weight_g(self, volume_mm3: float) -> float | None:
        """Peso estimado en gramos, o ``None`` si no se configuro una densidad.

        La densidad llega en g/cm3 y el volumen en mm3, de ahi la division
        por ``_MM3_PER_CM3``. Sin densidad no se inventa un valor por
        defecto: una densidad supuesta produciria un peso con apariencia de
        medicion.
        """
        if self.density_g_per_cm3 is None:
            return None
        return volume_mm3 / _MM3_PER_CM3 * self.density_g_per_cm3


def _first_adjacent_pair(instances: list[Instance]) -> tuple[int, int] | None:
    """Primer par de instancias mas cercano que ``ADJACENCY_DILATE_PX``.

    Devuelve sus ``track_id`` (para nombrarlos en el aviso) o ``None`` si
    todas estan suficientemente separadas. Se compara sobre las mascaras
    CRUDAS del detector, antes de refinar: el refinamiento es justamente lo
    que fusionaria el par, asi que preguntarselo despues seria tarde.

    Basta el PRIMER par: si dos frutos estan pegados, el cuadro entero deja
    de ser medible -- ``refine_mask`` toma el componente conexo mas grande
    del recorte, asi que la quimera fusionada puede contaminar la medida de
    cualquiera de los dos, no solo de uno.
    """
    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            if masks_are_adjacent(
                instances[i].mask, instances[j].mask, ADJACENCY_DILATE_PX
            ):
                return instances[i].track_id, instances[j].track_id
    return None

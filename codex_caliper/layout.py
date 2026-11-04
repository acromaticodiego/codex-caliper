"""Derivacion del ``marker_layout`` a partir de un solo frame del rig.

Que problema resuelve
---------------------
``cfg.marker_layout`` declara donde esta el centro de cada ArUco en
milimetros, y de el cuelga toda la metrica: una homografia ajustada contra
un layout equivocado rectifica a un plano equivocado y no lanza nada. El
unico camino que tenia el proyecto para obtenerlo era medir los cuatro
centros con metro, y hay que repetirlo cada vez que un marcador se mueve.

Por que NO se deriva con una escala unica
-----------------------------------------
La derivacion evidente -- ``mm_per_px`` desde el lado medio del marcador y
los centros en milimetros como los centros en pixeles por esa escala, que es
lo que calcula ``tools/aruco_probe.py`` -- supone que el plano se ve con una
escala constante, es decir, sin perspectiva. En este rig eso es falso de
forma medible: sobre un frame real, el lado del mismo marcador de 24 mm va
de 112.9 a 122.2 px segun donde caiga en el cuadro, un 8% de dispersion. Con
esa escala unica, el marcador mas lejano queda colocado mas cerca de lo que
esta.

Por que NO se deriva desde un solo marcador de referencia
---------------------------------------------------------
Tambien parece razonable tomar UN marcador, ajustar la homografia exacta de
su cuadrado de 24 mm conocido, y proyectar los demas por ella. MEDIDO sobre
el frame real del rig: el lado de los otros marcadores vuelve entre 11.0 y
21.7 mm en vez de 24.0, y el layout cambia mas de 100 mm segun cual se tome
de referencia. La razon es que un cuadrado de ~118 px no observa el termino
proyectivo a 1000 px de distancia: extrapolar desde el amplifica todo.

Como se deriva aqui
-------------------
Se ajustan a la vez las 8 incognitas de la homografia imagen->milimetros
exigiendo que **los cuatro** marcadores vuelvan como cuadrados del lado
declarado. Esa condicion -- cuatro cuadrados de tamano conocido repartidos
por el plano -- es la que fija el escorzo, y de paso fija la escala metrica
sin conocer ninguna distancia del mundo. Los centros salen despues, como
centroide de cada cuadrado ya en milimetros.

La condicion se compara contra el cuadrado GIRADO como este pegado cada
marcador (``mounting_quarter_turns``), no contra uno siempre alineado a los
ejes: ignorarlo fuerza un giro global de compromiso que deforma el layout
entero. Es el mismo defecto que ``world_points_for`` arrastraba.

Limite honesto
--------------
El modelo es una homografia pura, asi que no puede absorber la distorsion
del objetivo. Mientras los intrinsecos sean el marcador de posicion
(``rms_px = 0.0``, distorsion cero), el ``square_rms_mm`` que devuelve esta
funcion incluye esa distorsion y no baja de ella por bien montado que este
el rig. Medido sobre el rig real: 0.73 mm.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
from scipy.optimize import least_squares

from codex_caliper.aruco_tools import MarkerDetection
from codex_caliper.config import AppConfig
from codex_caliper.homography import mounting_quarter_turns

#: Marcadores que exige la derivacion. Con 3 quedan 12 esquinas y la
#: condicion de cuadrado deja el escorzo mal observado; el layout saldria
#: plausible y equivocado, que es justo lo que este modulo existe para no
#: producir.
_REQUIRED_MARKERS = 4


@dataclasses.dataclass(frozen=True)
class DerivedLayout:
    """Layout derivado de un frame, con la evidencia de cuanto cuadra.

    ``square_rms_mm`` no es decorativo: es lo unico que le dice al operador
    si puede confiar en ``centres_mm``. ``marker_sides_mm`` lo desglosa por
    marcador -- un solo marcador con el lado muy lejos de
    ``cfg.marker_length_mm`` senala a ESE marcador (despegado, doblado, mal
    detectado) en vez de dejar un RMS global sin culpable.
    """

    centres_mm: dict[int, tuple[float, float]]
    square_rms_mm: float
    marker_sides_mm: dict[int, float]


def _normalisation_for(image_points: np.ndarray) -> np.ndarray:
    """Matriz de acondicionamiento de Hartley: centra y escala los puntos de
    imagen antes del ajuste.

    Sin esto los parametros de la homografia viven en ordenes de magnitud muy
    distintos (los proyectivos son ~1e-4 frente a ~1 de los lineales) y el
    optimizador avanza mal en las direcciones pequenas.
    """
    centre = image_points.mean(axis=0)
    spread = float(image_points.std())
    return np.array(
        [[1.0 / spread, 0.0, -centre[0] / spread],
         [0.0, 1.0 / spread, -centre[1] / spread],
         [0.0, 0.0, 1.0]]
    )


def _expected_squares(detections: list[MarkerDetection], half_mm: float) -> np.ndarray:
    """El cuadrado que debe formar cada marcador, en el orden en que OpenCV
    entrega SUS esquinas -- girado segun como este pegado."""
    square = np.array([[-half_mm, -half_mm], [half_mm, -half_mm],
                       [half_mm, half_mm], [-half_mm, half_mm]])
    return np.stack(
        [
            np.array([square[(j + mounting_quarter_turns(d.corners)) % 4] for j in range(4)])
            for d in detections
        ]
    )


def _initial_homography(normalised: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Homografia de arranque, derivada SOLO de la imagen.

    Se toma el marcador mas central del cuadro y se resuelve la homografia
    exacta de sus 4 esquinas contra su cuadrado conocido. Como respuesta
    final eso es malo -- extrapolar desde un cuadrado pequeno deja el
    escorzo sin observar, y el lado de los demas marcadores vuelve entre 11
    y 22 mm sobre el rig real -- pero como ARRANQUE es justo lo que hace
    falta: escala y giro correctos en su entorno, y solo el termino
    proyectivo por ajustar, que es lo que el optimizador resuelve despues
    con los cuatro marcadores a la vez.

    No se arranca desde ``cfg.marker_layout`` a proposito: un layout
    declarado muy lejos del fisico -- justo el caso que esta funcion existe
    para reparar -- deja al optimizador en un minimo local y devuelve una
    geometria equivocada con buena pinta. Se eligio el marcador mas central
    porque es el que menos extrapola.
    """
    centres = normalised.mean(axis=1)
    reference = int(np.argmin(np.linalg.norm(centres - centres.mean(axis=0), axis=1)))
    homography = cv2.getPerspectiveTransform(
        normalised[reference].astype(np.float32), expected[reference].astype(np.float32)
    )
    return homography / homography[2, 2]


def derive_marker_layout(detections: list[MarkerDetection], cfg: AppConfig) -> DerivedLayout:
    """Deriva el centro en milimetros de cada marcador a partir de un frame.

    ``cfg.marker_layout`` NO se lee: seria circular leer el layout para
    derivar el layout, y en el caso que importa -- el declarado esta muy
    lejos del fisico -- arrancar desde el deja al optimizador en un minimo
    local. De ``cfg`` solo se usa ``marker_length_mm``, que es lo que fija
    la escala metrica, y ``marker_ids`` no hace falta porque los ids salen
    de las propias detecciones.

    Los centros se devuelven trasladados para que el menor x y el menor y
    sean 0, igual que un layout escrito a mano. La traslacion es libre (una
    homografia del plano no la observa) y cualquier eleccion es valida: se
    fija una para que el resultado sea reproducible.

    Lanza ``ValueError`` si no hay exactamente los 4 marcadores del rig.
    """
    if len(detections) != _REQUIRED_MARKERS:
        raise ValueError(
            f"Deriving a marker layout needs all {_REQUIRED_MARKERS} markers in one frame, "
            f"got {len(detections)}. Clear whatever is covering them and try again."
        )

    ordered = sorted(detections, key=lambda d: d.id)
    image_corners = np.stack([np.asarray(d.corners, dtype=np.float64) for d in ordered])
    expected = _expected_squares(ordered, cfg.marker_length_mm / 2.0)

    normalisation = _normalisation_for(image_corners.reshape(-1, 2))
    normalised = cv2.perspectiveTransform(
        image_corners.reshape(-1, 1, 2), normalisation
    ).reshape(len(ordered), 4, 2)

    def back_project(parameters: np.ndarray) -> np.ndarray:
        homography = np.append(parameters, 1.0).reshape(3, 3)
        return cv2.perspectiveTransform(
            normalised.reshape(-1, 1, 2), homography
        ).reshape(len(ordered), 4, 2)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        corners_mm = back_project(parameters)
        centred = corners_mm - corners_mm.mean(axis=1, keepdims=True)
        return (centred - expected).ravel()

    initial = _initial_homography(normalised, expected)

    solution = least_squares(
        residuals, initial.ravel()[:8], method="lm", xtol=1e-15, ftol=1e-15, max_nfev=40000
    )

    corners_mm = back_project(solution.x)
    centres = corners_mm.mean(axis=1)
    centres = centres - centres.min(axis=0)

    sides = {
        d.id: float(
            np.mean([np.linalg.norm(corners_mm[k][j] - corners_mm[k][(j + 1) % 4]) for j in range(4)])
        )
        for k, d in enumerate(ordered)
    }
    return DerivedLayout(
        centres_mm={d.id: (float(centres[k][0]), float(centres[k][1])) for k, d in enumerate(ordered)},
        square_rms_mm=float(np.sqrt(np.mean(solution.fun**2))),
        marker_sides_mm=sides,
    )

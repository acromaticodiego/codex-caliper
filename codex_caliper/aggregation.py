"""Agregacion temporal de medidas y su incertidumbre (Task 11).

Un solo frame rara vez es confiable: el contorno tiembla un par de pixeles
y la medida salta. Con una identidad persistente por fruto (``track_id``,
que entrega el tracker) se pueden acumular varias medidas del MISMO banano
y colapsarlas en un numero estable.

Tres decisiones, no una sola:

1. Se reporta la MEDIANA, no la media. La cadena de medicion (Task 8)
   devuelve ``None`` para frames no medibles, pero sigue produciendo de
   vez en cuando un frame malo-pero-valido (mascara rota, oclusion
   parcial); un solo frame asi arrastra una media, mientras que la
   mediana lo descarta sin esfuerzo. Robustez frente a atipicos es el
   requisito, no una optimizacion.
2. El ruido cae aproximadamente como la raiz de N: una ventana de 30
   frames compra ~5x de reduccion frente a un solo frame.
3. La desviacion estandar sobre la ventana ES la incertidumbre de la
   medida, y se reporta como tal (``length_std_mm``, ``caliper_std_mm``).
   El objetivo explicito del proyecto es nunca reportar una cifra sin
   saber que tan equivocada puede estar.

``locked`` exige las DOS condiciones a la vez: la ventana llena (``samples
== window``) Y la desviacion de la longitud por debajo de ``lock_std_mm``.
Ninguna de las dos basta por si sola -- una ventana llena puede seguir
siendo ruidosa, y una ventana silenciosa pero a medio llenar todavia puede
cambiar mucho con la siguiente muestra.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import Deque, Dict, List

import numpy as np

from codex_caliper.geometry.measures import BananaMeasures

#: Tamano por defecto de la ventana deslizante, en frames. Constante de
#: proyecto (no literal disperso): a 30 fps equivale a aproximadamente un
#: segundo de video.
TEMPORAL_WINDOW = 30


@dataclasses.dataclass(frozen=True)
class TrackAggregate:
    """Resumen estadistico de un ``track_id`` sobre su ventana actual.

    ``length_mm``, ``caliper_mm`` y ``volume_mm3`` son la mediana de la
    ventana; ``length_std_mm`` y ``caliper_std_mm`` son su desviacion
    estandar, reportada como la incertidumbre de la medida (no un dato de
    diagnostico interno).
    """

    track_id: int
    samples: int
    locked: bool
    length_mm: float
    length_std_mm: float
    caliper_mm: float
    caliper_std_mm: float
    volume_mm3: float


class MeasurementAggregator:
    """Acumula medidas por ``track_id`` y las colapsa en un ``TrackAggregate``.

    Mantiene un ``deque`` de tamano maximo ``window`` por cada ``track_id``
    visto hasta ahora; al llegar una muestra nueva con la ventana llena, la
    mas vieja se descarta automaticamente (comportamiento nativo de
    ``deque(maxlen=...)``).
    """

    def __init__(self, window: int = TEMPORAL_WINDOW, lock_std_mm: float = 1.0) -> None:
        self.window = window
        self.lock_std_mm = lock_std_mm
        self._buffers: Dict[int, Deque[BananaMeasures]] = {}

    def push(self, track_id: int, measures: BananaMeasures) -> TrackAggregate:
        """Agrega una medida nueva del fruto ``track_id`` y devuelve el agregado actualizado."""
        buffer = self._buffers.setdefault(track_id, deque(maxlen=self.window))
        buffer.append(measures)
        return self._summarize(track_id, buffer)

    def get(self, track_id: int) -> TrackAggregate | None:
        """Devuelve el agregado actual de ``track_id``, o ``None`` si no hay datos."""
        buffer = self._buffers.get(track_id)
        if not buffer:
            return None
        return self._summarize(track_id, buffer)

    def forget(self, track_id: int) -> None:
        """Descarta el buffer de ``track_id`` (el tracker perdio ese fruto de vista)."""
        self._buffers.pop(track_id, None)

    def active_ids(self) -> List[int]:
        """Lista los ``track_id`` que todavia tienen muestras acumuladas."""
        return list(self._buffers.keys())

    def _summarize(self, track_id: int, buffer: Deque[BananaMeasures]) -> TrackAggregate:
        lengths = np.array([m.codex_length_mm for m in buffer], dtype=float)
        calipers = np.array([m.caliper_mm for m in buffer], dtype=float)
        volumes = np.array([m.volume_mm3 for m in buffer], dtype=float)

        length_mm = float(np.median(lengths))
        length_std_mm = float(np.std(lengths))
        caliper_mm = float(np.median(calipers))
        caliper_std_mm = float(np.std(calipers))
        volume_mm3 = float(np.median(volumes))

        samples = len(buffer)
        window_full = samples == self.window
        locked = window_full and length_std_mm < self.lock_std_mm

        return TrackAggregate(
            track_id=track_id,
            samples=samples,
            locked=locked,
            length_mm=length_mm,
            length_std_mm=length_std_mm,
            caliper_mm=caliper_mm,
            caliper_std_mm=caliper_std_mm,
            volume_mm3=volume_mm3,
        )

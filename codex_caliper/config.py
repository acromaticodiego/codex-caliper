"""Configuracion de la aplicacion, persistida en JSON.

``AppConfig`` es el dataclass que leen todas las tareas posteriores: medidas
fisicas del montaje (marcador, distancia de trabajo), parametros de
procesamiento (ventana temporal, margenes) y la identidad de los marcadores
ArUco usados para la calibracion metrica (diccionario, ids, layout).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass(kw_only=True)
class AppConfig:
    """Parametros fisicos y de procesamiento del montaje de medicion.

    Los marcadores reales del rig son ``DICT_6X6_50`` con ids ``[1, 2, 3, 4]``;
    esos son los valores por defecto de ``aruco_dictionary`` y ``marker_ids``.
    ``marker_layout`` (posicion en mm de cada marcador sobre el lienzo) no
    tiene un valor por defecto significativo y queda vacio hasta que se mide
    el montaje.
    """

    marker_length_mm: float = 24.0
    marker_thickness_mm: float
    working_distance_mm: float = 270.0
    canvas_mm_per_px: float = 0.2
    temporal_window: int = 30
    pedicel_drop_ratio: float = 0.45
    border_margin_px: int = 2
    aruco_dictionary: str = "DICT_6X6_50"
    marker_ids: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 3, 4])
    marker_layout: dict[int, tuple[float, float]] = dataclasses.field(default_factory=dict)

    @property
    def effective_height_mm(self) -> float:
        """Distancia de trabajo neta, tras restar el espesor del marcador."""
        return self.working_distance_mm - self.marker_thickness_mm

    def save(self, path: str | Path) -> None:
        """Serializa la config a JSON.

        ``marker_layout`` usa claves ``int`` en Python, pero JSON solo admite
        claves ``str``; se convierten explicitamente a string al guardar.
        """
        data = dataclasses.asdict(self)
        data["marker_layout"] = {str(k): list(v) for k, v in self.marker_layout.items()}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        """Carga la config desde JSON, revirtiendo las claves de ``marker_layout`` a ``int``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_layout = data.get("marker_layout", {})
        data["marker_layout"] = {int(k): tuple(v) for k, v in raw_layout.items()}
        return cls(**data)

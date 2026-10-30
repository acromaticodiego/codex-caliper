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


#: Cantidad de coordenadas que describen la posicion de un marcador: (x, y)
#: en milimetros sobre el plano del lienzo. El layout es PLANO por
#: definicion (spec S5): una tercera coordenada no seria una posicion mas
#: precisa, seria otro modelo de montaje.
_LAYOUT_COORDINATES = 2


def _layout_entry(key: object, value: object) -> tuple[float, float]:
    """Valida UNA entrada de ``marker_layout`` y la devuelve como ``(x, y)``.

    Una entrada valida es una secuencia de exactamente dos numeros. Se
    rechaza explicitamente ``str`` y ``bytes`` antes de medir su longitud:
    ``"215,118"`` ES una secuencia de 7 elementos, asi que una comprobacion
    de longitud por si sola la aceptaria como si fuera una pareja de
    coordenadas y produciria ``('2', '1')``. Y los ``bool`` se rechazan como
    numeros porque ``isinstance(True, int)`` es ``True`` en Python: un
    ``true`` en el JSON se convertiria en la coordenada 1.0 sin decir nada.

    Lanza ``ValueError`` -- el tipo que ``_load_config`` ya atrapa en
    ``__main__.py`` -- y nombra la clave, que es el dato con el que el
    operador encuentra la linea que tiene que corregir.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(
            f"marker_layout entry {key!r} must be a list of two numbers [x, y] in "
            f"millimetres, got {type(value).__name__}"
        )
    if len(value) != _LAYOUT_COORDINATES:
        raise ValueError(
            f"marker_layout entry {key!r} must have exactly {_LAYOUT_COORDINATES} "
            f"coordinates [x, y], got {len(value)}"
        )
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError(
                f"marker_layout entry {key!r} must contain numbers, got "
                f"{coordinate!r} ({type(coordinate).__name__})"
            )
    x, y = value
    return float(x), float(y)


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
        """Carga la config desde JSON, revirtiendo las claves de ``marker_layout`` a ``int``.

        Valida la FORMA de lo leido antes de indexar en ella (ronda de fix 3,
        tras un hallazgo del re-revisor): un JSON sintacticamente valido pero
        con la forma equivocada -- la raiz no es un objeto, o
        ``marker_layout`` es una lista en vez de un objeto, ambos deslices de
        edicion a mano realistas, no entradas contrived -- hacia que
        ``data.get(...)``/``raw_layout.items()`` lanzaran ``AttributeError``.
        Ese tipo no estaba en la tupla de ``_load_config`` (``__main__.py``)
        y escapaba como traceback crudo -- justo lo que la Ruling 5 de ese
        modulo prohibe. La correccion NO es agregar ``AttributeError`` a esa
        tupla: eso volveria a tapar un bug real dentro de este metodo, que
        es exactamente lo que esa ruling existe para evitar. En cambio, este
        metodo valida lo que va a indexar y lo convierte en ``ValueError``
        -- que la tupla ya atrapa -- para que "nunca un traceback" se
        cumpla por construccion y no por enumerar tipos de excepcion.

        Se validan las ENTRADAS, no solo los contenedores (ronda de
        correcciones finales, I6). La ronda anterior comprobaba que la raiz
        fuera un objeto y que ``marker_layout`` fuera un objeto, y nada mas:
        ``{"1": {"x": 0, "y": 0}}`` -- el mismo genero de desliz de edicion
        a mano que los dos casos ya cerrados -- cargaba limpio como
        ``{1: ('x', 'y')}``, y el ``TypeError`` aparecia despues, dentro de
        ``canvas_size_for`` en el constructor del ``Pipeline``. Ese
        constructor vive en el ``try`` de ``main()`` cuyo unico manejador es
        ``except _StartupError``, asi que escapaba como traceback crudo.
        ``{"3": "215,118"}`` hacia lo mismo por otro camino (una cadena de 6
        caracteres no es una pareja de numeros). Validar la forma de cada
        VALOR es lo que cierra la clase entera, en vez de los dos ejemplos
        conocidos.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(
                f"expected a JSON object at the top level, got {type(data).__name__}"
            )
        raw_layout = data.get("marker_layout", {})
        if not isinstance(raw_layout, dict):
            raise ValueError(
                "marker_layout must be a JSON object mapping marker id to [x, y], got "
                f"{type(raw_layout).__name__}"
            )
        data["marker_layout"] = {
            int(key): _layout_entry(key, value) for key, value in raw_layout.items()
        }
        return cls(**data)

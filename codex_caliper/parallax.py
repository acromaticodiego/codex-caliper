"""Correccion de paralaje (Task 9).

El plano de calibracion es la mesa, pero un banano apoyado sobre ella no es
plano: tiene un grosor ``h`` (el calibre) y la silueta que la camara ve
desde arriba corresponde, aproximadamente, a la seccion mas ancha, que esta
a MEDIA altura del objeto (``h/2``) por encima de la mesa, no sobre la
mesa misma. Medir como si el objeto estuviera en el plano de calibracion
sobreestima toda magnitud lineal en la escena en un factor
``H_ef / (H_ef - h/2)``, donde ``H_ef`` (``AppConfig.effective_height_mm``)
es la distancia de trabajo neta (altura de la camara sobre la mesa, menos
el espesor del marcador: el plano de calibracion es la cara superior del
marcador impreso en 3D, no la mesa).

Con un banano de ~35 mm de calibre a ~267 mm de altura efectiva eso es un
~6.5% -- unos 12 mm sobre un banano de 190 mm, mas que todo el resto de la
cadena de error junto, contra un objetivo declarado de +-1 mm. Es la
correccion que mas importa de la fase.

``scale_factor`` devuelve el factor INVERSO (<= 1 para un objeto con
``h > 0``), que es el que hay que APLICAR a una magnitud medida en el plano
de calibracion para llevarla a su valor real:

    factor = (H_ef - h/2) / H_ef
    valor_real = valor_medido * factor

La correccion es EXACTA, no iterativa
-------------------------------------
La dificultad del problema es que ``h`` -- la altura real del objeto -- es
justo lo que el sistema recien termino de medir (el calibre), y ese calibre
crudo esta a su vez inflado por el mismo paralaje que se quiere corregir.
El factor depende de la altura y la altura depende del factor: la
correccion se refiere a si misma.

Esa autorreferencia NO necesita iterarse: cierra algebraicamente. La
condicion de consistencia es que la altura corregida ``h`` sea el calibre
crudo ``c`` reescalado por el factor que esa misma altura produce,
``h = c * (H_ef - h/2) / H_ef``. Es LINEAL en ``h``, asi que se despeja en
dos lineas:

    h * H_ef + c * h/2 = c * H_ef   ->   h = c * H_ef / (H_ef + c/2)

y sustituyendo ese ``h`` de vuelta en ``factor = 1 - h / (2 * H_ef)``:

    factor = 2 * H_ef / (2 * H_ef + c)

Esa es la forma cerrada que usa ``correct_measures``: una sola expresion,
sin pasadas, sin parametro que ajustar. Es el punto fijo EXACTO de la
iteracion -- no una aproximacion de el (ver
``tests/test_parallax.py::test_closed_form_factor_is_the_fixed_point``:
el residuo de la ecuacion de punto fijo es 0.0 y el valor coincide bit a
bit con el limite de la iteracion).

Por que no se dejo la version iterativa
---------------------------------------
El plan original decia "iterar dos veces, converge". No converge en dos:
OSCILA alrededor del punto fijo y se acerca por lados alternados. Sobre los
valores de referencia (``H_ef = 267``, calibre crudo 36, longitud cruda
284.905):

    pasada 1 -> factor 0.93258427, longitud 265.698
    pasada 2 -> factor 0.93712915, longitud 266.993  (de mas)
    pasada 3 -> factor 0.93682275, longitud 266.905  (de menos)
    pasada 6 -> factor 0.93684211, longitud 266.911
    limite   -> factor 0.93684211, longitud 266.910981

Dos pasadas se pasan del limite por 0.082 mm -- un 8% del presupuesto de
error total de +-1 mm, gastado en nada. Subir el contador de iteraciones
seria tratar un sintoma: la forma cerrada da el limite de una vez, asi que
el parametro ``iterations`` desaparecio del todo (un parametro que nadie
puede variar con sentido es peso muerto).

Que magnitudes se reescalan y con que exponente
-------------------------------------------------
``codex_length_mm`` y ``caliper_mm`` son longitudes: escalan con el factor
a la primera potencia. ``area_mm2`` escala al cuadrado, ``volume_mm3`` al
cubo -- son, por construccion (Task 8), integrales del mismo perfil de
grosor bajo el supuesto de seccion circular, asi que escalan junto con
``caliper_mm`` y preservan la consistencia que esa tarea fijo con un test a
``rel=1e-6``: escalar area y volumen con el MISMO factor que la longitud y
el calibre (cada uno a su propio exponente) es lo unico que no rompe esa
relacion, porque la relacion en si es un enunciado puramente geometrico
sobre longitudes, areas y volumenes de la MISMA figura reescalada.

``width_profile`` -- CRUDO, no se reescala
--------------------------------------------
``width_profile`` (la tupla ``(s_mm, w_mm)`` del eje medial completo, Task
6/8) NO se incluye entre las magnitudes que ``correct_measures`` reescala,
deliberadamente. Dos razones:

1. Es el perfil COMPLETO (sin recortar el pedicelo), mientras que el
   factor de paralaje se estima a partir del calibre del FRUTO; reescalar
   ``w_mm`` entero mezclaria una correccion pensada para el fruto con el
   pedicelo, que tiene su propia geometria (y probablemente otra altura
   real).
2. Mas de fondo: la abscisa ``s_mm`` que entrega ``width_profile_mm``
   (Task 6) esta corregida del sesgo de escalera SOLO en total (+0.42%
   contra la longitud analitica); su residuo LOCAL llega a +-2% (+2% en el
   40% inicial del camino, -0.6% en la mitad final -- ver revision de Task
   8). Esa correccion de escalera sirve como PESO de integracion (donde se
   cancela), pero no como coordenada puntual fiable para ubicar una
   caracteristica. Reescalar ``s_mm`` por el factor de paralaje la seguiria
   tratando como si fuera una coordenada puntual exacta, cuando no lo es;
   y como ningun consumidor de ``width_profile`` necesita esa abscisa ya
   corregida de paralaje (las cuatro magnitudes que SI lo necesitan ya
   salen corregidas por separado), no hay necesidad real de tocarla. Se
   deja tal cual sale de ``measure_mask``, documentado como sin corregir.
"""

from __future__ import annotations

import dataclasses

from codex_caliper.config import AppConfig
from codex_caliper.geometry.measures import BananaMeasures


def scale_factor(effective_height_mm: float, object_height_mm: float) -> float:
    """Factor de correccion de paralaje para un objeto de altura ``object_height_mm``.

    ``factor = (H_ef - h/2) / H_ef``: la silueta vista desde arriba
    corresponde a la seccion mas ancha del objeto, que esta a MEDIA altura
    (``h/2``) sobre el plano de calibracion, no sobre el plano mismo. Un
    objeto plano (``h = 0``) no necesita correccion (factor = 1.0).

    Ojo con el argumento: ``object_height_mm`` es la altura REAL (ya
    corregida) del objeto, no el calibre crudo. Quien tenga a mano solo el
    calibre crudo debe usar ``correct_measures``, que resuelve esa
    autorreferencia en forma cerrada (ver docstring del modulo).

    Degenera con seguridad (devuelve 1.0, sin corregir) si ``effective_height_mm``
    no es positivo -- un montaje mal configurado (espesor de marcador mayor
    o igual a la distancia de trabajo) no debe producir un factor negativo
    o una division por cero a mitad de un cuadro.
    """
    if effective_height_mm <= 0.0:
        return 1.0
    h = max(float(object_height_mm), 0.0)
    return (effective_height_mm - h / 2.0) / effective_height_mm


def correct_measures(
    measures: BananaMeasures | None,
    cfg: AppConfig,
) -> BananaMeasures | None:
    """Corrige las magnitudes crudas de ``measures`` por paralaje, en forma cerrada.

    El calibre crudo (``measures.caliper_mm``, medido como si el banano
    estuviera en el plano de calibracion) es la unica estimacion disponible
    de la altura real del objeto, y esta inflado por el mismo paralaje que
    se quiere corregir. Esa autorreferencia cierra algebraicamente: la
    altura consistente es ``h = c * H_ef / (H_ef + c/2)`` y el factor que le
    corresponde, el punto fijo EXACTO del esquema iterativo,

        factor = 2 * H_ef / (2 * H_ef + calibre_crudo)

    (derivacion de dos lineas y por que no se itera: docstring del modulo).

    El factor se aplica UNA vez a las cuatro magnitudes crudas: longitud y
    calibre a la primera potencia, area al cuadrado, volumen al cubo (ver
    docstring del modulo).

    ``width_profile`` se conserva SIN tocar -- ver docstring del modulo.

    Devuelve ``None`` sin lanzar si ``measures`` es ``None`` (mascara
    degenerada rio arriba, Task 8): una correccion aplicada a algo que no
    se pudo medir no es un error, es la misma ausencia de medida
    propagada, siguiendo el mismo contrato que el resto del pipeline.
    """
    if measures is None:
        return None

    effective_height_mm = cfg.effective_height_mm
    raw_caliper_mm = max(float(measures.caliper_mm), 0.0)

    if effective_height_mm <= 0.0:
        # Montaje mal medido (espesor de marcador >= distancia de trabajo):
        # misma degeneracion segura que scale_factor, sin corregir.
        factor = 1.0
    else:
        # Forma cerrada. El denominador es >= 2 * H_ef > 0, asi que no hay
        # division por cero ni factor negativo posibles.
        factor = 2.0 * effective_height_mm / (2.0 * effective_height_mm + raw_caliper_mm)

    return dataclasses.replace(
        measures,
        codex_length_mm=measures.codex_length_mm * factor,
        caliper_mm=measures.caliper_mm * factor,
        area_mm2=measures.area_mm2 * factor ** 2,
        volume_mm3=measures.volume_mm3 * factor ** 3,
    )

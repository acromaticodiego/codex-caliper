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

``width_profile`` -- se reescala en LOS DOS EJES
--------------------------------------------------
``width_profile`` (la tupla ``(s_mm, w_mm)`` del eje medial completo, Task
6/8) tambien se corrige: ``s_mm`` Y ``w_mm``, las dos por el MISMO factor a
la primera potencia. Las dos son longitudes medidas en el plano de
calibracion y las dos cargan la misma sobreestimacion de paralaje, por el
mismo motivo; corregir una sola, o ninguna, deja el perfil en unas unidades
distintas de las de los cuatro escalares.

Reescalar los dos ejes es lo que hace que las magnitudes derivadas salgan
solas, sin exponentes puestos a mano:

    integral(w ds)          escala con f**2  -> coincide con ``area_mm2``
    integral(pi*(w/2)**2 ds) escala con f**3  -> coincide con ``volume_mm3``

Es decir: el perfil corregido y los escalares corregidos son la MISMA
figura reescalada, y un consumidor que recalcule area o volumen integrando
el perfil sobre el tramo del fruto aterriza en los numeros que este modulo
reporta, no al lado. Esta fijado por
``tests/test_parallax.py::test_area_and_volume_recomputed_from_the_corrected_profile_match``.

Por que NO se deja crudo (era la decision anterior, y estaba mal)
------------------------------------------------------------------
Dejarlo crudo convertia ``BananaMeasures`` en una trampa silenciosa: un
consumidor que reprodujera ``trim_pedicel`` e integrara el ``w_mm`` crudo
sobre el tramo del fruto -- exactamente el calculo que ``_area_mm2`` y
``_volume_mm3`` ya hacen -- obtenia **+13.94% en area y +21.62% en
volumen** contra los valores corregidos que el mismo objeto reporta, sin
ningun aviso. Hoy no es alcanzable (``BananaMeasures`` no expone
``start``/``end`` y nada importa todavia este modulo), pero eso es una
trampa latente, no un default seguro.

Las dos razones que se habian dado no sostenian la decision:

1. El residuo LOCAL de +-2% del sesgo de escalera (+2% en el 40% inicial
   del camino, -0.6% en la mitad final -- ver revision de Task 8) es un
   problema de la ABSCISA ``s_mm``, y no dice nada sobre si los ANCHOS
   ``w_mm`` deben corregirse. Ademas es ortogonal al paralaje: multiplicar
   por una constante no cambia el tamano RELATIVO de ese residuo, asi que
   reescalar no lo mejora ni lo empeora. Sigue valiendo la advertencia de
   Task 8 -- ``s_mm`` sirve como PESO de integracion, donde ese residuo se
   cancela, y no como coordenada puntual exacta para ubicar una
   caracteristica -- solo que ahora en unidades corregidas.
2. Que el perfil incluya el pedicelo (y el factor se estime del calibre
   del FRUTO) es un argumento legitimo, pero solo contra reescalar el
   TRAMO DEL PEDICELO, no contra reescalar nada. Y en la practica no
   importa: el objeto entero esta a esencialmente la misma altura sobre la
   mesa, asi que un factor unico sobre todo el perfil es correcto a primer
   orden. El pedicelo es mas delgado y queda marginalmente mas bajo, pero
   se DESCARTA antes de reportar cualquier cosa, asi que perseguir un
   factor propio para el seria una precision que nadie consume.

Nota sobre el recorte, para quien reescale esto de nuevo
---------------------------------------------------------
El criterio de ``trim_pedicel`` compara la derivada normalizada
``|dw/ds| / w_max``, que tiene unidades de 1/mm, contra un umbral ABSOLUTO
(``_DERIVATIVE_THRESHOLD_PER_MM = 0.04``). Bajo un reescalado uniforme de
los dos ejes, ``dw/ds`` queda invariante pero ``w_max`` escala con ``f``,
asi que la cantidad comparada escala con ``1/f``: sobre el perfil corregido
crece un ~6.7%. El umbral sigue bracketeando los dos casos de la
calibracion de Task 8 (adelgazamiento gradual 0.0307 -> ~0.0328, pedicelo
0.0508 -> ~0.0542, umbral 0.04) y el recorte cae en los MISMOS indices
(verificado: ``(0, 449)`` sobre el banano de referencia, crudo y
corregido), pero el margen del lado gradual -- el ajustado de los dos -- se
angosta de 0.0093 a ~0.0072 1/mm. Quien recorte a partir del perfil
CORREGIDO debe saberlo; ``measure_mask`` recorta sobre el CRUDO, antes de
que este modulo intervenga, asi que las magnitudes reportadas no dependen
de ese margen.
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

    ``width_profile`` se reescala en LOS DOS EJES (``s_mm`` y ``w_mm``, las
    dos a la primera potencia), para que el perfil quede en las mismas
    unidades corregidas que los escalares y recalcular area o volumen a
    partir de el de los mismos numeros -- ver docstring del modulo.

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

    # Los dos ejes del perfil a la primera potencia. La multiplicacion
    # preserva el dtype de cada arreglo (``s_mm`` float64, ``w_mm``
    # float32), asi que el perfil corregido sigue ocupando lo mismo.
    raw_s_mm, raw_w_mm = measures.width_profile

    return dataclasses.replace(
        measures,
        codex_length_mm=measures.codex_length_mm * factor,
        caliper_mm=measures.caliper_mm * factor,
        area_mm2=measures.area_mm2 * factor ** 2,
        volume_mm3=measures.volume_mm3 * factor ** 3,
        width_profile=(raw_s_mm * factor, raw_w_mm * factor),
    )

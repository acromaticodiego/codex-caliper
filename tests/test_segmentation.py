"""Pruebas de Task 13: refinamiento de contorno por canal de saturacion.

Las cuatro pruebas del Task Brief original no discriminan (ver ruling de
Task 13): cada una se puede pasar con una implementacion incorrecta
concreta. Cada prueba de aqui esta escrita para que una de esas
implementaciones incorrectas la haga fallar -- la evidencia de que lo
logra (quiebre deliberado, corrida, reversion) va en el reporte, no aqui,
pero el diseño de cada prueba ya apunta a esa implementacion incorrecta
especifica (se indica en cada docstring).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from codex_caliper.segmentation import refine_mask
from tests.fixtures.synthetic import make_banana, make_banana_scene

_MM_PER_PX = 0.2
_YELLOW_BGR = (0, 216, 255)

# Banano de referencia ya usado en tests/test_synthetic_fixture.py y
# tests/test_measures.py -- sin pedicelo ni adelgazamiento floral, para que
# el area de pixeles (y la de ``refine_mask``) sea comparable DIRECTAMENTE
# contra ``GroundTruth.area_mm2`` (que describe la capsula pura; pedicelo y
# adelgazamiento floral la desvian del conteo de pixeles -- ver docstring de
# ``tests/fixtures/synthetic.GroundTruth``).
_REF_BANANA = {"radius_mm": 220.0, "arc_rad": 0.85, "width_mm": 36.0}

# Tolerancia relativa sobre el area analitica. Incluye el ~1% de sesgo de
# rasterizado del propio fixture (ver test_synthetic_fixture.py) mas el
# efecto de los kernels morfologicos de refine_mask (3-9 px de lado contra
# un banano de ~180 px de calibre -- un orden de magnitud mas chico).
_AREA_TOLERANCE_REL = 0.03


def _area_mm2(mask: np.ndarray, mm_per_px: float = _MM_PER_PX) -> float:
    return float(np.count_nonzero(mask)) * mm_per_px**2


def _num_components(mask: np.ndarray) -> int:
    num_labels, _ = cv2.connectedComponents(np.asarray(mask, dtype=np.uint8))
    return num_labels - 1  # label 0 es el fondo.


def _shadow_region_of(scene: np.ndarray) -> np.ndarray:
    """Mascara booleana de los pixeles de sombra (gris neutro, ni blanco ni
    casi negro) en una escena sintetica -- misma logica que
    ``tests/test_synthetic_fixture.py::shadow_region_of``, repetida aqui
    para no acoplar este archivo a otro modulo de test.
    """
    b = scene[..., 0].astype(np.int16)
    g = scene[..., 1].astype(np.int16)
    r = scene[..., 2].astype(np.int16)
    is_neutral = (np.abs(b - g) <= 2) & (np.abs(g - r) <= 2) & (np.abs(b - r) <= 2)
    is_mid_tone = (r < 250) & (r > 30)
    return is_neutral & is_mid_tone


def _true_banana_region_of(scene: np.ndarray) -> np.ndarray:
    """Mascara booleana EXACTA del banano en una escena sintetica, por
    igualdad de color contra ``_YELLOW_BGR``.

    Independiente de ``refine_mask``: no usa saturacion ni ningun umbral, es
    una igualdad de color directa contra el amarillo solido con el que
    ``make_banana_scene`` pinta la fruta (sin antialiasing -- el estampado
    de la escena es binario, ver ``_stamp_masked`` en el fixture). Sirve de
    verdad de referencia en coordenadas de la ESCENA, que ``GroundTruth`` no
    da (esa esta en coordenadas locales del banano aislado).
    """
    return np.all(scene == np.array(_YELLOW_BGR, dtype=scene.dtype), axis=-1)


def _overlap_fraction(mask: np.ndarray, region: np.ndarray) -> float:
    """Fraccion de ``region`` que cae dentro de ``mask``. ``0.0`` si ``region``
    esta vacia (nada que pueda filtrarse)."""
    region_count = int(np.count_nonzero(region))
    if region_count == 0:
        return 0.0
    return float(np.count_nonzero(mask & region)) / region_count


# ---------------------------------------------------------------------------
# (a) Sombra excluida -- Y banano recuperado.
# ---------------------------------------------------------------------------


def test_shadow_on_white_background_is_excluded_and_banana_is_recovered():
    """Quiebre que esta prueba debe atrapar: umbralar por VALOR/LUMINANCIA
    (canal V de HSV, o escala de grises) en vez de SATURACION. Sobre la
    sombra (oscura, saturacion ~0) un umbral de luminancia la clasifica
    como "oscuro = parte del objeto" y la mete en la mascara -- la razon de
    ser de este modulo entero (ver docstring de ``codex_caliper.segmentation``).

    La version del Task Brief solo pide "overlap(refined, shadow) < 0.05",
    que una mascara VACIA pasa trivialmente (overlap de una region vacia con
    cualquier otra region es 0). Por eso se exige TAMBIEN que el banano se
    recupere dentro de una tolerancia absoluta sobre el area analitica --
    una implementacion que devuelve casi nada ya no puede pasar las dos
    aserciones a la vez.
    """
    scene = make_banana_scene([_REF_BANANA], _MM_PER_PX)
    _, gt = make_banana(mm_per_px=_MM_PER_PX, **_REF_BANANA)

    true_banana = _true_banana_region_of(scene)
    shadow = _shadow_region_of(scene)
    assert shadow.any(), "el fixture debe producir una sombra real para que esta prueba tenga sentido"

    # Mascara "gruesa" tipo YOLO que, por error, incluyo la sombra propia del
    # banano (un fallo real documentado en S5.1.c: sombra + patas del
    # soporte son oscuras pero desaturadas, y a veces terminan en la mascara
    # del detector).
    coarse_mask_including_shadow = true_banana | shadow

    refined = refine_mask(scene, coarse_mask_including_shadow)
    assert refined is not None

    shadow_overlap = _overlap_fraction(refined, shadow)
    assert shadow_overlap < 0.05

    refined_area = _area_mm2(refined)
    assert refined_area == pytest.approx(gt.area_mm2, rel=_AREA_TOLERANCE_REL)


# ---------------------------------------------------------------------------
# (b) Area refinada mas cerca de la verdad que la mascara gruesa -- Y dentro
# de una tolerancia absoluta.
# ---------------------------------------------------------------------------


def test_refined_area_is_closer_to_truth_than_the_coarse_mask_and_within_absolute_tolerance():
    """Quiebre que esta prueba debe atrapar: refinar SIN el margen de
    recorte del bbox (Ruling 3) -- es decir, recortar exactamente al bbox
    de ``coarse_mask`` sin ``_BBOX_MARGIN_PX``. La version comparativa pura
    del Task Brief se puede pasar con una mascara gruesa arbitrariamente
    mala (erosionarla hasta casi nada hace que CUALQUIER area refinada
    quede "mas cerca"); por eso se fija tambien un limite ABSOLUTO.
    """
    scene = make_banana_scene([_REF_BANANA], _MM_PER_PX)
    _, gt = make_banana(mm_per_px=_MM_PER_PX, **_REF_BANANA)
    true_banana = _true_banana_region_of(scene)

    # Mascara gruesa degradada: erosion uniforme, como una mascara YOLO que
    # sistematicamente recorta el fruto hacia dentro. El radio de erosion
    # (20 px = 4 mm) es menor que el margen de recorte de refine_mask
    # (60 px = 12 mm): si el margen no se aplicara, el recorte heredaria el
    # mismo recorte que la mascara gruesa.
    erosion_kernel = np.ones((41, 41), np.uint8)  # radio ~20 px
    coarse_mask = cv2.erode(true_banana.astype(np.uint8), erosion_kernel).astype(bool)
    assert coarse_mask.any(), "la erosion no debe borrar toda la mascara gruesa"

    refined = refine_mask(scene, coarse_mask)
    assert refined is not None

    coarse_area = _area_mm2(coarse_mask)
    refined_area = _area_mm2(refined)

    coarse_error = abs(coarse_area - gt.area_mm2)
    refined_error = abs(refined_area - gt.area_mm2)

    assert refined_error < coarse_error
    assert refined_area == pytest.approx(gt.area_mm2, rel=_AREA_TOLERANCE_REL)


# ---------------------------------------------------------------------------
# Ruling 3 dedicada: mascara gruesa SUBDIMENSIONADA (recorta parte real de
# la fruta, no solo erosionada) igual recupera el area real gracias al
# margen de recorte.
# ---------------------------------------------------------------------------


def test_undersized_coarse_mask_still_recovers_true_area_via_bbox_padding():
    """Quiebre que esta prueba debe atrapar: el mismo que la prueba (b) --
    recortar exactamente al bbox de ``coarse_mask``, sin margen. Aqui la
    mascara gruesa no esta erosionada de forma pareja: esta CORTADA en un
    extremo (como si YOLO hubiera perdido la punta floral del fruto), asi
    que su propio bbox es mas chico que el bbox real del banano en una sola
    direccion -- el caso mas directo de "la caja de YOLO no llega a cubrir
    toda la fruta real" que describe el Ruling 3.
    """
    scene = make_banana_scene([_REF_BANANA], _MM_PER_PX)
    _, gt = make_banana(mm_per_px=_MM_PER_PX, **_REF_BANANA)
    true_banana = _true_banana_region_of(scene)

    ys, xs = np.nonzero(true_banana)
    x_max = int(xs.max())
    # Recorta 80 px (16 mm) del extremo derecho del bbox real -- MAS que el
    # margen de refine_mask (60 px = 12 mm). Elegido asi a proposito, no por
    # error: el extremo floral es un casquete que se afina hacia la punta,
    # asi que el area perdida en esa franja externa es chica (ver medicion
    # en el reporte) y el margen SI alcanza para recuperarla casi entera
    # (dentro de la tolerancia relativa). Sin margen (recorte exacto al
    # bbox de la mascara gruesa), el area queda pegada a la de la mascara
    # gruesa -- bien afuera de la tolerancia (ver evidencia de quiebre en
    # el reporte: 0.9432 vs 0.97 de piso).
    clip_from = x_max - 80 + 1
    coarse_mask = true_banana.copy()
    coarse_mask[:, clip_from:] = False
    assert coarse_mask.any()
    assert _area_mm2(coarse_mask) < _area_mm2(true_banana) * 0.96, "el recorte debe quitar area real de forma medible"

    refined = refine_mask(scene, coarse_mask)
    assert refined is not None
    assert _area_mm2(refined) == pytest.approx(gt.area_mm2, rel=_AREA_TOLERANCE_REL)


# ---------------------------------------------------------------------------
# (c) Solo sobrevive el componente conexo mayor -- y ES el banano.
# ---------------------------------------------------------------------------


def test_keeps_only_the_largest_connected_component_and_it_is_the_banana():
    """Quiebre que esta prueba debe atrapar: quedarse con el PRIMER
    componente etiquetado (orden de barrido de ``cv2.connectedComponents``,
    arriba-izquierda primero) en vez del MAS GRANDE. La version del Task
    Brief solo cuenta "== 1" componente, que pasa igual si ese unico
    componente sobreviviente es la mota en vez de la fruta; aqui se verifica
    que el area del componente sobreviviente coincide con la del banano, no
    con la de la mota.

    La mota se dibuja arriba-izquierda del banano (que queda mas abajo y a
    la derecha en el recorte) para que, en un barrido top-left-first, la
    mota reciba la etiqueta 1 y el banano la etiqueta 2 -- asi, una
    implementacion que se queda con "el primer componente" en vez de "el
    mas grande" produce un resultado DISTINTO y detectable (la mota, no el
    banano), no el mismo resultado por otra via.
    """
    scene = make_banana_scene([_REF_BANANA], _MM_PER_PX)
    _, gt = make_banana(mm_per_px=_MM_PER_PX, **_REF_BANANA)
    true_banana = _true_banana_region_of(scene)

    ys, xs = np.nonzero(true_banana)
    # Mota saturada (10x10 px = 2x2 mm), chica frente al banano pero mayor
    # que el kernel de apertura (3 px) -- sobrevive la limpieza morfologica
    # como componente propio. Se ubica DENTRO del bbox+margen de refine_mask
    # (para que quede en el mismo recorte que el banano) pero separada de el,
    # arriba-izquierda.
    speckle_y0 = max(0, int(ys.min()) - 40)
    speckle_x0 = max(0, int(xs.min()) - 5)
    scene_with_speckle = scene.copy()
    scene_with_speckle[speckle_y0 : speckle_y0 + 10, speckle_x0 : speckle_x0 + 10] = (0, 0, 255)  # rojo saturado

    coarse_mask = true_banana.copy()
    # La caja gruesa tambien "ve" la zona de la mota (simula una caja YOLO
    # generosa que incluye zona de fondo cerca de la fruta).
    coarse_mask[speckle_y0 : speckle_y0 + 10, speckle_x0 : speckle_x0 + 10] = True

    refined = refine_mask(scene_with_speckle, coarse_mask)
    assert refined is not None
    assert _num_components(refined) == 1
    assert _area_mm2(refined) == pytest.approx(gt.area_mm2, rel=_AREA_TOLERANCE_REL)


# ---------------------------------------------------------------------------
# Cierre morfologico: una mancha chica desaturada ADENTRO del fruto (analogo
# sintetico de una mancha de madurez, S5.1.d) se recupera; no queda como
# agujero.
# ---------------------------------------------------------------------------


def test_small_desaturated_blemish_inside_the_fruit_is_closed_not_left_as_a_hole():
    """Quiebre que esta prueba debe atrapar: quitar el paso de CIERRE
    morfologico. Una mancha oscura de un banano muy maduro cae a saturacion
    45-61, por debajo del umbral operativo (~83 en el rig real) que separa
    fruta de fondo (S5.1.d) -- sin cierre, esa mancha queda como agujero
    (o corta la mascara) en vez de contarse como fruta. El fixture sintetico
    no modela manchas de madurez; esta prueba pincha un parche desaturado
    (B=G=R, saturacion 0, igual que la sombra) de 3 px de radio DENTRO del
    banano, mas chico que el kernel de cierre (9 px), para que sea el
    analogo mas directo que el fixture permite reproducir.
    """
    scene = make_banana_scene([_REF_BANANA], _MM_PER_PX)
    true_banana = _true_banana_region_of(scene)
    ys, xs = np.nonzero(true_banana)
    center_y = (int(ys.min()) + int(ys.max())) // 2
    center_x = (int(xs.min()) + int(xs.max())) // 2
    assert true_banana[center_y, center_x], "el centro del bbox debe caer dentro del fruto"

    blemished_scene = scene.copy()
    cv2.circle(blemished_scene, (center_x, center_y), 3, (180, 180, 180), -1)

    refined = refine_mask(blemished_scene, true_banana)
    assert refined is not None
    assert bool(refined[center_y, center_x]), "la mancha debe quedar CERRADA (rellena), no como agujero"
    assert _num_components(refined) == 1


# ---------------------------------------------------------------------------
# (d) Nada saturado -> None, incluso con ruido que Otsu podria partir en dos.
# ---------------------------------------------------------------------------


def test_returns_none_when_nothing_saturated_is_found_even_with_sensor_noise():
    """Quiebre que esta prueba debe atrapar: quitar el piso ABSOLUTO de
    saturacion y confiar solo en Otsu (Ruling 2). Una escena perfectamente
    blanca y plana ya devuelve ``None`` incluso sin piso (Otsu no tiene nada
    que partir); por eso esta prueba inyecta ruido sintetico de sensor (una
    perturbacion aleatoria chica por canal) que SI le da a Otsu una
    distribucion bimodal-ish para partir en dos clases no triviales -- sin
    el piso absoluto, Otsu devuelve una mascara con la FORMA del ruido en
    vez de ``None``.
    """
    rng = np.random.default_rng(13)
    scene = make_banana_scene([], _MM_PER_PX)
    height, width = scene.shape[:2]
    if height == 0 or width == 0:
        scene = np.full((200, 200, 3), 255, np.uint8)
        height, width = scene.shape[:2]

    # Ruido de sensor sintetico: +-6 por canal alrededor de 255, saturacion
    # resultante acotada muy por debajo del piso (ver calculo en el reporte:
    # maximo realista ~12, piso en 20).
    noise = rng.integers(-6, 7, size=(height, width, 3), endpoint=False)
    noisy_scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(noisy_scene, cv2.COLOR_BGR2HSV)
    max_saturation = int(hsv[..., 1].max())
    assert max_saturation < 20, f"el ruido sintetico debe quedar bajo el piso absoluto (fue {max_saturation})"

    coarse_mask = np.zeros((height, width), dtype=bool)
    coarse_mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = True

    assert refine_mask(noisy_scene, coarse_mask) is None


# ---------------------------------------------------------------------------
# ``saturation_threshold`` explicito reemplaza a Otsu, no se ignora.
# ---------------------------------------------------------------------------


def test_explicit_saturation_threshold_is_used_instead_of_otsu():
    """Cubre la rama de ``saturation_threshold`` explicito, que ninguna otra
    prueba de este archivo ejercita (todas usan el ``None`` por defecto,
    Otsu). Un umbral por encima del maximo posible de saturacion (255) no
    puede dejar pasar ningun pixel; si la implementacion IGNORARA el
    argumento y siguiera calculando Otsu internamente, devolveria el banano
    igual que ``test_refined_area_is_closer_to_truth_...`` -- esta prueba
    falla en ese caso, no solo si el umbral se aplica mal.
    """
    scene = make_banana_scene([_REF_BANANA], _MM_PER_PX)
    true_banana = _true_banana_region_of(scene)

    assert refine_mask(scene, true_banana, saturation_threshold=260) is None
    assert refine_mask(scene, true_banana) is not None  # el default (Otsu) si encuentra el banano

# Banana Codex Vision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Medir bananos en vivo sobre una base calibrada con 4 ArUcos y reportar los dos parámetros de la norma CODEX STAN 205-1997 (longitud por curva exterior y calibre), más área, volumen, peso estimado e incertidumbre.

**Architecture:** Detección neuronal + medición clásica. YOLOv8n-seg con tracking dice *qué* y *dónde*; toda la geometría se mide con visión clásica sobre un lienzo cenital rectificado por homografía, recalculada en cada frame desde los 4 marcadores. El núcleo geométrico son funciones puras probadas contra un generador de bananos sintéticos con verdad analítica exacta.

**Tech Stack:** Python 3.11 · opencv-python · scikit-image · ultralytics · NumPy · SciPy · PySide6 · pandas · pytest

**Spec:** [docs/superpowers/specs/2026-10-23-banano-codex-vision-design.md](../specs/2026-10-23-banano-codex-vision-design.md)

## Global Constraints

- `MARKER_LENGTH_MM = 24.0` — lado del cuadrado negro completo, igual al `markerLength` de OpenCV.
- `CODEX_MIN_LENGTH_MM = 140.0`, `CODEX_MIN_CALIPER_MM = 27.0`.
- Grados comerciales de calibre Cavendish: 34 mm (mínimo comercial) a 42+ mm.
- `CANVAS_MM_PER_PX = 0.2` — escala fija del lienzo rectificado.
- `TEMPORAL_WINDOW = 30` frames por `track_id`.
- `INTRINSICS_MAX_RMS_PX = 0.5` — umbral de aceptación de la calibración intrínseca.
- `YOLO_BANANA_CLASS_ID = 46` (COCO), modelo `yolov8n-seg.pt`, inferencia sobre frame reducido a 640 px.
- **Diccionario ArUco: `DICT_6X6_50`, ids `[1, 2, 3, 4]`** — medido en el montaje real.
- **`perspectiveRemovePixelPerCell = 8`** (el defecto de OpenCV es 4). Sin esto el id 3 no se decodifica en ningún frame. No sustituir por `errorCorrectionRate`: ver Task 4.
- Distancia de trabajo `H` y espesor de marcador `t` son **configurables**, nunca constantes en el código. Valores iniciales: `H = 270.0`, `t` por medir.
- Altura efectiva: `H_ef = H - t`.
- Cámara: ELP-USBFHD01M-BFV, 1920×1080 @ 30 fps MJPEG. `cv2.CAP_DSHOW` en Windows.
- **Todos los textos visibles de la interfaz en inglés.** Comentarios y documentación en español.
- Panel lateral de la UI: **240 px fijos**, nunca porcentaje.
- Toda medición ocurre en el lienzo rectificado, jamás en la imagen cruda.
- Nunca reportar una medida sin calibración válida: antes "sin dato" que un número falso.

## Review Focus

Modos de fallo que el diseño implica y que cada uno recibe su test en la tarea indicada:

1. **Menos de 4 marcadores visibles** — con 3 debe seguir midiendo y señalarlo; con menos de 3 debe bloquear la medición, no extrapolar. → Task 5.
2. **Máscaras de dos bananos fusionadas o adyacentes** — debe rehusar medir y pedir separarlos, nunca reportar un objeto fundido. → Task 12 (detección de adyacencia) y Task 14 (el pipeline rehusa).
3. **Banano cortado por el borde del lienzo** — una máscara que toca el borde está incompleta; su longitud es falsa y debe rechazarse. → Task 8.
4. **Máscara degenerada** (muy pequeña, o esqueleto sin dos extremos distintos) — debe devolver "no medible" sin lanzar excepción. → Task 7.
5. **Intrínsecos ausentes o de otra posición de zoom** — debe rehusar entrar en modo medición y decir por qué, en vez de medir con parámetros equivocados. → Task 4.

## Fuera del alcance de este plan

El spec §9 ofrece dos caminos para fijar las posiciones de los marcadores. Este plan
implementa solo el **preferido**: plantilla impresa con posiciones conocidas, declaradas en
`cfg.marker_layout`. El **auto-levantamiento** (derivar la disposición desde un marcador y
guardarla) queda sin implementar; si el montaje real hace imposible colocar los marcadores
con precisión, es una tarea a añadir, no un ajuste.

---

## File Structure

```
codex_caliper/
  __init__.py
  config.py             AppConfig, MarkerLayout, carga/guardado JSON
  intrinsics.py         calibración intrínseca, persistencia, validación de zoom
  aruco_tools.py        identificación de diccionario, detección de marcadores
  homography.py         homografía plano↔imagen, residual, rectificación
  detection.py          wrapper YOLO seg + tracking + reasociación por centroide
  segmentation.py       refinamiento de contorno por canal de saturación
  geometry/
    __init__.py
    medial.py           eje medial, poda, camino principal, perfil de grosor
    outline.py          curva exterior, longitud de arco corregida
    measures.py         recorte de pedicelo, calibre, área, volumen
  parallax.py           corrección iterativa por altura
  aggregation.py        buffers temporales por track_id, mediana e incertidumbre
  codex.py              veredicto Codex y grado comercial
  pipeline.py           orquestación por frame
  camera.py             hilo de captura
  validation.py         registro CSV, regresión de densidad, métricas de error
  ui/
    main_window.py      ventana principal, layout variante 1
    video_view.py       lienzo rectificado + overlay + PiP
    side_panel.py       tarjetas de medición de 240 px
    calibration_view.py pantallas de calibración intrínseca y de plano
    validation_view.py  tabla de validación
    overlay.py          dibujo de cotas sobre el frame
tools/
  find_aruco_dictionary.py
tests/
  fixtures/synthetic.py  generador de bananos sintéticos con verdad analítica
  test_*.py
```

---

### Task 1: Scaffold, configuración y buscador de diccionario ArUco

**Files:**
- Create: `pyproject.toml`, `codex_caliper/__init__.py`, `codex_caliper/config.py`, `tools/find_aruco_dictionary.py`
- Test: `tests/test_config.py`, `tests/test_find_dictionary.py`

**Interfaces:**
- Produces: `AppConfig` (dataclass con `marker_length_mm: float = 24.0`, `marker_thickness_mm: float`, `working_distance_mm: float = 270.0`, `canvas_mm_per_px: float = 0.2`, `temporal_window: int = 30`, `pedicel_drop_ratio: float = 0.45`, `border_margin_px: int = 2`, `aruco_dictionary: str`, `marker_ids: list[int]`, `marker_layout: dict[int, tuple[float, float]]`), `AppConfig.load(path) -> AppConfig`, `AppConfig.save(path) -> None`, `AppConfig.effective_height_mm -> float`
- Produces: `find_dictionary(image: np.ndarray, expected_count: int = 4) -> list[DictionaryMatch]` en `tools/find_aruco_dictionary.py`, donde `DictionaryMatch` tiene `name: str`, `ids: list[int]`, `count: int`

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_config.py
def test_effective_height_subtracts_marker_thickness():
    cfg = AppConfig(marker_thickness_mm=3.0, working_distance_mm=270.0)
    assert cfg.effective_height_mm == 267.0

def test_roundtrip_preserves_marker_layout(tmp_path):
    cfg = AppConfig(marker_thickness_mm=2.5, marker_layout={0: (0.0, 0.0), 1: (300.0, 0.0)})
    cfg.save(tmp_path / "c.json")
    assert AppConfig.load(tmp_path / "c.json").marker_layout == cfg.marker_layout

# tests/test_find_dictionary.py
def test_identifies_the_dictionary_a_marker_was_drawn_with():
    # dibujar 4 marcadores DICT_5X5_50 (ids 0..3) sobre lienzo blanco con cv2.aruco.generateImageMarker
    matches = find_dictionary(canvas, expected_count=4)
    assert matches[0].name == "DICT_5X5_50"
    assert sorted(matches[0].ids) == [0, 1, 2, 3]

def test_returns_empty_list_when_no_markers_present():
    assert find_dictionary(np.full((400, 400, 3), 255, np.uint8)) == []
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_config.py tests/test_find_dictionary.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Crear `pyproject.toml` con las dependencias del stack y declarar `codex_caliper` como paquete**

Dependencias: `opencv-python`, `numpy`, `scipy`, `scikit-image`, `ultralytics`, `PySide6`, `pandas`. De desarrollo: `pytest` y **`pytest-qt`** (las Tasks 16, 18 y 19 usan la fixture `qtbot`).

- [ ] **Step 4: Implementar `AppConfig` en `codex_caliper/config.py`**

Dataclass con los campos del bloque Interfaces, serializada a JSON. Las claves de `marker_layout` son enteros y JSON solo admite strings: convertir en ambos sentidos explícitamente.

- [ ] **Step 5: Implementar `find_dictionary` en `tools/find_aruco_dictionary.py`**

Recorrer todos los `cv2.aruco.DICT_*` predefinidos, detectar con `cv2.aruco.ArucoDetector`, devolver los que produzcan detecciones ordenados por cercanía a `expected_count` y luego por número de marcadores. Añadir un `main()` que acepte la ruta de una imagen e imprima el informe.

- [ ] **Step 6: Correr los tests**

Run: `pytest tests/test_config.py tests/test_find_dictionary.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml codex_caliper/ tools/ tests/
git commit -m "feat: project scaffold, config and ArUco dictionary finder"
```

---

### Task 2: Generador de bananos sintéticos con verdad analítica

Fixture de pruebas del que dependen todas las tareas de geometría. Un banano se modela como una **cápsula de ancho constante sobre un arco de circunferencia**, porque esa forma tiene longitud y volumen exactos en forma cerrada: no hay que confiar en ninguna medición para saber la respuesta correcta.

**Files:**
- Create: `tests/fixtures/__init__.py`, `tests/fixtures/synthetic.py`
- Test: `tests/test_synthetic_fixture.py`

**Interfaces:**
- Produces: `make_banana(radius_mm: float, arc_rad: float, width_mm: float, mm_per_px: float, pedicel_mm: float = 0.0, pedicel_width_mm: float = 0.0, blossom_taper_mm: float = 0.0, canvas_margin_mm: float = 20.0) -> tuple[np.ndarray, GroundTruth]`
  - `blossom_taper_mm` estrecha **gradualmente** el extremo opuesto al pedicelo a lo largo de esa longitud. Lo exige `test_gradual_blossom_taper_is_not_trimmed` de Task 8: el recorte debe distinguir la caída abrupta del pedicelo de la gradual del extremo floral. Con 0.0 el extremo queda redondeado como hasta ahora, y la verdad analítica no cambia.
- Produces: `GroundTruth` dataclass con `centerline_length_mm`, `outer_length_mm`, `caliber_mm`, `area_mm2`, `volume_mm3`
- Produces: `make_banana_scene(bananas: list[dict], mm_per_px: float, markers: dict[int, tuple[float, float]] | None = None, marker_dict: str = "DICT_6X6_50") -> np.ndarray` — imagen BGR, bananos amarillos sobre fondo blanco con sombra gris desaturada
  - Cuando `markers` no es `None`, dibuja un ArUco de 24 mm centrado en cada posición (mm) con `cv2.aruco.generateImageMarker`. Lo exige `test_end_to_end_on_a_synthetic_scene_recovers_ground_truth` de Task 14, que necesita una escena calibrable.

Verdad analítica de la cápsula sobre arco de radio `R`, ángulo `θ`, ancho `w`:

```
centerline_length = R·θ
outer_length      = (R + w/2)·θ
caliber           = w
area              = w·R·θ + π·(w/2)²
volume            = π·(w/2)²·R·θ + (4/3)·π·(w/2)³
```

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_synthetic_fixture.py
def test_ground_truth_matches_closed_form():
    _, gt = make_banana(radius_mm=220.0, arc_rad=0.85, width_mm=36.0, mm_per_px=0.2)
    assert gt.centerline_length_mm == pytest.approx(220.0 * 0.85)
    assert gt.outer_length_mm == pytest.approx((220.0 + 18.0) * 0.85)
    assert gt.caliber_mm == 36.0

def test_rendered_area_matches_analytic_area_within_1_percent():
    mask, gt = make_banana(radius_mm=220.0, arc_rad=0.85, width_mm=36.0, mm_per_px=0.2)
    rendered = mask.sum() / 255 * 0.2 ** 2
    assert rendered == pytest.approx(gt.area_mm2, rel=0.01)

def test_pedicel_adds_pixels_but_not_ground_truth():
    plain, gt_a = make_banana(220.0, 0.85, 36.0, 0.2)
    stemmed, gt_b = make_banana(220.0, 0.85, 36.0, 0.2, pedicel_mm=25.0, pedicel_width_mm=9.0)
    assert stemmed.sum() > plain.sum()
    assert gt_b.outer_length_mm == gt_a.outer_length_mm

def test_scene_shadow_is_desaturated():
    scene = make_banana_scene([{"radius_mm": 220.0, "arc_rad": 0.85, "width_mm": 36.0}], 0.2)
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    # la sombra debe tener saturación baja pese a ser oscura en luminancia
    assert hsv[..., 1][shadow_region_of(scene)].max() < 40
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_synthetic_fixture.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implementar `GroundTruth` y `make_banana` en `tests/fixtures/synthetic.py`**

Rasterizar la cápsula barriendo la línea media (`cv2.circle` de radio `w/2` en cada punto del arco muestreado finamente, o `cv2.polylines` con grosor). El pedicelo es un segundo barrido de radio menor que arranca en una punta y se aleja tangencialmente. Devolver máscara `uint8` 0/255.

- [ ] **Step 4: Implementar `make_banana_scene`**

Máscara → amarillo saturado en BGR sobre fondo blanco; añadir una sombra desplazada unos milímetros, gris neutro (igual B, G y R) para que quede oscura en luminancia pero desaturada.

- [ ] **Step 5: Correr los tests**

Run: `pytest tests/test_synthetic_fixture.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/ tests/test_synthetic_fixture.py
git commit -m "test: synthetic banana fixture with closed-form ground truth"
```

---

### Task 3: Calibración intrínseca

**Files:**
- Create: `codex_caliper/intrinsics.py`
- Test: `tests/test_intrinsics.py`

**Interfaces:**
- Produces: `Intrinsics` dataclass con `camera_matrix: np.ndarray`, `dist_coeffs: np.ndarray`, `rms_px: float`, `image_size: tuple[int, int]`, `zoom_note: str`
- Produces: `calibrate_from_corners(object_points, image_points, image_size) -> Intrinsics`
- Produces: `Intrinsics.save(path)`, `Intrinsics.load(path) -> Intrinsics`, `Intrinsics.is_acceptable -> bool` (RMS < 0.5)
- Produces: `undistort(frame, intr) -> np.ndarray`
- Produces: `find_chessboard(frame, pattern_size) -> np.ndarray | None` con refinamiento subpíxel vía `cv2.cornerSubPix`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_recovers_known_intrinsics_from_synthetic_views():
    # proyectar un tablero 9x6 conocido con cv2.projectPoints usando K y dist conocidos,
    # desde 15 poses variadas, y recalibrar
    intr = calibrate_from_corners(obj_pts, img_pts, (1920, 1080))
    assert intr.camera_matrix[0, 0] == pytest.approx(known_fx, rel=0.02)
    assert intr.dist_coeffs[0] == pytest.approx(known_k1, abs=0.02)

def test_is_acceptable_follows_the_half_pixel_threshold():
    assert Intrinsics(rms_px=0.21, ...).is_acceptable is True
    assert Intrinsics(rms_px=0.80, ...).is_acceptable is False

def test_roundtrip_preserves_matrix_and_zoom_note(tmp_path):
    ...

# Review Focus 5
def test_load_raises_when_image_size_differs_from_current_capture():
    intr = Intrinsics(image_size=(1280, 720), ...)
    with pytest.raises(IntrinsicsMismatch):
        intr.require_compatible((1920, 1080))
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_intrinsics.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `codex_caliper/intrinsics.py`**

`calibrate_from_corners` envuelve `cv2.calibrateCamera`. `require_compatible(size)` lanza `IntrinsicsMismatch` si la resolución no coincide — es la única verificación automática posible del cambio de zoom; el resto es el protocolo manual de la §5.2 del spec, que `zoom_note` registra en texto libre.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_intrinsics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/intrinsics.py tests/test_intrinsics.py
git commit -m "feat: intrinsic calibration with persistence and zoom guard"
```

---

### Task 4: Detección de marcadores y guarda de intrínsecos

**Files:**
- Create: `codex_caliper/aruco_tools.py`
- Test: `tests/test_aruco_tools.py`

**Interfaces:**
- Consumes: `AppConfig` (Task 1), `Intrinsics` (Task 3)
- Produces: `MarkerDetection` dataclass con `id: int`, `corners: np.ndarray` (4×2, orden de OpenCV)
- Produces: `detect_markers(frame, cfg) -> list[MarkerDetection]`
- Produces: `build_detector_params() -> cv2.aruco.DetectorParameters` con el ajuste para marcadores impresos en 3D
- Produces: `CalibrationState` enum: `OK` (4 marcadores), `DEGRADED` (3), `BLOCKED` (<3)
- Produces: `calibration_state(detections) -> CalibrationState`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_detects_four_markers_in_a_rendered_board():
    assert len(detect_markers(rendered_four_markers, cfg)) == 4

# Review Focus 1
@pytest.mark.parametrize("n,expected", [(4, CalibrationState.OK),
                                        (3, CalibrationState.DEGRADED),
                                        (2, CalibrationState.BLOCKED),
                                        (0, CalibrationState.BLOCKED)])
def test_calibration_state_degrades_then_blocks(n, expected):
    assert calibration_state([fake_detection(i) for i in range(n)]) == expected

def test_ignores_markers_whose_id_is_not_in_the_configured_set():
    # render ids 0,1,2,3,9 con cfg.marker_ids = [0,1,2,3]
    assert {d.id for d in detect_markers(frame, cfg)} == {0, 1, 2, 3}

def test_cell_sampling_is_raised_above_the_opencv_default():
    # el defecto de 4 px/celda deja indecodificable el id 3 del montaje real:
    # una linea de capa de la impresion FDM domina una muestra de 16 px
    assert build_detector_params().perspectiveRemovePixelPerCell >= 8

def test_all_four_real_markers_decode_in_every_capture():
    # regresion contra capturas/probe_*.png del montaje real
    for frame in real_captures():
        assert {d.id for d in detect_markers(frame, cfg)} == {1, 2, 3, 4}
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_aruco_tools.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `codex_caliper/aruco_tools.py`**

`build_detector_params` ajusta `adaptiveThreshWinSizeMin/Max`, `minMarkerPerimeterRate`, fija **`perspectiveRemovePixelPerCell = 8`** y activa `CORNER_REFINE_SUBPIX`. El refinamiento subpíxel sostiene el objetivo de ±1 mm; el muestreo por celda es lo que hace decodificable el marcador id 3. `detect_markers` filtra por `cfg.marker_ids`.

No sustituir ese ajuste por `errorCorrectionRate = 0.8`, que también hace aparecer el marcador: esa palanca no mejora la muestra, acepta más bits erróneos, y con ~50 cuadriláteros espurios por frame procedentes de la veta de la superficie invita a falsos positivos.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_aruco_tools.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/aruco_tools.py tests/test_aruco_tools.py
git commit -m "feat: ArUco detection tuned for 3D-printed markers"
```

---

### Task 5: Homografía del plano y rectificación

**Files:**
- Create: `codex_caliper/homography.py`
- Test: `tests/test_homography.py`

**Interfaces:**
- Consumes: `MarkerDetection`, `CalibrationState` (Task 4), `AppConfig` (Task 1)
- Produces: `PlaneCalibration` dataclass con `H: np.ndarray` (3×3, imagen→lienzo), `residual_mm: float`, `state: CalibrationState`, `mm_per_px: float`
- Produces: `compute_plane_calibration(detections, cfg) -> PlaneCalibration | None` — devuelve `None` si `state is BLOCKED`
- Produces: `world_points_for(detections, cfg) -> np.ndarray` — 4 esquinas por marcador a partir de `cfg.marker_layout` y `MARKER_LENGTH_MM`
- Produces: `rectify(frame, plane: PlaneCalibration, canvas_size_px) -> np.ndarray`
- Produces: `canvas_size_for(cfg) -> tuple[int, int]`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_recovers_scale_from_a_known_synthetic_homography():
    # construir H conocida, proyectar las 16 esquinas de los 4 marcadores, recuperarla
    plane = compute_plane_calibration(projected_detections, cfg)
    assert plane.mm_per_px == pytest.approx(0.2, rel=1e-6)
    assert plane.residual_mm < 0.05

def test_measures_the_same_length_at_canvas_centre_and_corner():
    # un segmento de 100 mm colocado en el centro y en la esquina del plano debe
    # medir lo mismo tras rectificar — esta es la razón de ser de la homografía
    assert length_at_centre == pytest.approx(length_at_corner, abs=0.3)

def test_residual_grows_when_a_marker_is_misplaced():
    # desplazar un marcador 5 mm respecto al layout declarado
    assert compute_plane_calibration(shifted, cfg).residual_mm > 1.0

# Review Focus 1
def test_returns_none_when_fewer_than_three_markers():
    assert compute_plane_calibration(two_detections, cfg) is None

def test_degrades_to_twelve_points_with_three_markers():
    plane = compute_plane_calibration(three_detections, cfg)
    assert plane.state == CalibrationState.DEGRADED
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_homography.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `codex_caliper/homography.py`**

`world_points_for` genera, para cada marcador detectado, sus 4 esquinas en milímetros a partir del centro declarado en `cfg.marker_layout` y de `MARKER_LENGTH_MM`, en el mismo orden que devuelve OpenCV. `compute_plane_calibration` llama a `cv2.findHomography` con `method=0` (todos los puntos son de confianza; no hay outliers que descartar con RANSAC) y calcula `residual_mm` reproyectando los puntos del mundo y midiendo el RMS en milímetros. `rectify` envuelve `cv2.warpPerspective`.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_homography.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/homography.py tests/test_homography.py
git commit -m "feat: per-frame plane homography with residual and graceful degradation"
```

---

### Task 6: Eje medial, poda y perfil de grosor

**Files:**
- Create: `codex_caliper/geometry/__init__.py` (vacío — Tasks 7 y 8 importan de los submódulos, no reexportan), `codex_caliper/geometry/medial.py`
- Test: `tests/test_medial.py`

**Interfaces:**
- Consumes: `make_banana` (Task 2)
- Produces: `MedialAxis` dataclass con `path_px: np.ndarray` (N×2, ordenada de punta a punta), `radius_px: np.ndarray` (N,), `tips_px: tuple[np.ndarray, np.ndarray]`
- Produces: `medial_axis_path(mask: np.ndarray) -> MedialAxis | None` — `None` si la máscara es degenerada
- Produces: `width_profile_mm(axis: MedialAxis, mm_per_px: float) -> tuple[np.ndarray, np.ndarray]` — abscisa curvilínea `s` y ancho `w(s)`, ambos en mm
- Produces: `arc_length_px(path_px: np.ndarray) -> float`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_tips_land_near_the_ends_of_the_synthetic_banana():
    mask, gt = make_banana(220.0, 0.85, 36.0, 0.2)
    axis = medial_axis_path(mask)
    assert distance_between(axis.tips_px, expected_tips_px) < 5  # px

def test_centerline_length_matches_ground_truth_within_2_percent():
    mask, gt = make_banana(220.0, 0.85, 36.0, 0.2)
    axis = medial_axis_path(mask)
    length = arc_length_px(axis.path_px) * 0.2
    assert length == pytest.approx(gt.centerline_length_mm, rel=0.02)

def test_width_profile_is_flat_for_a_constant_width_capsule():
    s, w = width_profile_mm(medial_axis_path(mask), 0.2)
    middle = w[len(w) // 4: 3 * len(w) // 4]
    assert middle.std() < 0.5
    assert middle.mean() == pytest.approx(36.0, abs=0.6)

def test_spurs_do_not_lengthen_the_path():
    # una máscara con una protuberancia pequeña genera ramas; el camino principal
    # debe seguir midiendo lo mismo
    assert length_with_bump == pytest.approx(length_plain, rel=0.02)

# Review Focus 4
@pytest.mark.parametrize("mask", [np.zeros((50, 50), np.uint8), tiny_blob(), perfect_disc()])
def test_returns_none_for_degenerate_masks(mask):
    assert medial_axis_path(mask) is None
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_medial.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `medial_axis_path` en `codex_caliper/geometry/medial.py`**

Usar `skimage.morphology.medial_axis(mask, return_distance=True)`: devuelve esqueleto y transformada de distancia en una sola llamada, y el valor de distancia en cada píxel del esqueleto es el radio del círculo inscrito, o sea medio ancho local.

Para el camino principal, construir un grafo de los píxeles del esqueleto (vecindad 8, peso 1 en ortogonal y √2 en diagonal) y tomar el **camino geodésico más largo**: BFS/Dijkstra desde un extremo cualquiera para hallar el nodo más lejano, repetir desde él para hallar el otro, y devolver el camino entre ambos. Eso poda las ramas espurias sin necesidad de reglas ad hoc. Devolver `None` si el esqueleto tiene menos de 2 píxeles o el camino es más corto que unos pocos píxeles.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_medial.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/geometry/ tests/test_medial.py
git commit -m "feat: medial axis with geodesic main path and width profile"
```

---

### Task 7: Curva exterior y longitud Codex con corrección de escalonado

**Files:**
- Create: `codex_caliper/geometry/outline.py`
- Test: `tests/test_outline.py`

**Interfaces:**
- Consumes: `MedialAxis` (Task 6)
- Produces: `split_contour_at_tips(contour_px, tips_px) -> tuple[np.ndarray, np.ndarray]`
- Produces: `outer_curve(contour_px, tips_px) -> np.ndarray` — la cadena más larga de las dos
- Produces: `smoothed_arc_length_mm(chain_px, mm_per_px, smoothing: float = 1.0) -> float`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_raw_pixel_chain_overestimates_a_known_circle():
    # un círculo de radio 50 mm: la cadena de píxeles 8-conexa sobrestima el perímetro
    raw = cv2.arcLength(circle_contour, True) * 0.2
    assert raw > 2 * np.pi * 50.0 * 1.01   # el defecto existe

def test_smoothing_brings_the_circle_within_1_percent():
    got = smoothed_arc_length_mm(circle_contour, 0.2)
    assert got == pytest.approx(2 * np.pi * 50.0, rel=0.01)

def test_outer_curve_is_the_longer_of_the_two_chains():
    mask, gt = make_banana(220.0, 0.85, 36.0, 0.2)
    axis = medial_axis_path(mask)
    a, b = split_contour_at_tips(contour_of(mask), axis.tips_px)
    outer = outer_curve(contour_of(mask), axis.tips_px)
    assert len(outer) == max(len(a), len(b))

def test_codex_length_matches_outer_curve_ground_truth_within_2_percent():
    mask, gt = make_banana(220.0, 0.85, 36.0, 0.2)
    got = smoothed_arc_length_mm(outer_curve(contour_of(mask), tips), 0.2)
    assert got == pytest.approx(gt.outer_length_mm, rel=0.02)

def test_outer_curve_is_longer_than_the_medial_axis():
    # el motivo de todo este módulo: en un objeto curvo el borde convexo
    # es más largo que la línea media
    assert codex_length > centerline_length
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_outline.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `codex_caliper/geometry/outline.py`**

`split_contour_at_tips` localiza los índices del contorno más cercanos a cada punta y parte la secuencia cerrada en dos cadenas. `smoothed_arc_length_mm` ajusta un spline suavizante con `scipy.interpolate.splprep` sobre la cadena y lo integra; sin este paso la longitud de una cadena 8-conexa sobrestima la de una curva suave por el efecto escalera.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_outline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/geometry/outline.py tests/test_outline.py
git commit -m "feat: Codex outer-curve length with staircase correction"
```

---

### Task 8: Recorte de pedicelo, calibre, área y volumen

**Files:**
- Create: `codex_caliper/geometry/measures.py`
- Test: `tests/test_measures.py`

**Interfaces:**
- Consumes: `MedialAxis` (Task 6), `outer_curve`/`smoothed_arc_length_mm` (Task 7)
- Produces: `trim_pedicel(s_mm, w_mm, drop_ratio: float = 0.45) -> tuple[int, int]` — índices de inicio y fin del fruto
- Produces: `BananaMeasures` dataclass con `codex_length_mm`, `caliper_mm`, `area_mm2`, `volume_mm3`, `width_profile` (tupla `s`, `w` en mm — nombrada así para no chocar con la función `width_profile_mm` de Task 6)
- Produces: `measure_mask(mask, mm_per_px, cfg) -> BananaMeasures | None`
- Produces: `touches_border(mask, margin_px: int = 2) -> bool`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_caliper_matches_ground_truth_within_half_a_millimetre():
    mask, gt = make_banana(220.0, 0.85, 36.0, 0.2)
    assert measure_mask(mask, 0.2, cfg).caliper_mm == pytest.approx(gt.caliber_mm, abs=0.5)

def test_area_and_volume_match_ground_truth_within_3_percent():
    mask, gt = make_banana(220.0, 0.85, 36.0, 0.2)
    m = measure_mask(mask, 0.2, cfg)
    assert m.area_mm2 == pytest.approx(gt.area_mm2, rel=0.03)
    assert m.volume_mm3 == pytest.approx(gt.volume_mm3, rel=0.03)

def test_pedicel_is_trimmed_so_length_matches_the_stemless_banana():
    plain, gt = make_banana(220.0, 0.85, 36.0, 0.2)
    stemmed, _ = make_banana(220.0, 0.85, 36.0, 0.2, pedicel_mm=25.0, pedicel_width_mm=9.0)
    assert measure_mask(stemmed, 0.2, cfg).codex_length_mm == pytest.approx(
        measure_mask(plain, 0.2, cfg).codex_length_mm, rel=0.03)

def test_gradual_blossom_taper_is_not_trimmed():
    # el extremo floral se adelgaza de forma gradual y debe conservarse;
    # solo la caída abrupta del pedicelo se corta
    assert tapered_length == pytest.approx(plain_length, rel=0.03)

# Review Focus 3
def test_mask_touching_the_canvas_border_is_rejected():
    assert touches_border(mask_cut_by_edge) is True
    assert measure_mask(mask_cut_by_edge, 0.2, cfg) is None

# Review Focus 4
def test_degenerate_mask_returns_none_without_raising():
    assert measure_mask(np.zeros((80, 80), np.uint8), 0.2, cfg) is None
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_measures.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `trim_pedicel`**

Recorrer `w(s)` desde cada extremo hacia el centro buscando el corte por la **derivada** de `w`, no por su valor absoluto: el extremo floral también se adelgaza, pero de forma gradual, mientras que la transición al pedicelo es abrupta. Cortar donde la derivada normalizada supera el umbral y el ancho ha caído por debajo de `drop_ratio · w_max`.

- [ ] **Step 4: Implementar `measure_mask` y `touches_border`**

`measure_mask` encadena: `touches_border` → `medial_axis_path` → `width_profile_mm` → `trim_pedicel` → `outer_curve` recortada a los índices del fruto → `smoothed_arc_length_mm`. Calibre = `2 · radio` en el punto medio del eje **ya recortado**. Área = píxeles × `mm_per_px²`. Volumen = `Σ π(w/2)² Δs` sobre el tramo del fruto. Devuelve `None` en cuanto cualquier paso falle.

- [ ] **Step 5: Correr los tests**

Run: `pytest tests/test_measures.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add codex_caliper/geometry/measures.py tests/test_measures.py
git commit -m "feat: pedicel trimming, caliper, area and volume"
```

---

### Task 9: Corrección de paralaje

**Files:**
- Create: `codex_caliper/parallax.py`
- Test: `tests/test_parallax.py`

**Interfaces:**
- Consumes: `BananaMeasures` (Task 8), `AppConfig` (Task 1)
- Produces: `scale_factor(effective_height_mm: float, object_height_mm: float) -> float`
- Produces: `correct_measures(measures, cfg, iterations: int = 2) -> BananaMeasures`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_scale_factor_uses_half_the_object_height():
    # la silueta vista desde arriba corresponde a la sección más ancha,
    # que está a media altura del objeto
    assert scale_factor(267.0, 35.0) == pytest.approx((267.0 - 17.5) / 267.0)

def test_flat_object_is_left_untouched():
    assert scale_factor(267.0, 0.0) == 1.0

def test_uncorrected_banana_is_overestimated_by_about_six_percent():
    corrected = correct_measures(raw, cfg)
    assert raw.codex_length_mm / corrected.codex_length_mm == pytest.approx(1.065, rel=0.02)

def test_two_iterations_converge():
    a = correct_measures(raw, cfg, iterations=2)
    b = correct_measures(raw, cfg, iterations=6)
    assert a.codex_length_mm == pytest.approx(b.codex_length_mm, abs=0.05)

def test_volume_scales_with_the_cube_of_the_factor():
    # longitud y ancho escalan con el factor; el volumen va al cubo
    assert corrected.volume_mm3 == pytest.approx(raw.volume_mm3 * f ** 3, rel=1e-6)
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_parallax.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `codex_caliper/parallax.py`**

`scale_factor = (H_ef - h/2) / H_ef`. `correct_measures` itera: tomar el calibre actual como `h`, calcular el factor, reescalar longitud y calibre por `f`, área por `f²` y volumen por `f³`, y repetir.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_parallax.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/parallax.py tests/test_parallax.py
git commit -m "feat: iterative parallax correction from measured thickness"
```

---

### Task 10: Veredicto Codex y grado comercial

**Files:**
- Create: `codex_caliper/codex.py`
- Test: `tests/test_codex.py`

**Interfaces:**
- Produces: `CodexVerdict` dataclass con `passes: bool`, `length_ok: bool`, `caliper_ok: bool`, `commercial_grade: int | None`
- Produces: `evaluate(length_mm: float, caliper_mm: float) -> CodexVerdict`

- [ ] **Step 1: Escribir los tests que fallan**

```python
@pytest.mark.parametrize("length,caliper,expected", [
    (140.0, 27.0, True),    # exactamente en los mínimos: cumple
    (139.9, 27.0, False),
    (140.0, 26.9, False),
    (190.0, 38.0, True),
])
def test_codex_thresholds(length, caliper, expected):
    assert evaluate(length, caliper).passes is expected

def test_commercial_grade_is_the_truncated_caliper_in_mm():
    assert evaluate(190.0, 38.7).commercial_grade == 38

def test_no_commercial_grade_below_the_minimum():
    assert evaluate(190.0, 33.2).commercial_grade is None
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_codex.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `codex_caliper/codex.py`**

Umbrales de las Global Constraints. `commercial_grade` es el calibre truncado a entero cuando alcanza los 34 mm del mínimo comercial, `None` por debajo.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_codex.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/codex.py tests/test_codex.py
git commit -m "feat: CODEX STAN 205-1997 verdict and commercial grade"
```

---

### Task 11: Agregación temporal e incertidumbre

**Files:**
- Create: `codex_caliper/aggregation.py`
- Test: `tests/test_aggregation.py`

**Interfaces:**
- Consumes: `BananaMeasures` (Task 8)
- Produces: `TrackAggregate` dataclass con `track_id: int`, `samples: int`, `locked: bool`, `length_mm: float`, `length_std_mm: float`, `caliper_mm: float`, `caliper_std_mm: float`, `volume_mm3: float`
- Produces: `MeasurementAggregator(window: int = 30, lock_std_mm: float = 1.0)` con `push(track_id, measures) -> TrackAggregate`, `get(track_id) -> TrackAggregate | None`, `forget(track_id)`, `active_ids() -> list[int]`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_reports_the_median_not_the_mean():
    # un frame atípico no debe arrastrar el resultado
    agg = MeasurementAggregator(window=30)
    for _ in range(29): agg.push(1, measures(length=190.0))
    out = agg.push(1, measures(length=900.0))   # frame malo
    assert out.length_mm == pytest.approx(190.0, abs=0.1)

def test_std_is_reported_as_the_uncertainty():
    for v in jittered_lengths: agg.push(1, measures(length=v))
    assert agg.get(1).length_std_mm == pytest.approx(np.std(jittered_lengths), rel=0.05)

def test_does_not_lock_until_the_window_is_full():
    for _ in range(29): out = agg.push(1, measures(length=190.0))
    assert out.locked is False
    assert agg.push(1, measures(length=190.0)).locked is True

def test_does_not_lock_while_the_measurement_is_still_noisy():
    for v in very_noisy_lengths[:40]: out = agg.push(1, measures(length=v))
    assert out.locked is False

def test_window_slides_and_drops_the_oldest_sample():
    assert agg.get(1).samples == 30
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_aggregation.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `codex_caliper/aggregation.py`**

Un `collections.deque(maxlen=window)` por `track_id`. Mediana por su robustez frente a frames malos — con media, un solo frame con la máscara rota arrastra el resultado. `locked` cuando el buffer está lleno **y** la desviación está por debajo de `lock_std_mm`.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_aggregation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/aggregation.py tests/test_aggregation.py
git commit -m "feat: temporal median aggregation with reported uncertainty"
```

---

### Task 12: Detección YOLO con tracking y reasociación

**Files:**
- Create: `codex_caliper/detection.py`
- Test: `tests/test_detection.py`

**Interfaces:**
- Produces: `Instance` dataclass con `track_id: int`, `class_id: int`, `mask: np.ndarray`, `bbox: tuple[int, int, int, int]`, `confidence: float`
- Produces: `BananaDetector(model_path="yolov8n-seg.pt", infer_size=640, every_n_frames=1)` con `detect(frame) -> list[Instance]`
- Produces: `CentroidReassigner(max_distance_px: float)` con `assign(instances, previous) -> list[Instance]`
- Produces: `masks_are_adjacent(a, b, dilate_px: int = 3) -> bool`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_centroid_reassigner_recovers_a_dropped_track_id():
    # el objeto está quieto; el tracker le asigna un id nuevo tras perderlo
    out = CentroidReassigner(30.0).assign([Instance(track_id=7, ...)], previous=[Instance(track_id=2, ...)])
    assert out[0].track_id == 2

def test_centroid_reassigner_keeps_distinct_ids_for_distant_objects():
    assert {i.track_id for i in out} == {1, 2}

def test_skipped_frames_reuse_the_previous_instances():
    det = BananaDetector(every_n_frames=3, model=fake_model)
    first = det.detect(frame); det.detect(frame); det.detect(frame)
    assert fake_model.call_count == 1

def test_only_the_banana_class_survives():
    # el modelo falso devuelve clases 46 y 47
    assert all(i.class_id == 46 for i in det.detect(frame))

# Review Focus 2
def test_adjacent_masks_are_detected():
    assert masks_are_adjacent(mask_a, mask_touching_a) is True
    assert masks_are_adjacent(mask_a, mask_far_away) is False
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_detection.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `codex_caliper/detection.py`**

`BananaDetector` acepta el modelo por inyección para que los tests usen un doble y no descarguen pesos. Internamente llama a `model.track(frame_resized, persist=True, classes=[46])` y reescala las máscaras al tamaño del frame original. `every_n_frames` reutiliza las instancias previas entre inferencias — los objetos están quietos, así que no se pierde nada y es el respaldo para equipos sin GPU. `masks_are_adjacent` dilata y comprueba intersección.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_detection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/detection.py tests/test_detection.py
git commit -m "feat: YOLO segmentation with tracking, frame skipping and centroid fallback"
```

---

### Task 13: Refinamiento de contorno por saturación

**Files:**
- Create: `codex_caliper/segmentation.py`
- Test: `tests/test_segmentation.py`

**Interfaces:**
- Consumes: `Instance` (Task 12), `make_banana_scene` (Task 2)
- Produces: `refine_mask(rectified_bgr, coarse_mask, saturation_threshold: int | None = None) -> np.ndarray | None`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_shadow_on_white_background_is_excluded():
    # el motivo de segmentar por saturación y no por luminancia:
    # la sombra es oscura pero tan desaturada como el fondo blanco
    scene = make_banana_scene([...], 0.2)
    refined = refine_mask(scene, coarse_mask_including_shadow)
    assert overlap(refined, shadow_region) < 0.05

def test_refined_area_is_closer_to_truth_than_the_coarse_yolo_mask():
    assert abs(area(refined) - gt.area_mm2) < abs(area(coarse) - gt.area_mm2)

def test_keeps_only_the_largest_connected_component():
    assert number_of_components(refine_mask(scene, mask_with_speckles)) == 1

def test_returns_none_when_nothing_saturated_is_found():
    assert refine_mask(all_white_scene, coarse_mask) is None
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_segmentation.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `refine_mask` en `codex_caliper/segmentation.py`**

Recortar al bbox de `coarse_mask`, pasar a HSV y umbralar el **canal de saturación** con Otsu cuando `saturation_threshold is None`. Limpiar con apertura y cierre morfológicos y quedarse con el componente conexo mayor. No usar luminancia: sobre cartulina blanca la sombra es oscura pero desaturada, así que por saturación se queda con el fondo y por luminancia entraría en la máscara.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_segmentation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/segmentation.py tests/test_segmentation.py
git commit -m "feat: saturation-channel contour refinement"
```

---

### Task 14: Orquestación del pipeline

**Files:**
- Create: `codex_caliper/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: todo lo anterior
- Produces: `FrameResult` dataclass con `rectified: np.ndarray`, `raw_undistorted: np.ndarray`, `plane: PlaneCalibration | None`, `results: list[BananaResult]`, `warnings: list[str]`
- Produces: `BananaResult` dataclass con `track_id`, `aggregate: TrackAggregate`, `verdict: CodexVerdict`, `estimated_weight_g: float | None`, `mask: np.ndarray`
- Produces: `Pipeline(cfg, intrinsics, detector, density_g_per_cm3: float | None = None)` con `process(frame) -> FrameResult`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_end_to_end_on_a_synthetic_scene_recovers_ground_truth():
    # escena sintética con 4 ArUcos y un banano de dimensiones conocidas
    r = Pipeline(cfg, intr, fake_detector).process(scene)
    assert r.results[0].aggregate.length_mm == pytest.approx(gt.outer_length_mm, rel=0.03)

def test_no_results_and_a_warning_when_calibration_is_blocked():
    r = pipeline.process(scene_with_two_markers)
    assert r.results == []
    assert any("calibration" in w.lower() for w in r.warnings)

# Review Focus 2
def test_adjacent_bananas_produce_a_warning_and_no_measurements():
    r = pipeline.process(scene_with_two_touching_bananas)
    assert r.results == []
    assert any("separate" in w.lower() for w in r.warnings)

def test_zero_detections_returns_an_empty_result_without_raising():
    assert pipeline.process(empty_scene).results == []

def test_weight_is_none_until_a_density_is_supplied():
    assert Pipeline(cfg, intr, det, density_g_per_cm3=None).process(scene).results[0].estimated_weight_g is None
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `Pipeline.process` en `codex_caliper/pipeline.py`**

Orden por frame: `undistort` → `detect_markers` → `compute_plane_calibration` (abortar con warning si es `None`) → `rectify` → `detector.detect` sobre el rectificado → `CentroidReassigner` → comprobar `masks_are_adjacent` entre pares (abortar con warning si hay alguno) → por instancia `refine_mask` → `measure_mask` → `correct_measures` → `aggregator.push` → `codex.evaluate` → peso desde el volumen y la densidad si la hay. Acumular avisos en `warnings` en lugar de lanzar excepciones.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/pipeline.py tests/test_pipeline.py
git commit -m "feat: per-frame pipeline orchestration"
```

---

### Task 15: Validación, regresión de densidad y métricas de error

**Files:**
- Create: `codex_caliper/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Produces: `ValidationLog(path)` con `record(sample_id: str, measures: BananaMeasures, actual_weight_g: float)`, `to_frame() -> pd.DataFrame`
  - Recibe `BananaMeasures`, no `TrackAggregate`: este módulo es independiente y comprobable solo. Es la `ValidationView` de Task 19 la que construye un `BananaMeasures` a partir del `TrackAggregate` fijado.
- Produces: `fit_density(frame) -> DensityFit` con `g_per_cm3: float`, `r_squared: float`, `n: int`
- Produces: `error_metrics(frame, density: DensityFit) -> ErrorMetrics` con `mae_g`, `mape_pct`, `r_squared`
- Produces: `dimensional_check(measured_mm, reference_mm) -> float` — para los patrones planos (tarjeta ISO 85.60 × 53.98 mm, ArUco de 24.0 mm)

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_density_is_fitted_not_assumed():
    # volúmenes y pesos generados con una densidad conocida
    fit = fit_density(frame_with_known_density_0_94)
    assert fit.g_per_cm3 == pytest.approx(0.94, rel=0.02)
    assert fit.r_squared > 0.99

def test_mae_is_zero_for_a_perfect_fit():
    assert error_metrics(perfect_frame, fit).mae_g == pytest.approx(0.0, abs=1e-6)

def test_mape_is_reported_in_percent():
    assert error_metrics(frame_off_by_ten_percent, fit).mape_pct == pytest.approx(10.0, rel=0.01)

def test_log_roundtrips_through_csv(tmp_path):
    assert len(ValidationLog(tmp_path / "v.csv").to_frame()) == 3

def test_dimensional_check_returns_signed_error_in_mm():
    assert dimensional_check(24.4, 24.0) == pytest.approx(0.4)

def test_fit_density_requires_a_minimum_sample_size():
    with pytest.raises(ValueError):
        fit_density(frame_with_two_rows)
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_validation.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `codex_caliper/validation.py`**

`fit_density` hace regresión lineal por el origen de peso contra volumen: la pendiente es la densidad efectiva del sistema, que es más honesto y más preciso que tomar una densidad de tabla. Exigir al menos 10 muestras. `error_metrics` calcula MAE, MAPE y R² sobre el peso predicho.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_validation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/validation.py tests/test_validation.py
git commit -m "feat: validation log, fitted density and error metrics"
```

---

### Task 16: Captura de cámara en hilo propio

**Files:**
- Create: `codex_caliper/camera.py`
- Test: `tests/test_camera.py`

**Interfaces:**
- Produces: `CameraThread(index: int, width=1920, height=1080, fourcc="MJPG", backend=cv2.CAP_DSHOW)` — `QThread` con señal `frame_ready(np.ndarray)`, métodos `start()`, `stop()`, propiedad `actual_size -> tuple[int, int]`
- Produces: `open_capture(index, ...) -> cv2.VideoCapture` — aislado para poder sustituirlo en tests

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_emits_frames_from_a_fake_capture(qtbot):
    with qtbot.waitSignal(thread.frame_ready, timeout=2000): thread.start()

def test_stop_terminates_the_loop_and_releases_the_capture():
    assert fake_capture.released is True

def test_requests_mjpeg_at_1080p():
    # a 1080p sobre USB 2.0 MJPEG es la única opción a 30 fps
    assert fake_capture.properties[cv2.CAP_PROP_FOURCC] == mjpg_code

def test_actual_size_reports_what_the_device_granted_not_what_was_asked():
    assert thread.actual_size == (1280, 720)
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_camera.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `codex_caliper/camera.py`**

`QThread` con bucle de lectura y señal Qt. `actual_size` lee las propiedades tras configurarlas, porque el driver puede no conceder lo pedido — y los intrínsecos dependen de la resolución real (Task 3, `require_compatible`).

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_camera.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/camera.py tests/test_camera.py
git commit -m "feat: threaded camera capture"
```

---

### Task 17: Overlay de cotas sobre el frame

**Files:**
- Create: `codex_caliper/ui/overlay.py`
- Test: `tests/test_overlay.py`

**Interfaces:**
- Consumes: `BananaResult` (Task 14)
- Produces: `draw_results(canvas: np.ndarray, results: list[BananaResult], mm_per_px: float) -> np.ndarray`
- Produces: `draw_marker_overlay(frame, detections) -> np.ndarray`
- Produces: `format_measurement(value_mm: float, std_mm: float) -> str`

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_measurement_text_is_english_and_carries_the_uncertainty():
    assert format_measurement(187.34, 0.82) == "187.3 ± 0.8 mm"

def test_decimals_are_capped_by_the_uncertainty():
    # anti-objetivo del spec: nada de falsa precisión
    assert format_measurement(187.34, 5.0) == "187 ± 5 mm"

def test_draw_results_does_not_mutate_the_input_canvas():
    before = canvas.copy()
    draw_results(canvas, results, 0.2)
    assert np.array_equal(canvas, before)

def test_labels_follow_the_banana_position():
    # las cotas se dibujan sobre el video siguiendo a cada objeto
    assert text_anchor_for(result_at_left) != text_anchor_for(result_at_right)

def test_unlocked_results_are_drawn_with_a_progress_indicator():
    assert "19/30" in rendered_text_of(draw_results(canvas, [unlocked_result], 0.2))
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_overlay.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `codex_caliper/ui/overlay.py`**

`format_measurement` deriva el número de decimales de la incertidumbre — es la defensa contra la falsa precisión del spec. `draw_results` trabaja sobre una copia y ancla cada etiqueta al bbox de su banano, de modo que los números sigan al objeto cuando se mueve.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_overlay.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/ui/overlay.py tests/test_overlay.py
git commit -m "feat: on-frame measurement overlay with precision guard"
```

---

### Task 18: Ventana principal, vista de video y panel lateral

**Files:**
- Create: `codex_caliper/ui/main_window.py`, `codex_caliper/ui/video_view.py`, `codex_caliper/ui/side_panel.py`
- Test: `tests/test_ui_layout.py`

**Interfaces:**
- Consumes: `FrameResult` (Task 14), `draw_results` (Task 17), `CameraThread` (Task 16)
- Produces: `VideoView` — `QWidget` con `set_frame(rectified, pip_frame)`, `toggle_pip()`, propiedad `pip_visible: bool`
- Produces: `SidePanel` — ancho fijo 240 px, `update_results(results)`
- Produces: `MainWindow` — `QMainWindow` con barra de estado superior, `VideoView` central, `SidePanel` a la derecha

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_side_panel_width_is_fixed_at_240_px(qtbot):
    # al maximizar debe crecer el video, no el panel
    window.resize(2400, 1400)
    assert window.side_panel.width() == 240

def test_video_view_takes_the_remaining_width(qtbot):
    assert window.video_view.width() == window.width() - 240

def test_pip_toggles_with_the_keyboard(qtbot):
    qtbot.keyClick(window, Qt.Key_P)
    assert window.video_view.pip_visible is False

def test_pip_is_rendered_at_reduced_resolution():
    # ya se paga un warpPerspective por frame; no duplicar el coste de render
    assert view.pip_pixmap.width() < view.main_pixmap.width() // 3

def test_status_bar_shows_calibration_state_in_english(qtbot):
    assert "4/4 markers" in window.status_text()

def test_blocked_calibration_is_shown_as_a_warning(qtbot):
    assert window.status_is_warning() is True
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_ui_layout.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar las tres vistas**

Layout variante 1 del spec §8: barra fina de estado arriba, `VideoView` central con el lienzo rectificado a pantalla casi completa y el crudo como PiP en la esquina, `SidePanel` de 240 px fijos a la derecha con una tarjeta por `track_id`. El panel usa `setFixedWidth(240)`, no un stretch factor: al maximizar la ventana debe crecer el video. Todos los textos en inglés.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_ui_layout.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/ui/ tests/test_ui_layout.py
git commit -m "feat: main window with video-dominant layout"
```

---

### Task 19: Vistas de calibración y validación

**Files:**
- Create: `codex_caliper/ui/calibration_view.py`, `codex_caliper/ui/validation_view.py`
- Modify: `codex_caliper/ui/main_window.py` — navegación entre vistas
- Test: `tests/test_calibration_view.py`, `tests/test_validation_view.py`

**Interfaces:**
- Consumes: `Intrinsics` (Task 3), `PlaneCalibration` (Task 5), `ValidationLog`/`fit_density`/`error_metrics` (Task 15)
- Produces: `IntrinsicCalibrationView` con `capture_pose()`, `run_calibration()`, propiedad `undistort_preview: bool`
- Produces: `PlaneCalibrationView` — muestra esquinas detectadas, IDs, residual en mm, mm/px
- Produces: `ValidationView` — tabla y métricas de error

- [ ] **Step 1: Escribir los tests que fallan**

```python
def test_undistortion_toggle_switches_the_preview(qtbot):
    # el recurso visual más eficaz del proyecto: líneas rectas enderezándose
    view.undistort_preview = True
    assert view.displayed_frame_is_undistorted() is True

def test_calibration_is_rejected_above_the_rms_threshold(qtbot):
    assert view.result_accepted is False
    assert "0.5" in view.rejection_message()

def test_capture_counter_increases_with_each_accepted_pose(qtbot):
    assert view.pose_count == 3

def test_plane_view_reports_residual_in_millimetres(qtbot):
    assert "mm" in view.residual_text()

# Review Focus 5
def test_measurement_mode_is_blocked_without_valid_intrinsics(qtbot):
    window.load_intrinsics(None)
    assert window.measurement_enabled is False
    assert "calibration" in window.blocked_reason().lower()

def test_validation_view_shows_mae_and_r_squared(qtbot):
    assert "MAE" in view.summary_text() and "R²" in view.summary_text()
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_calibration_view.py tests/test_validation_view.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar las vistas y la navegación**

`IntrinsicCalibrationView` acumula poses con `find_chessboard`, llama a `calibrate_from_corners` y rechaza por encima de 0.5 px de RMS. El toggle antes/después de la corrección de distorsión es un requisito explícito del spec, no un extra. `MainWindow` deshabilita el modo medición mientras no haya intrínsecos compatibles con la resolución real de la cámara.

- [ ] **Step 4: Correr los tests**

Run: `pytest tests/test_calibration_view.py tests/test_validation_view.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add codex_caliper/ui/ tests/
git commit -m "feat: calibration and validation views"
```

---

### Task 20: Punto de entrada y prueba en hardware real

**Files:**
- Create: `codex_caliper/__main__.py`, `README.md`
- Test: `tests/test_entrypoint.py`

**Interfaces:**
- Produces: `main(argv) -> int` — carga config e intrínsecos, abre la cámara, lanza la ventana

- [ ] **Step 1: Escribir el test que falla**

```python
def test_main_reports_a_clear_error_when_config_is_missing(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.json")]) == 1
    assert "config" in capsys.readouterr().err.lower()

def test_main_reports_a_clear_error_when_the_camera_cannot_be_opened():
    assert main(["--camera", "99"]) == 1
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/test_entrypoint.py -v`
Expected: FAIL

- [ ] **Step 3: Implementar `codex_caliper/__main__.py`**

Argumentos: `--config`, `--camera`, `--intrinsics`. Fallar con mensaje claro y código 1, nunca con un stack trace.

- [ ] **Step 4: Escribir el `README.md` con el procedimiento de puesta en marcha**

Debe recoger, del spec: identificar el diccionario con `tools/find_aruco_dictionary.py`; ajustar el zoom a ~4 mm; **apretar los prisioneros y marcar los anillos**; medir la distancia lente→cartulina; medir el espesor apilando los cuatro marcadores; colocar la cartulina mate sobre base rígida; calibrar intrínsecos; fijar el `marker_layout`.

- [ ] **Step 5: Correr toda la batería de tests**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 6: Verificación en hardware real**

Sin tests automáticos — comprobar a mano y anotar los resultados:
- Los 4 marcadores se detectan de forma estable, sin parpadeo
- Residual de la homografía por debajo de 1 mm
- Una tarjeta ISO 7810 mide 85.6 × 54.0 mm con error menor a 1 mm
- Un ArUco dejado fuera de la calibración mide 24.0 ± 0.5 mm
- Un banano se mide de forma estable y se fija la medida en menos de 2 segundos

- [ ] **Step 7: Commit**

```bash
git add codex_caliper/__main__.py README.md tests/test_entrypoint.py
git commit -m "feat: entrypoint and setup documentation"
```

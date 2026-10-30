# codex-caliper

Live dimensional measurement of bananas against **CODEX STAN 205-1997**, from a single
USB camera, with four 3D-printed ArUco markers for metric calibration.

The standard defines two parameters for export-grade bananas:

| Parameter | Minimum | How the standard defines it |
|---|---|---|
| Length | ≥ 140 mm | measured **along the outer curve**, from the floral end to the base of the pedicel where the edible pulp ends |
| Grade (caliper) | ≥ 27 mm | diameter of the cross-section, at the reference finger |

**That first definition is the whole project.** A bounding box cannot measure a curve.
Measured on one real banana in this rig (details in *Accuracy*, below):

| Measurement | Value |
|---|---|
| Tip-to-tip chord — what a bounding box gives you | 182.3 mm |
| Inner (concave) curve | 217.0 mm |
| **Outer curve — the Codex length** | **233.5 mm** |

The standard's definition reads **51 mm longer than the chord — 28 % more**. A detector
that reports a box is not slightly imprecise here; it is wrong by nearly a third.

## Architecture: neural detection, classical measurement

The two responsibilities are split and never mixed:

| Stage | Tool | Answers |
|---|---|---|
| Detect and identify | YOLOv8n-seg (COCO class `banana`) + ByteTrack | *what* it is and *where* |
| Measure | classical CV on a rectified, metric canvas | the exact outline and its dimensions |

COCO segmentation masks are polygons annotated by hand for speed, not precision contours.
Measuring directly on a YOLO mask would mean the accuracy of the system is set by a human
annotator working fast. YOLO contributes what classical CV cannot have — knowing that *this*
is a banana, and keeping its identity across frames — and classical CV contributes what the
network does not give: a precise, repeatable edge.

Per-frame chain:

```
undistort → detect 4 ArUcos → fit homography → rectify to a 0.2 mm/px canvas
          → YOLO detect/track (on the canvas) → centroid re-association
          → adjacency guard → saturation-based mask refinement
          → medial axis → outer-curve arc length + caliper + area + volume
          → parallax correction → temporal median over 30 frames → Codex verdict → weight
```

The homography is refitted **every frame**, not once per session. The four markers are
permanently in view, so it costs nothing, and it turns the rig's most obvious weakness into
a strength: knock the stand, let the table vibrate, and the system re-calibrates on the next
frame. The one thing it cannot self-correct is a moved zoom or focus ring — see *Known
limits*.

With 3 of 4 markers visible the fit still has 12 correspondences for 8 unknowns, so the
system keeps measuring and says so (`⚠ 3/4 markers`). Below 3, measurement is **blocked**:
no measurement is ever reported without a valid calibration.

## Install

```bash
py -3.11 -m venv .venv
```

What to install next depends on what you want to do:

| You want to | Install |
|---|---|
| Read the code | `.venv/Scripts/python.exe -m pip install -e .` |
| Run the tests (262 of them) | `.venv/Scripts/python.exe -m pip install -e ".[dev]"` |
| Run it live against the rig | `.venv/Scripts/python.exe -m pip install -e ".[dev,live]"` |

The base install (`opencv-python`, `numpy`, `scipy`, `scikit-image`, `PySide6`, `pandas`)
is enough to read the code with every import resolving. `dev` adds `pytest` and
`pytest-qt` — enough to run the full suite. `live` adds `ultralytics`, the only thing that
pulls in `torch` (about 2.5 GB): it is not a base dependency because its import is lazy
(see `detection.py`), and the full test suite runs without it by injecting a fake model.
Reading the code or running the tests never pays that cost; pointing the app at a real
camera does. If the `live` extra is missing, the entry point says so plainly rather than
failing with an import traceback.

Developed and tested on Python 3.11, Windows 11. `pyproject.toml` declares
`requires-python = ">=3.10"` as the real floor; 3.10 and 3.12+ are untested, not known to
be broken.

Run it:

```bash
.venv/Scripts/python.exe -m codex_caliper
.venv/Scripts/python.exe -m codex_caliper --config config/app.json --intrinsics config/intrinsics.json --camera 1
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `config/app.json` | rig configuration (physical measurements of *your* setup) |
| `--intrinsics` | `config/intrinsics.json` | intrinsic calibration, valid only for the zoom/focus it was taken at |
| `--camera` | `0` | camera device index |

Neither file is created for you. `marker_thickness_mm` and `marker_layout` have no
meaningful defaults — they are measurements of a physical rig, and a config generated with
invented numbers would measure wrong *silently*. Missing or unreadable config, a
`marker_layout` that does not cover the configured marker ids, missing intrinsics, a camera
that will not open, or a missing YOLO backend each produce one line on stderr and exit
code 1.

## Setting up the rig

The order matters. Steps 4 and 5 are irreversible in the sense that redoing them forces you
to redo step 9.

**1 — Build a plane.** All the metrology rests on the work area being a *plane*. White card
on carpet, a mat, or anything soft is not a plane: it sags under the banana and ripples.
Put the card on MDF, acrylic, glass, or just move the whole rig onto a table. The camera
stand must rest on the same rigid surface.

**2 — Matte white card, diffuse light.** Matte is a requirement, not a preference: gloss
produces specular highlights that segmentation mistakes for the object. Light the scene
diffusely or from the side — **never a point source directly overhead**: black PLA is
shiny, and specular reflections destroy marker corners and make detection flicker. A
mains-powered LED lamp will put visible horizontal bands in the video (rolling shutter plus
PWM dimming); that is cosmetic, not metrological — see *Accuracy*.

**3 — Identify the ArUco dictionary and the ids.** Photograph the four markers on the card
and run:

```bash
.venv/Scripts/python.exe tools/find_aruco_dictionary.py capturas/markers.png
```

It tries every predefined OpenCV dictionary and prints which ones decoded, with the ids.
The predefined dictionaries are nested — the first 50 codes of `DICT_5X5_100` *are*
`DICT_5X5_50` — so a low-id marker decodes identically in 50/100/250/1000 and the tool
breaks that tie toward the smallest dictionary (greater Hamming distance, fewer false
positives). On this rig the answer is `DICT_6X6_50`, ids `[1, 2, 3, 4]`. Write both into
the config.

**4 — Set the zoom to ≈ 4 mm and focus on the work plane.** The lens is a manual 2.8–12 mm
varifocal. The focal length is a trade: at 2.8 mm the field is 565 mm wide but barrel
distortion is severe and resolution is wasted; at 6 mm the work area no longer fits. At
≈ 4 mm and 270 mm of working distance the field is ≈ 395 × 221 mm and the scale ≈ 0.206
mm/px, which is where this project's 0.2 mm/px canvas comes from.

Check the result live:

```bash
.venv/Scripts/python.exe tools/aruco_probe.py --camera 0
```

The probe reports the dictionary, marker count and per-id detection stability, the marker
side in pixels, the scale, and the field of view. Targets: all 4 markers at ≥ 98 %
stability, marker side ≥ 40 px (this rig measures 119–123 px — 3× the margin needed for
subpixel corner refinement), field of view in the 300–460 mm band. Press `c` to open the
DirectShow property panel if you need *Powerline Frequency / anti-flicker*; `s` saves a
snapshot; `q` quits.

**5 — Lock the varifocal. This is the single most dangerous step in the whole procedure.**
The intrinsic parameters are valid **only** for the exact zoom and focus position they were
calibrated at. If either ring moves — including by accident, while handling the rig — the
camera matrix and distortion coefficients stop corresponding to reality and **every
measurement is corrupted silently**. So:

1. Tighten the set screws on **both** rings.
2. Mark the ring positions with a stroke of paint or tape, so a glance tells you whether
   they moved.
3. Record what you did in the `zoom_note` field of the intrinsics file.
4. Check the marks before every session.

The software can only verify one symptom of this — that the resolution the camera delivers
now matches the resolution the intrinsics were taken at (`Intrinsics.require_compatible`,
which blocks measurement when it fails). It cannot see a moved ring. The paint can.

**6 — Measure the lens-to-card distance** with a tape and write it into
`working_distance_mm`. On this rig the 30 cm stand plus the hanging camera mount gives
≈ 270 mm; do not take that number on faith, measure yours.

**7 — Measure the marker thickness.** The markers are 3D-printed, so the printed face sits
2–3 mm *above* the plane of the table, not in it — at 270 mm of working distance that is a
systematic scale error of roughly 1 % on every measurement, and the code corrects it by
subtracting the thickness from the working distance (`effective_height_mm`).

No calipers needed: **stack all four markers and measure the stack with the tape, then
divide by 4.** A tape with 1 mm resolution gives ≈ 0.25 mm of precision that way, which is
enough. (Alternatively, read the part height off the STL or the slicer.) Write it into
`marker_thickness_mm`.

**8 — Place the markers and measure the layout.** Open them out to the widest rectangle the
stand legs allow — on this rig ≈ 233 mm horizontally and the full 213 mm vertically. This
matters: an earlier layout with the markers 184 × 122 mm apart **did not fit a banana**
(bananas are 180–230 mm), while the camera had 380 × 213 mm of field going spare. A long
banana can also lie **diagonally**; the homography and the medial axis are indifferent to
orientation.

Leave a clear quiet zone of white card around each black square — the detector needs it —
and do not push the markers against the edge of the card.

Then measure the **centre** of each marker, in millimetres, in a flat coordinate system of
your choosing (the first marker's centre at the origin is the easy choice), and write the
config:

```json
{
  "marker_length_mm": 24.0,
  "marker_thickness_mm": 2.4,
  "working_distance_mm": 271.0,
  "canvas_mm_per_px": 0.2,
  "temporal_window": 30,
  "pedicel_drop_ratio": 0.45,
  "border_margin_px": 2,
  "aruco_dictionary": "DICT_6X6_50",
  "marker_ids": [1, 2, 3, 4],
  "marker_layout": {
    "1": [0.0, 0.0],
    "2": [215.0, 0.0],
    "3": [215.0, 118.0],
    "4": [0.0, 118.0]
  }
}
```

`marker_length_mm` is the side of the **complete black square** (24.0 mm here), the same
quantity OpenCV calls `markerLength`. The reprojection residual the app shows is, among
other things, a direct measure of how well you measured this layout.

**9 — Calibrate the intrinsics.** Capture ≈ 20 photos of a chessboard in varied poses —
tilted, near, far, in all four corners of the frame; a set of near-identical frontal views
is ill-conditioned and will not converge. Save them somewhere, then:

```python
import glob

import cv2
import numpy as np

from codex_caliper.intrinsics import calibrate_from_corners, find_chessboard

PATTERN_SIZE = (9, 6)        # inner corners: (columns, rows)
SQUARE_MM = 25.0             # printed square side, measured with the tape
IMAGE_SIZE = (1920, 1080)    # the resolution the driver actually grants

grid = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), dtype=np.float32)
grid[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * SQUARE_MM

object_points, image_points = [], []
for path in sorted(glob.glob("capturas/board_*.png")):
    corners = find_chessboard(cv2.imread(path), PATTERN_SIZE)
    if corners is None:
        print(f"no board found in {path}")
        continue
    object_points.append(grid)
    image_points.append(corners)

intr = calibrate_from_corners(object_points, image_points, IMAGE_SIZE)
print(f"RMS = {intr.rms_px:.3f} px   acceptable = {intr.is_acceptable}")
intr.zoom_note = "zoom ~4 mm, set screws tight, rings marked with white paint"
intr.save("config/intrinsics.json")
```

**Acceptance criterion: RMS < 0.5 px.** Above that, capture again — do not proceed. This
step cannot be skipped: the wide-angle lens has real barrel distortion, and without
correcting it the homography tries to compensate a non-linear effect with a linear model.
The symptom is bewildering rather than obvious — *the same banana measures differently
depending on where you put it*, and the error cannot be attributed to anything in the
algorithm.

**10 — Run it.** `.venv/Scripts/python.exe -m codex_caliper`. The status bar shows the
intrinsics RMS, the marker count, and the live scale in mm/px. Measurement stays blocked
until all three are healthy.

## Accuracy

**What the system targets: ±1 mm on length, ±0.5 mm on caliper.** That is the optical
budget at this working point (0.2 mm/px canvas, 24 mm markers at ~120 px, subpixel corner
refinement), not a measured end-to-end result.

What *has* been measured on this rig:

| Quantity | Measured | Where it came from |
|---|---|---|
| Granted resolution | 1920 × 1080 | `tools/aruco_probe.py`, 3 captures |
| Marker side | 119.3–122.7 px | same (optical budget predicted ~116 px) |
| Scale | 0.196–0.199 mm/px | same — confirms the 0.2 mm/px canvas |
| Field of view | ≈ 392 × 221 mm | second capture round, with the matte card |
| Camera–plane parallelism | size ratio 0.974 (≈ 2° tilt) | `aruco_probe` |
| **Corner repeatability** | **σ 0.05–0.34 mm between frames** | `aruco_probe` — well inside the ±1 mm target |
| Marker decoding, 4/4 | in all 7 captures, card *and* stone | with `perspectiveRemovePixelPerCell = 8` |
| Shadow rejection | 57 of 913 372 dark-but-desaturated pixels entered the mask (0.006 %) | real capture with a banana on the matte card |
| Flicker banding | 1.2–1.8 saturation units, against a ~55-unit margin to the segmentation threshold; segmented area stable to **0.12 %** across 4 banded captures | real captures |
| Pixel-staircase overestimate | 253.2 mm raw chain vs 233.5 mm smoothed — **+8.4 %** | same banana as the table at the top |
| Parallax, uncorrected | **≈ +6.5 %** (≈ 12 mm on a 190 mm banana) | geometric model: a 35 mm-thick banana at 270 mm is 13 % of the working distance |

Two notes on how to read those numbers.

The **absolute** lengths at the top of this README (182.3 / 217.0 / 233.5 mm) were measured
before the full correction chain existed — no distortion correction, no homography, no
parallax correction, using the scale derived straight from the markers, which sit at radius
~598 px from the optical centre while the banana sits at 0–300 px. Different parts of the
lens. The **relative** comparisons (chord vs arc, raw vs smoothed) are solid, because both
terms were measured the same way on the same image; the absolute values are provisional.
That gap is itself the empirical argument for why step 9 is not optional.

**There is no single end-to-end accuracy figure here yet, on purpose.** The synthetic
end-to-end test measures +0.32 % on length, and that number is *not* the system's accuracy:
it is two biases of opposite sign cancelling for that one fixture's geometry — the pedicel
trim removes ~13 mm of the fixture's floral taper while the fixture's own outer-length bias
pushes the other way. The project's test suite pins both terms separately precisely so that
the cancellation cannot hide a regression in either. The honest figures are the ±1 mm /
±0.5 mm target above, with the pedicel trim's ±3 mm dominating on fruit with a stalk.

The real end-to-end number comes from the validation table, which requires the hardware run
below.

### Validation protocol

Two different questions, deliberately not mixed — mixing them makes it impossible to tell
where an error came from:

1. **Is the metrology right?** Validate against flat objects of known normalised dimension:
   an ISO/IEC 7810 ID-1 card (85.60 × 53.98 mm), an ArUco marker deliberately left out of
   the calibration (must read 24.0 mm), coins. Being flat, they isolate calibration error
   from parallax error.
2. **Is the banana segmentation right?** Validate against a kitchen scale.

**Density is not assumed, it is fitted.** Rather than taking a density from a table, the
validation view regresses real weight against computed volume through the origin; the slope
*is* the system's effective density, and the R² of that regression is a publishable result.
At least 10 samples are required before a fitted slope is reported — below that it is noise
with pretensions.

The validation log also carries a hand-measured length column, and it is not optional
decoration: the scale cannot detect a pedicel-trim error at all. Weighing a banana says
nothing about where its stalk was cut. Samples are appended to `validation_log.csv`, next
to your config file.

## Known limits

Said out loud, with numbers, because hiding them would cost more credibility than they do.

- **The pedicel trim lands within about ±3 mm of the real base of the stalk.** The cut is
  placed where the thickness profile drops below the body plateau, which is not exactly the
  geometric base. Measured across five synthetic geometries with a 25 mm pedicel, what the
  trim removes falls between 82 % and 105 % of that length. **On fruit with a stalk, this
  dominates the ±1 mm target for the rest of the chain.** It is not an implementation
  defect: it is the achievable precision of a derivative-based criterion, and a refinement
  that did not reopen instability on small calipers was looked for and not found. Closing
  it means changing the criterion (e.g. fitting two lines to the thickness profile and
  taking their intersection), not tuning the threshold. And the scale cannot catch it —
  which is why the validation log has a hand-measured length column.

- **The medial axis recedes from flat cut edges by roughly half the local width.** Measured
  3.6–3.7 mm on a 9 mm-wide cut and ≈ 17.6 mm on a 36 mm one. The detected tip therefore
  sits inside the body rather than at the knife cut. This is a property of the method, not a
  bug, and it is bounded in practice because the pedicel end is discarded anyway — but **the
  synthetic fixture has rounded caps and does not reproduce this case at all**, so it still
  needs validating against real knife-cut fruit.

- **A very ripe banana can lose area.** Segmentation runs on the HSV *saturation* channel
  (not luminance — a shadow is dark but just as desaturated as white card, so luminance
  would swallow the shadow into the mask). Brown patches on over-ripe fruit drop to
  saturation 45–61, below the segmentation threshold (Otsu, which lands near 83 on this rig). The morphological closing plus largest-
  connected-component selection recover them on a normal banana, but a banana with a lot of
  brown surface can still lose area. Needs validating across ripeness states.

- **No bunches, no touching fruit.** 1 to 3 separated bananas, lying down. If two masks come
  out adjacent or fused, the pipeline **refuses to measure** and says "Separate them on the tray."
  rather than reporting a plausible number for a chimera. The threshold is derived, not
  chosen: the mask-refinement closing bridges gaps up to 8 px (1.6 mm at canvas scale), so
  the adjacency guard sits at 10 px (2.0 mm) — strictly above it, or there would be a band
  of separations where two fruit silently fuse into one measurement.

- **The intrinsics are valid only for the zoom and focus they were calibrated at.** If
  either ring moves, every measurement is corrupted *silently* — there is no error, no
  warning, just wrong numbers. The only automatic check possible is that the resolution
  still matches. This is why step 5 says to tighten the set screws and mark the rings with
  paint, and why you should check the marks before each session.

- **The in-app "Calibrate camera" screen cannot originate a calibration.** It is wired to
  display one: the toggle shows the before/after of `undistort` against whatever
  intrinsics are already loaded, and `IntrinsicCalibrationView` itself has working
  `capture_pose`/`set_live_frame`/`run_calibration` methods with their own tests. What is
  missing is the wiring between it and the running app — `main_window.py` instantiates the
  view and connects its `calibration_accepted` signal to unlock measurement, but nothing
  ever calls `set_live_frame` or `capture_pose` on it, there is no button to capture a
  pose, and there is no method to save a result to disk. Open the screen and it says "No
  frame yet" forever; click "Run calibration" and it reports "Cannot calibrate yet",
  because no pose was ever captured for it to work with. The real path to an intrinsics
  file is step 9 above — the script against `codex_caliper.intrinsics` directly, which is
  complete and does the whole job.

- **Measurement does not run at 30 fps, and does not need to.** `medial_axis_path` alone
  costs ~11 ms per banana, which is the entire per-banana budget at 30 fps. What the viewer
  sees moving at 30 fps is the warp and the overlay; the measurement settles over a 30-frame
  temporal window regardless.

Also out of scope in this version, by choice: training a model (COCO-pretrained YOLO is used
as-is), ripeness classification, surface-defect detection, objects tall relative to the
270 mm working distance, multi-camera, and production deployment.

## Reported uncertainty

Segmentation trembles between frames: an edge moves a pixel or two and the number jumps.
With a persistent `track_id` per fruit, the system keeps a 30-sample circular buffer per
banana and reports the **median** (robust to bad frames, unlike the mean), with the
**standard deviation of the buffer as the reported measurement uncertainty**. A measurement
"locks" once the buffer is full and the deviation falls below threshold; until then the
panel shows progress (`19/30`).

No figure is ever shown without its ±, and no more decimals are printed than the calibration
supports.

## Hardware verification checklist — not yet executed

**Everything below is unfilled on purpose.** These checks require the physical rig: the
camera connected, the markers placed and their thickness measured, an ISO 7810 card, a spare
ArUco left out of the calibration, and `ultralytics` + `torch` installed. None of them has
been run, and no result has been estimated, inferred, or filled in from the synthetic tests.

Record what you actually observe, including the failures:

- [ ] All 4 markers detected steadily, without flicker — per-id stability from
      `tools/aruco_probe.py`: id 1 ____ %, id 2 ____ %, id 3 ____ %, id 4 ____ %
- [ ] Homography reprojection residual below 1 mm — observed: ____ mm
- [ ] An ISO 7810 ID-1 card reads 85.6 × 54.0 mm with error under 1 mm — observed:
      ____ × ____ mm (error ____ / ____ mm)
- [ ] An ArUco left out of the calibration reads 24.0 ± 0.5 mm — observed: ____ mm
- [ ] A banana measures stably and the measurement locks in under 2 seconds — observed:
      ____ s to lock, reported σ ____ mm (length) / ____ mm (caliper)

Then, to produce the validation table: ≥ 10 bananas through the scale *and* the tape
measure, so the fitted density, the weight error metrics, and the pedicel-trim error all
have real numbers behind them.

## Tests

```bash
.venv/Scripts/python.exe -m pytest
```

262 tests, no camera and no `ultralytics` required: every test that touches the camera
injects a fake `VideoCapture`, and every test that touches the detector injects a fake
model.

A standing rule of this project, learned the hard way: **green tests do not mean the code is
right.** Eleven tests here turned out to pass equally with correct and with wrong
implementations — "approximately X %" assertions, and bounds in one direction only. So every
behaviour a test claims to protect was verified by deliberately breaking the implementation,
confirming the test went red, and reverting. At one point this project had 75 green tests and
a −19.4 % area error.

## Reference

CODEX STAN 205-1997 (CXS 205-1997, amended 2005 and 2022) —
[FAO](https://www.fao.org/input/download/standards/329/CXS_205e.pdf)

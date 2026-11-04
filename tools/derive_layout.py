"""Deriva ``marker_layout`` de un frame del rig y lo escribe en la config.

Para que sirve
--------------
El layout dice donde esta el centro de cada ArUco en milimetros y de el
cuelga toda la metrica. Cada vez que un marcador se mueve hay que volver a
obtenerlo, y hasta ahora el unico camino era medir los cuatro centros con
metro. Esta herramienta lo deriva de un solo frame, a partir del tamano
conocido del marcador (ver ``codex_caliper/layout.py`` para el metodo y sus
limites).

Uso tipico, con el rig montado y la aplicacion CERRADA:

    .venv/Scripts/python.exe -m tools.derive_layout --camera 1
    .venv/Scripts/python.exe -m tools.derive_layout --camera 1 --write

Sin ``--write`` solo informa: muestra el layout derivado, lo bien que
cuadra, y que residual de homografia daria comparado con el que ya esta en
la config. Nada se guarda hasta que se pide.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import cv2

from codex_caliper.aruco_tools import detect_markers
from codex_caliper.camera import open_capture
from codex_caliper.config import AppConfig
from codex_caliper.homography import compute_plane_calibration, mounting_quarter_turns
from codex_caliper.layout import derive_marker_layout

#: Frames que se descartan antes de quedarse con uno. La ELP arranca con la
#: exposicion automatica todavia ajustando y los primeros salen oscuros, que
#: es justo cuando el detector de ArUco pierde esquinas.
_WARMUP_FRAMES = 15

#: Por encima de este RMS la derivacion no es de fiar y la herramienta lo
#: dice en vez de dejar al operador interpretando un numero suelto. El rig
#: real da 0.73 mm con los intrinsecos de relleno, asi que 2 mm deja margen
#: para una sesion peor sin tragarse un montaje que se movio de verdad.
_RMS_WARN_MM = 2.0


def _frame_from_camera(index: int) -> "cv2.Mat":
    capture = open_capture(index)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {index}. Is another application using it?")
    frame = None
    for _ in range(_WARMUP_FRAMES):
        read_ok, candidate = capture.read()
        if read_ok:
            frame = candidate
    capture.release()
    if frame is None:
        raise SystemExit(
            f"Camera {index} opened but delivered no frame. Close the codex-caliper app and retry."
        )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, help="camera index to grab one frame from")
    source.add_argument("--image", type=Path, help="derive from this image instead of a camera")
    parser.add_argument("--config", type=Path, default=Path("config/app.json"))
    parser.add_argument(
        "--write", action="store_true", help="write the derived layout back into the config file"
    )
    args = parser.parse_args()

    cfg = AppConfig.load(args.config)
    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"Could not read image {args.image}")
    else:
        frame = _frame_from_camera(args.camera)

    detections = sorted(detect_markers(frame, cfg), key=lambda d: d.id)
    print(f"Markers detected: {[d.id for d in detections]}")
    if len(detections) != 4:
        raise SystemExit(
            "Need all 4 markers in one frame to derive a layout. Clear whatever is covering "
            "them, check the lighting, and retry."
        )

    for detection in detections:
        print(f"  #{detection.id} glued at {mounting_quarter_turns(detection.corners) * 90} deg")

    derived = derive_marker_layout(detections, cfg)
    print(f"\nSquare-condition RMS: {derived.square_rms_mm:.3f} mm")
    for marker_id, side in sorted(derived.marker_sides_mm.items()):
        print(f"  #{marker_id} side came back as {side:.2f} mm (known {cfg.marker_length_mm} mm)")

    print("\nDerived layout (mm):")
    for marker_id, (x, y) in sorted(derived.centres_mm.items()):
        print(f"  #{marker_id}: ({x:7.2f}, {y:7.2f})")

    before = compute_plane_calibration(detections, cfg)
    after = compute_plane_calibration(
        detections, dataclasses.replace(cfg, marker_layout=derived.centres_mm)
    )
    print(
        f"\nHomography residual on this frame: {before.residual_mm:.3f} mm with the layout "
        f"already in the config, {after.residual_mm:.3f} mm with the derived one."
    )

    if derived.square_rms_mm > _RMS_WARN_MM:
        print(
            f"\nWARNING: {derived.square_rms_mm:.3f} mm is high. The markers may not be flat on "
            "the same plane, one may be peeling off, or the lens distortion is too strong to "
            "ignore. Check the rig before trusting this layout."
        )

    if not args.write:
        print("\nNothing written. Re-run with --write to save this layout into the config.")
        return 0

    if after.residual_mm > before.residual_mm:
        print(
            f"\nRefusing to write: the derived layout fits this frame WORSE than the one "
            f"already in the config ({after.residual_mm:.3f} mm against "
            f"{before.residual_mm:.3f} mm). Something is off -- check the rig and retry."
        )
        return 1

    dataclasses.replace(cfg, marker_layout=derived.centres_mm).save(args.config)
    print(f"\nWritten to {args.config}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

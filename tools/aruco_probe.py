"""Sonda de diagnostico para el montaje de camara + ArUcos.

NO es parte de la implementacion planificada: es una herramienta desechable para
contestar cuatro preguntas antes de escribir codigo de verdad.

  1. Que diccionario ArUco son los marcadores impresos, y que IDs tienen.
  2. Cuanto ocupa un marcador en pixeles (presupuesto optico del spec, 5.1).
  3. Que escala mm/px y que campo de vision da el zoom actual.
  4. Si la deteccion es estable o parpadea (reflejos especulares en el PLA negro).

Uso:
    .venv/Scripts/python.exe tools/aruco_probe.py
    .venv/Scripts/python.exe tools/aruco_probe.py --camera 1

Teclas:
    q / ESC   salir
    r         repetir el escaneo de diccionario
    s         guardar una captura en capturas/
    c         abrir el panel de propiedades del driver (anti-flicker, exposicion)
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

MARKER_LENGTH_MM = 24.0  # lado del cuadrado negro completo
STABILITY_WINDOW = 60  # frames sobre los que se mide la tasa de deteccion
SCAN_FRAMES = 3  # frames usados para identificar el diccionario


def predefined_dictionaries() -> list[tuple[str, int]]:
    """Todos los diccionarios predefinidos, deduplicados por valor.

    OpenCV expone alias con distinta capitalizacion (DICT_APRILTAG_16H5 y
    DICT_APRILTAG_16h5 son el mismo entero); se queda el primer nombre visto.
    """
    seen: dict[int, str] = {}
    for name in dir(cv2.aruco):
        if not name.startswith("DICT_"):
            continue
        value = getattr(cv2.aruco, name)
        if isinstance(value, int) and value not in seen:
            seen[value] = name
    return sorted(((n, v) for v, n in seen.items()), key=lambda kv: kv[0])


def build_detector_params() -> cv2.aruco.DetectorParameters:
    """Parametros ajustados para marcadores impresos en 3D.

    El blanco de la impresion FDM es un gris sucio con textura de capas, asi que
    se ensancha el rango del umbralado adaptativo. El refinamiento subpixel es lo
    que sostiene el objetivo de +/-1 mm del spec.

    perspectiveRemovePixelPerCell = 8 (el defecto es 4) es el ajuste decisivo,
    medido sobre las capturas reales del montaje: con el defecto, el marcador
    id 3 no se decodificaba en ningun frame; con 8 se decodifican 4/4 siempre.

    Al decodificar, ArUco endereza el marcador a un cuadrado canonico y decide
    cada bit muestreando una celda de NxN pixeles. Con N=4 son 16 pixeles por
    bit, y en una pieza impresa en FDM una sola linea de capa o una mota puede
    dominar esa muestra. Con N=8 son 64 pixeles y la textura se promedia.

    Subir errorCorrectionRate tambien hace que aparezca, pero es la palanca
    equivocada: no mejora la muestra, sino que acepta mas bits erroneos. Con ~50
    cuadrilateros espurios por frame procedentes de la veta de la superficie,
    aflojar la validacion invita a falsos positivos. Mejor muestrear mejor que
    validar menos.
    """
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 45
    params.adaptiveThreshWinSizeStep = 6
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.perspectiveRemovePixelPerCell = 8
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return params


def scan_dictionaries(frames: list[np.ndarray]) -> list[tuple[str, int, list[int], int, int]]:
    """Prueba cada diccionario sobre varios frames.

    Devuelve (nombre, valor, ids, detecciones_totales, tam_diccionario) de mejor
    a peor.

    Los diccionarios predefinidos estan ANIDADOS: los primeros 50 codigos de
    DICT_5X5_100 son exactamente DICT_5X5_50, asi que un marcador de ID bajo se
    detecta igual en 50, 100, 250 y 1000, y mirando la imagen es imposible
    distinguirlos. El empate se rompe eligiendo el diccionario mas pequeno:
    menos codigos significa mayor distancia de Hamming entre ellos, es decir
    menos falsos positivos y mas tolerancia a oclusion y mala iluminacion.
    """
    params = build_detector_params()
    results = []
    for name, value in predefined_dictionaries():
        dictionary = cv2.aruco.getPredefinedDictionary(value)
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        ids_seen: set[int] = set()
        total = 0
        for frame in frames:
            _, ids, _ = detector.detectMarkers(frame)
            if ids is not None:
                total += len(ids)
                ids_seen.update(int(i) for i in ids.flatten())
        if total:
            size = int(dictionary.bytesList.shape[0])
            results.append((name, value, sorted(ids_seen), total, size))
    # mas ids distintos y mas detecciones es mejor; a igualdad, gana el
    # diccionario mas pequeno (ver docstring)
    results.sort(key=lambda r: (-len(r[2]), -r[3], r[4]))
    return results


def banding_metrics(frame: np.ndarray) -> tuple[float, float]:
    """Mide bandas horizontales de parpadeo y si amenazan la segmentacion.

    El OV2710 es de obturador rodante: expone fila por fila. Una lampara LED
    regulada por PWM (o el rizado de 120 Hz de la red) se apaga y enciende
    cientos de veces por segundo, asi que cada fila pilla el LED en un punto
    distinto del ciclo y aparecen bandas horizontales.

    Devuelve (amplitud en luminancia en % de la media, amplitud en saturacion en
    UNIDADES ABSOLUTAS de 0-255).

    La segunda cifra va en unidades y no en porcentaje a proposito. Sobre
    cartulina blanca la saturacion media ronda 7 de 255, asi que un porcentaje
    sobre esa media infla cualquier ruido hasta parecer alarmante. Lo que decide
    si el parpadeo importa no es su tamano relativo sino su tamano frente al
    margen que separa la cartulina del umbral de segmentacion, que en este
    montaje es de ~55 unidades. Medido: el parpadeo mueve la saturacion menos de
    1 unidad. Es un problema estetico del video, no metrologico.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = []
    for channel, relative in ((gray, True), (hsv[..., 1], False)):
        rows = channel.mean(axis=1).astype(np.float64)
        trend = cv2.GaussianBlur(rows.reshape(-1, 1), (1, 61), 0).ravel()
        swing = float((rows - trend).std())
        out.append(100.0 * swing / max(rows.mean(), 1e-6) if relative else swing)
    return out[0], out[1]


def marker_side_px(corner_set: np.ndarray) -> float:
    """Lado medio del marcador en pixeles, a partir de sus 4 esquinas."""
    pts = corner_set.reshape(4, 2)
    edges = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(edges))


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return cap
    # A 1080p sobre USB 2.0, MJPEG es la unica opcion que alcanza 30 fps.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def list_cameras(max_index: int = 4) -> list[int]:
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                available.append(i)
        cap.release()
    return available


def draw_hud(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    pad, line_h = 12, 30
    box_h = pad * 2 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (560, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + line_h * (i + 1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None, help="indice de camara")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    if args.camera is None:
        found = list_cameras()
        if not found:
            print("ERROR: ninguna camara responde en los indices 0-3.")
            return 1
        print(f"Camaras disponibles: {found}  ->  usando {found[-1]}")
        print("(si es la equivocada, relanza con --camera N)")
        index = found[-1]
    else:
        index = args.camera

    cap = open_camera(index, args.width, args.height)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir la camara {index}.")
        return 1

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolucion concedida por el driver: {real_w}x{real_h}")
    if (real_w, real_h) != (args.width, args.height):
        print("  OJO: el driver no concedio la resolucion pedida.")
        print("  Los intrinsecos solo valen para la resolucion real.")

    params = build_detector_params()
    detector = None
    dict_name = None
    need_scan = True
    history: dict[int, deque] = {}
    frame_no = 0
    out_dir = Path("capturas")

    print("\nApunta la camara a los 4 marcadores. Escaneando diccionario...\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: se perdio el flujo de video.")
            break
        frame_no += 1

        if need_scan:
            frames = [frame]
            for _ in range(SCAN_FRAMES - 1):
                ok2, f2 = cap.read()
                if ok2:
                    frames.append(f2)
            results = scan_dictionaries(frames)
            if results:
                dict_name, dict_value, ids, total, size = results[0]
                detector = cv2.aruco.ArucoDetector(
                    cv2.aruco.getPredefinedDictionary(dict_value), params)
                need_scan = False
                history.clear()
                print("=" * 66)
                print(f"  DICCIONARIO: {dict_name}   ({size} codigos)")
                print(f"  IDs:         {ids}")
                print("=" * 66)
                nested = [r for r in results[1:] if r[2] == ids]
                if nested:
                    print("Diccionarios anidados que dan el mismo resultado:")
                    print("  " + ", ".join(f"{r[0]} ({r[4]})" for r in nested))
                    print("  Se eligio el mas pequeno: mayor distancia de Hamming,")
                    print("  menos falsos positivos. Anota este nombre en la config.")
                others = [r for r in results[1:] if r[2] != ids]
                if others:
                    print("Otros diccionarios con detecciones distintas (sospechosos):")
                    for name, _, other_ids, _, osize in others[:3]:
                        print(f"    {name:<26} ids={other_ids}")
                print()
            else:
                cv2.putText(frame, "No markers found - show the 4 ArUcos to the camera",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("ArUco probe", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

        corners, ids, _ = detector.detectMarkers(frame)
        display = frame.copy()
        detected_ids: list[int] = []
        sides: list[float] = []

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            for corner_set, marker_id in zip(corners, ids.flatten()):
                mid = int(marker_id)
                detected_ids.append(mid)
                side = marker_side_px(corner_set)
                sides.append(side)
                centre = corner_set.reshape(4, 2).mean(axis=0).astype(int)
                cv2.putText(display, f"{side:.0f}px", (centre[0] - 30, centre[1] + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        for mid in set(list(history.keys()) + detected_ids):
            history.setdefault(mid, deque(maxlen=STABILITY_WINDOW)).append(mid in detected_ids)

        count_colour = (0, 255, 0) if len(detected_ids) >= 4 else (
            (0, 200, 255) if len(detected_ids) == 3 else (0, 0, 255))
        lines = [
            (f"Dictionary: {dict_name}", (255, 255, 255)),
            (f"Markers: {len(detected_ids)}/4   ids={sorted(detected_ids)}", count_colour),
        ]

        if sides:
            mean_side = float(np.mean(sides))
            mm_per_px = MARKER_LENGTH_MM / mean_side
            fov_w = real_w * mm_per_px
            fov_h = real_h * mm_per_px
            size_colour = (0, 255, 0) if mean_side >= 40 else (0, 0, 255)
            fov_colour = (0, 255, 0) if 300 <= fov_w <= 460 else (0, 200, 255)
            lines += [
                (f"Marker side: {mean_side:.0f} px  (min 40)", size_colour),
                (f"Scale: {mm_per_px:.4f} mm/px", (255, 255, 255)),
                (f"Field of view: {fov_w:.0f} x {fov_h:.0f} mm  (target ~395x221)", fov_colour),
            ]

        if history:
            rates = []
            for mid in sorted(history):
                seen = history[mid]
                rates.append(f"{mid}:{100.0 * sum(seen) / len(seen):.0f}%")
            worst = min(100.0 * sum(s) / len(s) for s in history.values())
            stab_colour = (0, 255, 0) if worst >= 98 else (
                (0, 200, 255) if worst >= 90 else (0, 0, 255))
            lines.append((f"Stability: {'  '.join(rates)}", stab_colour))

        band_y, band_s = banding_metrics(frame)
        band_colour = (0, 255, 0) if band_y < 1.0 else (
            (0, 200, 255) if band_y < 3.0 else (0, 0, 255))
        lines.append((f"Flicker banding: {band_y:.1f}% luma / {band_s:.2f} sat units"
                      f" (harmless below ~5)", band_colour))
        lines.append(("q quit   r rescan   s snapshot   c driver settings",
                      (180, 180, 180)))
        draw_hud(display, lines)

        shown = display
        if real_w > 1280:
            shown = cv2.resize(display, (1280, int(1280 * real_h / real_w)))
        cv2.imshow("ArUco probe", shown)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            need_scan = True
            print("Reescaneando diccionario...\n")
        if key == ord("c"):
            # Panel de propiedades de DirectShow: ahi vive "Powerline Frequency
            # (Anti Flicker)", que es el arreglo real del parpadeo.
            print("Abriendo el panel del driver (busca 'Powerline Frequency')...")
            cap.set(cv2.CAP_PROP_SETTINGS, 1)
        if key == ord("s"):
            out_dir.mkdir(exist_ok=True)
            path = out_dir / f"probe_{int(time.time())}.png"
            cv2.imwrite(str(path), frame)
            print(f"Captura guardada: {path}")

    cap.release()
    cv2.destroyAllWindows()

    if history:
        print("\nResumen de estabilidad sobre los ultimos frames:")
        for mid in sorted(history):
            seen = history[mid]
            print(f"  id {mid}: {100.0 * sum(seen) / len(seen):.1f}% de deteccion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

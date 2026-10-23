"""Identifica a que diccionario ArUco predefinido pertenecen unos marcadores.

Herramienta de calibracion: se le da una imagen con marcadores fisicos y
recorre todos los `cv2.aruco.DICT_*` predefinidos probando cuales decodifican
algo, para saber que diccionario anotar en `AppConfig.aruco_dictionary`.

Uso:
    .venv/Scripts/python.exe tools/find_aruco_dictionary.py captura.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DictionaryMatch:
    """Un diccionario predefinido que decodifico marcadores en la imagen."""

    name: str
    ids: list[int]
    count: int


def _predefined_dictionaries() -> list[tuple[str, int]]:
    """Todos los diccionarios ArUco predefinidos, deduplicados por valor entero.

    OpenCV expone alias con distinta capitalizacion para el mismo diccionario
    (p.ej. `DICT_APRILTAG_16H5` y `DICT_APRILTAG_16h5` son el mismo entero);
    se conserva solo el primer nombre visto por cada valor.
    """
    seen: dict[int, str] = {}
    for name in dir(cv2.aruco):
        if not name.startswith("DICT_"):
            continue
        value = getattr(cv2.aruco, name)
        if isinstance(value, int) and value not in seen:
            seen[value] = name
    return sorted(((name, value) for value, name in seen.items()), key=lambda item: item[1])


def find_dictionary(image: np.ndarray, expected_count: int = 4) -> list[DictionaryMatch]:
    """Prueba todos los diccionarios ArUco predefinidos contra `image`.

    Devuelve los diccionarios que detectaron al menos un marcador, ordenados
    por cercania a `expected_count` y luego por numero de marcadores
    detectados (mas es mejor).

    Los diccionarios predefinidos estan anidados: los primeros 50 codigos de
    `DICT_5X5_100` son exactamente `DICT_5X5_50`, asi que un marcador de ID
    bajo decodifica igual en 50/100/250/1000 y es indistinguible mirando solo
    la imagen. El empate se rompe eligiendo el diccionario mas chico (menos
    codigos, mayor distancia de Hamming, menos falsos positivos).
    """
    params = cv2.aruco.DetectorParameters()
    candidates: list[tuple[str, list[int], int]] = []
    for name, value in _predefined_dictionaries():
        dictionary = cv2.aruco.getPredefinedDictionary(value)
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        _, ids, _ = detector.detectMarkers(image)
        if ids is None or len(ids) == 0:
            continue
        found_ids = sorted(int(i) for i in ids.flatten())
        dictionary_size = int(dictionary.bytesList.shape[0])
        candidates.append((name, found_ids, dictionary_size))

    candidates.sort(key=lambda c: (abs(len(c[1]) - expected_count), -len(c[1]), c[2]))
    return [DictionaryMatch(name=name, ids=ids, count=len(ids)) for name, ids, _ in candidates]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Identify the predefined ArUco dictionary of an image."
    )
    parser.add_argument("image", type=Path, help="Path to the image containing the markers")
    parser.add_argument(
        "--expected-count", type=int, default=4, help="Expected number of markers (default: 4)"
    )
    args = parser.parse_args(argv)

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"ERROR: could not read image {args.image}")
        return 1

    matches = find_dictionary(image, expected_count=args.expected_count)
    if not matches:
        print("No ArUco markers were detected in the image.")
        return 1

    print(f"{'Dictionary':<26} {'#':>3}  IDs")
    for match in matches:
        print(f"{match.name:<26} {match.count:>3}  {match.ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

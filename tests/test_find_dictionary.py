import cv2
import numpy as np

from tools.find_aruco_dictionary import find_dictionary


def _canvas_with_markers(dictionary_id=cv2.aruco.DICT_5X5_50, ids=(0, 1, 2, 3), marker_px=120, margin=40):
    """Dibuja `ids` marcadores de `dictionary_id` sobre un lienzo blanco, en una fila."""
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    cell = marker_px + margin
    canvas = np.full((cell + margin, cell * len(ids) + margin, 3), 255, np.uint8)
    for i, marker_id in enumerate(ids):
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        x = margin + i * cell
        y = margin // 2
        canvas[y:y + marker_px, x:x + marker_px] = marker_bgr
    return canvas


def test_identifies_the_dictionary_a_marker_was_drawn_with():
    canvas = _canvas_with_markers()
    matches = find_dictionary(canvas, expected_count=4)
    assert matches[0].name == "DICT_5X5_50"
    assert sorted(matches[0].ids) == [0, 1, 2, 3]


def test_returns_empty_list_when_no_markers_present():
    assert find_dictionary(np.full((400, 400, 3), 255, np.uint8)) == []

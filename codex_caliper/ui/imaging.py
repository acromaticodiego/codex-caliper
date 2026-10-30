"""Conversion de pixeles de OpenCV a Qt, compartida por toda la interfaz.

``bgr_to_qpixmap`` estaba duplicada literalmente en ``video_view.py``
(``_ndarray_bgr_to_qpixmap``) y en ``calibration_view.py``
(``_bgr_to_qpixmap``), con un comentario que justificaba la duplicacion
("ese helper es privado de video_view.py"). La justificacion era sobre la
VISIBILIDAD del helper, no sobre su contenido: las dos copias eran
identicas linea por linea, incluida la parte que importa de verdad (la
copia del buffer), asi que la forma correcta de resolver esa objecion es
un modulo compartido donde el helper sea publico, no una segunda copia.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap


def bgr_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """Convierte un arreglo BGR de OpenCV a ``QPixmap``.

    ``QImage.copy()`` es obligatorio: el buffer de ``frame`` suele ser un
    arreglo de numpy local al llamador, y sin copiar el ``QImage`` quedaria
    apuntando a memoria que puede liberarse en cuanto ese arreglo sale de
    alcance -- un pixmap corrupto sin ninguna excepcion que lo delate.
    """
    frame = np.ascontiguousarray(frame)
    height, width = frame.shape[:2]
    bytes_per_line = frame.strides[0]
    image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(image.copy())

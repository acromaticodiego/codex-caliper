"""Panel lateral: una tarjeta por ``track_id`` (Task 18, spec S8 variante 1).

**El ancho es fijo, no un factor de estiramiento.** ``setFixedWidth`` (en vez
de dejar que un ``QHBoxLayout`` lo reparta por stretch) es lo que hace que
al maximizar la ventana crezca el video y no el panel -- la decision
vinculante de la encargante, no una preferencia de implementacion.

El contenido de cada tarjeta reusa ``compose_label_lines`` de
``overlay.py`` (Task 17) en vez de reimplementar el formato del veredicto
y las medidas: es la misma funcion PURA que decide que mostrar sobre el
video, asi que el panel y el overlay nunca pueden mostrar numeros
distintos para el mismo fruto.
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget

from codex_caliper.pipeline import BananaResult
from codex_caliper.ui.overlay import compose_label_lines

#: Ancho fijo del panel, en pixeles (spec S8, vinculante).
SIDE_PANEL_WIDTH_PX = 240

#: Escala por defecto del lienzo (spec S5.1), igual al default de
#: ``format_measurement``. ``update_results`` acepta un ``mm_per_px``
#: explicito para la calibracion real del cuadro; este default solo cubre
#: a un llamador que lo omita.
_DEFAULT_MM_PER_PX = 0.2

# Paleta: mismos tokens que ``video_view.py``/``main_window.py`` (instrumento
# de medicion, no tarjetas SaaS -- ver el informe de la tarea para la
# justificacion completa de la paleta).
_SURFACE_HEX = "#1D2227"
_RULE_HEX = "#333A42"
_INK_PRIMARY_HEX = "#E7EAED"
_INK_MUTED_HEX = "#8A929B"
_SIGNAL_PASS_HEX = "#5FBF8F"
_SIGNAL_FAIL_HEX = "#E0676B"

_EMPTY_STATE_TEXT = "No fruit detected"


def _header_color(result: BananaResult) -> str:
    """Color del encabezado de la tarjeta.

    El color es un REFUERZO, no el unico canal: el texto de
    ``compose_label_lines`` ya dice "OK"/"FAIL" o el id a secas mientras se
    mide, asi que perder el color (p.ej. en escala de grises) no pierde la
    informacion.
    """
    if not result.aggregate.locked:
        return _INK_MUTED_HEX
    return _SIGNAL_PASS_HEX if result.verdict.passes else _SIGNAL_FAIL_HEX


def _build_card(result: BananaResult, mm_per_px: float) -> QFrame:
    """Construye la tarjeta de un ``BananaResult``, con las lineas de
    ``compose_label_lines`` -- la misma composicion que ``draw_results``
    pone sobre el video."""
    card = QFrame()
    card.setStyleSheet(
        f"QFrame {{ background-color: {_SURFACE_HEX}; border: 1px solid {_RULE_HEX}; }}"
    )
    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(3)

    lines = compose_label_lines(result, mm_per_px)
    header_color = _header_color(result)

    header = QLabel(lines[0])
    header.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
    header.setStyleSheet(f"color: {header_color}; border: none;")
    layout.addWidget(header)

    mono_font = QFont("Consolas", 10)
    for line in lines[1:]:
        label = QLabel(line)
        label.setFont(mono_font)
        label.setStyleSheet(f"color: {_INK_PRIMARY_HEX}; border: none;")
        layout.addWidget(label)

    return card


class SidePanel(QWidget):
    """Panel de 240 px fijos con una tarjeta por fruto rastreado.

    ``update_results`` reconstruye las tarjetas en cada llamada: el numero
    de frutos visibles cambia cuadro a cuadro (un fruto entra o sale de
    escena) y no hay estado que valga la pena conservar entre tarjetas --
    ``compose_label_lines`` ya es una funcion pura del ``BananaResult``
    actual.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(SIDE_PANEL_WIDTH_PX)
        self.setStyleSheet(f"background-color: {_SURFACE_HEX};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent; border: none;")
        outer.addWidget(scroll)

        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(8, 8, 8, 8)
        self._cards_layout.setSpacing(8)
        self._cards_layout.addStretch(1)
        scroll.setWidget(self._cards_container)

        # Estado inicial: el mismo camino que cualquier otra actualizacion
        # ("sin resultados" todavia), para no duplicar la construccion del
        # aviso vacio en dos sitios distintos.
        self.update_results([])

    def update_results(self, results: list[BananaResult], mm_per_px: float = _DEFAULT_MM_PER_PX) -> None:
        """Reconstruye las tarjetas a partir de los resultados del cuadro actual."""
        while self._cards_layout.count() > 1:
            item = self._cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # ``takeAt`` solo lo saca de la contabilidad del layout --
                # el widget sigue siendo hijo visible de ``_cards_container``
                # con su geometria vieja hasta que de verdad se destruye, y
                # ``deleteLater`` por si sola solo AGENDA esa destruccion
                # para una vuelta futura del bucle de eventos. Sin
                # ``setParent(None)`` la tarjeta del cuadro anterior queda
                # dibujada encima/detras de las nuevas durante ese hueco --
                # visible en una captura, no en ningun conteo de tarjetas.
                widget.setParent(None)
                widget.deleteLater()

        if not results:
            label = QLabel(_EMPTY_STATE_TEXT)
            label.setFont(QFont("Segoe UI", 9))
            label.setStyleSheet(f"color: {_INK_MUTED_HEX};")
            self._cards_layout.insertWidget(0, label)
            return

        for result in results:
            self._cards_layout.insertWidget(self._cards_layout.count() - 1, _build_card(result, mm_per_px))

"""Panel lateral: una tarjeta por ``track_id`` (Task 18, spec S8 variante 1).

**El ancho es fijo, no un factor de estiramiento.** ``setFixedWidth`` (en vez
de dejar que un ``QHBoxLayout`` lo reparta por stretch) es lo que hace que
al maximizar la ventana crezca el video y no el panel -- la decision
vinculante de la encargante, no una preferencia de implementacion.

El contenido sale de ``readout.compose_readout`` -- la misma fuente de la
que ``overlay.py`` saca el rotulo del video -- en vez de reimplementar el
formato: panel y overlay no pueden discrepar sobre el mismo fruto.

Aqui vive la lectura COMPLETA y sobre el video solo queda un rotulo de una
linea. El reparto es deliberado: sobre el fruto el texto competia con la
curva Codex (la protagonista, y literalmente la geometria detras del
numero) y quedaba ilegible en gris sobre amarillo; aqui hay sitio para
decir contra que minimo de CODEX STAN 205 se compara cada medida, que es
lo que vuelve la cifra entendible por alguien que no escribio el programa.

Cada fila es "nombre / valor / minimo exigido", y debajo va el contraste
cuerda-contra-arco: el argumento del proyecto en dos numeros y un
porcentaje -- una caja contenedora mide la recta entre las puntas, la
norma exige el arco.
"""

from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from codex_caliper.pipeline import BananaResult
from codex_caliper.ui.readout import MeasurementReadout, compose_readout
from codex_caliper.ui.theme import (
    INK_MUTED_HEX,
    INK_PRIMARY_HEX,
    RULE_HEX,
    SIGNAL_FAIL_HEX,
    SIGNAL_PASS_HEX,
    SIGNAL_WARN_HEX,
    SURFACE_HEX,
)

#: Ancho fijo del panel, en pixeles (spec S8, vinculante).
SIDE_PANEL_WIDTH_PX = 290

_EMPTY_STATE_TEXT = "No fruit detected"


def _verdict_color(readout: MeasurementReadout) -> str:
    """Color del veredicto. REFUERZO, nunca el unico canal: el texto ya
    dice "PASS"/"FAIL" o "MEASURING n/m", asi que en escala de grises no
    se pierde nada."""
    if not readout.locked:
        return INK_MUTED_HEX
    return SIGNAL_PASS_HEX if readout.verdict == "PASS" else SIGNAL_FAIL_HEX


def _row_color(meets: bool | None) -> str:
    """``None`` -- todavia midiendo -- se pinta neutro, no rojo: "aun no se
    sabe" no es "no cumple"."""
    if meets is None:
        return INK_PRIMARY_HEX
    return SIGNAL_PASS_HEX if meets else SIGNAL_FAIL_HEX


def _label(text: str, *, size: int, color: str, bold: bool = False,
           mono: bool = False, spacing: float = 0.0) -> QLabel:
    label = QLabel(text)
    font = QFont("Consolas" if mono else "Segoe UI", size)
    if bold:
        font.setWeight(QFont.Weight.DemiBold)
    if spacing:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    label.setFont(font)
    label.setStyleSheet(f"color: {color}; border: none;")
    return label


def _build_measurement_row(row, layout: QVBoxLayout) -> None:
    """Una medida: nombre en versalitas apagadas, cifra grande, y debajo el
    minimo de la norma con el que se la juzga.

    El minimo no es decoracion: "271 mm" a secas no dice si cumple, y es
    justo lo que el encargo señalaba como no entendible por nadie salvo el
    autor.
    """
    layout.addWidget(_label(row.name, size=8, color=INK_MUTED_HEX, spacing=1.2))

    value_line = QWidget()
    value_layout = QHBoxLayout(value_line)
    value_layout.setContentsMargins(0, 0, 0, 0)
    value_layout.setSpacing(6)
    value_layout.addWidget(_label(row.value, size=14, color=_row_color(row.meets), bold=True, mono=True))
    value_layout.addStretch(1)
    if row.meets is not None:
        value_layout.addWidget(
            _label("PASS" if row.meets else "FAIL", size=9, color=_row_color(row.meets), bold=True)
        )
    layout.addWidget(value_line)

    layout.addWidget(_label(f"CODEX {row.requirement}", size=8, color=INK_MUTED_HEX))


def _build_card(result: BananaResult, mm_per_px: float) -> QFrame:
    """Ficha completa de un fruto, compuesta desde ``compose_readout``."""
    readout = compose_readout(result, mm_per_px)

    card = QFrame()
    card.setStyleSheet(
        f"QFrame {{ background-color: {SURFACE_HEX}; border: 1px solid {RULE_HEX}; }}"
    )
    layout = QVBoxLayout(card)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(2)

    header = QWidget()
    header_layout = QHBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.addWidget(_label(readout.title, size=10, color=INK_PRIMARY_HEX, bold=True, spacing=0.8))
    header_layout.addStretch(1)
    header_layout.addWidget(
        _label(readout.verdict or readout.status, size=9, color=_verdict_color(readout), bold=True)
    )
    layout.addWidget(header)
    if readout.verdict is not None:
        layout.addWidget(_label(readout.status, size=8, color=INK_MUTED_HEX))

    for row in readout.rows:
        layout.addSpacing(8)
        _build_measurement_row(row, layout)

    layout.addSpacing(10)
    rule = QFrame()
    rule.setFrameShape(QFrame.Shape.HLine)
    rule.setStyleSheet(f"color: {RULE_HEX}; border: none; background-color: {RULE_HEX};")
    rule.setFixedHeight(1)
    layout.addWidget(rule)
    layout.addSpacing(6)

    # El argumento del proyecto, en tres lineas: la recta entre los dos
    # extremos del fruto, el arco que la norma exige entre esos MISMOS dos
    # extremos, y cuanto se separan.
    #
    # La frase dice "the arc is X% longer" y no "a bounding box underreads
    # by X%": lo que se mide es la cuerda entre los extremos del arco, no
    # el lado de una caja contenedora. Son parecidos y no son lo mismo, y
    # afirmar el segundo seria publicar una comparacion que este programa
    # no ha hecho.
    layout.addWidget(
        _label("STRAIGHT LINE vs CODEX ARC", size=8, color=INK_MUTED_HEX, spacing=1.2)
    )
    layout.addWidget(
        _label(f"straight chord   {readout.chord_mm:6.1f} mm", size=9, color=INK_MUTED_HEX, mono=True)
    )
    layout.addWidget(
        _label(f"Codex arc        {readout.arc_mm:6.1f} mm", size=9, color=INK_PRIMARY_HEX, mono=True)
    )
    layout.addWidget(
        _label(
            f"the arc is {readout.curvature_gain_pct:.0f}% longer",
            size=9,
            color=SIGNAL_WARN_HEX,
            bold=True,
        )
    )

    if readout.weight_g is not None:
        layout.addSpacing(8)
        layout.addWidget(_label("ESTIMATED WEIGHT", size=8, color=INK_MUTED_HEX, spacing=1.2))
        layout.addWidget(
            _label(f"{readout.weight_g} g", size=12, color=INK_PRIMARY_HEX, bold=True, mono=True)
        )

    return card


class SidePanel(QWidget):
    """Panel de ancho fijo con una ficha por fruto rastreado.

    ``update_results`` reconstruye las tarjetas en cada llamada: el numero
    de frutos visibles cambia cuadro a cuadro (un fruto entra o sale de
    escena) y no hay estado que valga la pena conservar entre tarjetas --
    ``compose_readout`` ya es una funcion pura del ``BananaResult``
    actual.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(SIDE_PANEL_WIDTH_PX)
        self.setStyleSheet(f"background-color: {SURFACE_HEX};")

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

    def update_results(self, results: list[BananaResult], mm_per_px: float | None = None) -> None:
        """Reconstruye las tarjetas a partir de los resultados del cuadro actual.

        ``mm_per_px`` solo se usa para FORMATEAR tarjetas, asi que es
        opcional unicamente en el caso en que no hay ninguna que formatear.
        La version anterior tenia un default de 0.2 (la escala del lienzo del
        rig): probablemente muerto -- el unico llamador que lo omitia es el
        ``update_results([])`` del constructor, y ``results`` vacio no
        formatea nada -- pero un default plausible en una firma publica es
        una invitacion a medir con una escala que nadie calibro. Si llegan
        resultados sin escala, se lanza en vez de inventarla.
        """
        if results and mm_per_px is None:
            raise ValueError(
                "update_results needs the frame's real mm_per_px to format results; "
                "there is no safe default scale for a measurement"
            )
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
            label.setStyleSheet(f"color: {INK_MUTED_HEX};")
            self._cards_layout.insertWidget(0, label)
            return

        for result in results:
            self._cards_layout.insertWidget(self._cards_layout.count() - 1, _build_card(result, mm_per_px))

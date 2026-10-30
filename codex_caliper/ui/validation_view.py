"""Vista de validacion: tabla y metricas de error (Task 19, Task 15).

Pantalla SEPARADA de la operacion en vivo (spec S8): la tabla de
validacion y su R² son un resultado PUBLICABLE (spec S7 etapa 10), no una
pantalla de depuracion que competiria por espacio con el video.

El adaptador ``TrackAggregate`` -> ``BananaMeasures`` (Ruling 1 del encargo)
------------------------------------------------------------------------------
El plan original asumia que esta vista podia construir un ``BananaMeasures``
completo a partir de un ``TrackAggregate`` LOCKED para alimentar
``ValidationLog.record``. Pero ``TrackAggregate`` (Task 11) solo agrega
``length_mm``, ``caliper_mm`` y ``volume_mm3`` -- Task 11 llamo a
``area_mm2`` y ``width_profile`` "auxiliares" y deliberadamente no los
agrego sobre la ventana temporal. Reabrir ``TrackAggregate`` para cargar el
area es exactamente lo que el encargo prohibe (interfaz ya revisada en
Task 11), y FABRICAR un area -- re-integrando un perfil que nunca se
guardo, o reusando el de un solo frame en vez de la mediana de la ventana --
pondria en la tabla de validacion un numero que ninguna medida respalda:
el anti-objetivo exacto de la spec S4 ("reportar cifras sin error
asociado").

Por eso ``measures_from_aggregate`` marca ``area_mm2`` como ``NaN`` y
``width_profile`` como arreglos vacios. Esto no le cuesta ninguna cifra
publicada: ``fit_density``/``error_metrics`` (Task 15) regresan sobre
VOLUMEN, nunca sobre area, y la columna ``area_mm2`` del CSV existe para
las muestras de patron plano que pasan por ``dimensional_check`` (no por
aqui) -- un ``NaN`` en esta columna, para una muestra de banano real, es
honesto ("no medido") en vez de mentir con un numero plausible.

Por que una fila NO LOCKED se rechaza
----------------------------------------
``record_sample`` exige ``aggregate.locked``: una medida que todavia no se
asento (ventana sin llenar, o la desviacion de longitud/calibre todavia por
encima del umbral de Task 11) es exactamente el tipo de cifra
plausible-pero-cambiante que el proyecto ya evito en otros cinco sitios.
Entrar a la bitacora de validacion es PERMANENTE (el CSV se reescribe
completo en cada ``record``, Task 15) -- a diferencia de la interfaz en
vivo, donde una medida inestable solo se ve un instante y el frame
siguiente la corrige, una fila mala en el CSV corrompe la regresion de
densidad para siempre hasta que alguien la edite a mano.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from codex_caliper.aggregation import TrackAggregate
from codex_caliper.geometry.measures import BananaMeasures
from codex_caliper.ui.theme import (
    BG_VOID_HEX,
    INK_MUTED_HEX,
    INK_PRIMARY_HEX,
    RULE_HEX,
    SURFACE_HEX,
)
from codex_caliper.validation import MIN_DENSITY_SAMPLES, ValidationLog, error_metrics, fit_density

#: Cadena mostrada para un valor NaN en la tabla (p.ej. ``area_mm2`` de una
#: muestra que vino de un ``TrackAggregate``, nunca medida). Nunca "0" (eso
#: diria "area cero", un dato distinto de "no medido") ni "nan" crudo (poco
#: legible para un operador que no programa).
_UNAVAILABLE_CELL_TEXT = "—"

#: Columnas de la tabla, en el mismo orden que ``validation._COLUMNS`` --
#: esta vista no decide un orden propio para los mismos datos.
_COLUMNS = [
    "sample_id",
    "codex_length_mm",
    "caliper_mm",
    "area_mm2",
    "volume_mm3",
    "actual_weight_g",
    "actual_length_mm",
]

_COLUMN_HEADERS = {
    "sample_id": "Sample",
    "codex_length_mm": "Length (mm)",
    "caliper_mm": "Caliper (mm)",
    "area_mm2": "Area (mm²)",
    "volume_mm3": "Volume (mm³)",
    "actual_weight_g": "Actual weight (g)",
    "actual_length_mm": "Actual length (mm)",
}


class ValidationView(QWidget):
    """Tabla de muestras de validacion + metricas de error (Task 15).

    No posee el camino de camara/pipeline: recibe un ``TrackAggregate`` ya
    LOCKED (de la ventana en vivo, Task 11) y un peso/longitud reales
    capturados a mano (bascula, regla) a traves de ``record_sample``. Lo
    que hace propio es el adaptador hacia ``BananaMeasures`` (Ruling 1) y
    la presentacion de la tabla/metricas -- ``ValidationLog`` sigue siendo
    la unica duena de la persistencia en CSV.
    """

    def __init__(self, log: ValidationLog, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log = log
        self._summary_text = ""

        self.setStyleSheet(f"background-color: {BG_VOID_HEX};")

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels([_COLUMN_HEADERS[c] for c in _COLUMNS])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setStyleSheet(
            f"QTableWidget {{ background-color: {SURFACE_HEX}; color: {INK_PRIMARY_HEX}; "
            f"gridline-color: {RULE_HEX}; border: none; }}"
            f"QHeaderView::section {{ background-color: {SURFACE_HEX}; color: {INK_MUTED_HEX}; "
            f"border: none; border-bottom: 1px solid {RULE_HEX}; padding: 4px; }}"
        )

        self._summary_label = QLabel()
        self._summary_label.setFont(QFont("Consolas", 10))
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet(f"color: {INK_PRIMARY_HEX}; padding: 8px;")

        layout = QVBoxLayout(self)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._summary_label)

        self.refresh()

    # -- Adaptador TrackAggregate -> BananaMeasures (Ruling 1) ---------------

    @staticmethod
    def measures_from_aggregate(aggregate: TrackAggregate) -> BananaMeasures:
        """Construye un ``BananaMeasures`` a partir de un ``TrackAggregate``
        LOCKED, para alimentar ``ValidationLog.record``.

        ``area_mm2`` queda ``NaN`` y ``width_profile`` queda vacio -- ver el
        docstring del modulo para por que, en vez de fabricar cualquiera de
        los dos. ``codex_length_mm``, ``caliper_mm`` y ``volume_mm3`` son
        exactamente los campos que ``TrackAggregate`` SI agrega (la mediana
        de la ventana temporal, Task 11), sin ninguna transformacion.
        """
        return BananaMeasures(
            codex_length_mm=aggregate.length_mm,
            caliper_mm=aggregate.caliper_mm,
            area_mm2=float("nan"),
            volume_mm3=aggregate.volume_mm3,
            width_profile=(np.array([]), np.array([])),
        )

    def record_sample(
        self,
        sample_id: str,
        aggregate: TrackAggregate,
        actual_weight_g: float,
        actual_length_mm: float | None = None,
    ) -> None:
        """Registra una muestra de validacion a partir de un agregado LOCKED.

        Lanza ``ValueError`` si ``aggregate`` todavia no esta asentado (ver
        el docstring del modulo): una fila mala en el CSV de validacion es
        permanente, a diferencia de una medida inestable en el video en
        vivo, que el frame siguiente corrige solo.
        """
        if not aggregate.locked:
            raise ValueError(
                f"cannot record a validation sample from track {aggregate.track_id}: its "
                "measurement has not settled yet (aggregate.locked is False). Wait for the "
                "temporal window to lock before recording it."
            )
        measures = self.measures_from_aggregate(aggregate)
        self._log.record(sample_id, measures, actual_weight_g, actual_length_mm)
        self.refresh()

    # -- Presentacion ----------------------------------------------------------

    def refresh(self) -> None:
        """Vuelve a leer el log y reconstruye tabla y resumen."""
        frame = self._log.to_frame()
        self._rebuild_table(frame)
        self._summary_text = self._compute_summary_text(frame)
        self._summary_label.setText(self._summary_text)

    def summary_text(self) -> str:
        return self._summary_text

    def sample_count(self) -> int:
        return len(self._log.to_frame())

    def _column_index(self, column: str) -> int:
        return _COLUMNS.index(column)

    def _rebuild_table(self, frame: pd.DataFrame) -> None:
        self._table.setRowCount(len(frame))
        for row_idx in range(len(frame)):
            row = frame.iloc[row_idx]
            for col_idx, column in enumerate(_COLUMNS):
                value = row[column]
                item = QTableWidgetItem(self._format_cell(column, value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(row_idx, col_idx, item)

    @staticmethod
    def _format_cell(column: str, value) -> str:
        if column == "sample_id":
            return str(value)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return _UNAVAILABLE_CELL_TEXT
        return f"{float(value):.1f}"

    def _compute_summary_text(self, frame: pd.DataFrame) -> str:
        """Decide deliberadamente que mostrar en cada uno de los tres
        estados posibles -- nunca deja que ``fit_density``/``error_metrics``
        exploten en silencio ni que se imprima "MAE: nan g" (spec S4).

        1. Cero muestras: el primer estado en que CUALQUIER instancia de
           esta vista va a estar (el log recien creado). Dice cuantas
           muestras hacen falta, no solo "sin datos" -- una pantalla vacia
           es una invitacion a actuar, no un callejon sin salida.
        2. Menos de ``MIN_DENSITY_SAMPLES``: ``fit_density`` lanzaria
           ``ValueError``; se absorbe con una pantalla de progreso, nunca
           con un traceback.
        3. Suficientes muestras: ajusta la densidad y reporta MAE/MAPE/R²
           reales, con su cuenta de muestras (nunca una cifra de error sin
           decir sobre cuantas se calculo -- regla del modulo
           ``validation.py``).
        """
        n = len(frame)
        if n == 0:
            return (
                "No validation samples recorded yet. Record at least "
                f"{MIN_DENSITY_SAMPLES} samples (weighed on a scale) to fit the "
                "density regression and report error metrics."
            )
        if n < MIN_DENSITY_SAMPLES:
            return (
                f"Collecting samples: {n}/{MIN_DENSITY_SAMPLES} needed to fit the "
                "density regression. Keep recording locked measurements with their "
                "real weight."
            )

        density = fit_density(frame)
        metrics = error_metrics(frame, density)
        summary = (
            f"Density {density.g_per_cm3:.3f} g/cm³ (R² {density.r_squared:.3f}, n={density.n})  |  "
            f"MAE {metrics.mae_g:.1f} g  MAPE {metrics.mape_pct:.1f}%  R² {metrics.r_squared:.3f}  "
            f"(n={metrics.n})"
        )
        if metrics.mae_length_mm is not None:
            summary += f"  |  Length MAE {metrics.mae_length_mm:.1f} mm (n={metrics.n_length})"
        return summary

"""Paleta unica de la interfaz: instrumento de medicion, no tablero SaaS.

Los ocho tokens de abajo estaban declarados por separado en
``video_view.py``, ``side_panel.py``, ``main_window.py``,
``calibration_view.py`` y ``validation_view.py`` -- veinte literales
hexadecimales repetidos a mano, con el comentario "mismos tokens que ..."
en cada modulo como unica garantia de que siguieran coincidiendo.

Eso no es un problema de estetica del codigo: el autor va a ajustar la
paleta antes de grabar el video de presentacion (spec S13), y la deriva
entre la vista en vivo y la vista de calibracion -- dos pantallas que el
espectador ve una detras de la otra -- es exactamente lo que se nota en
una grabacion. Un solo modulo hace que ese ajuste sea un cambio, no cinco.

Por que no es el fondo "#111" generico
---------------------------------------
El fondo es un grafito FRIO, no casi-negro: mantiene el video (no el
chrome) como lo mas brillante del cuadro, sin el contraste brutal de un
negro puro contra un lienzo blanco de cartulina. Los colores de senal
estan desaturados a proposito -- un verde/rojo de semaforo saturado le
daria al estado de calibracion mas peso visual que a la fruta medida.

Los colores de ``overlay.py`` NO viven aqui: son tuplas BGR para
``cv2.putText`` sobre el lienzo (otra convencion de canales y otro medio),
no hojas de estilo de Qt.
"""

from __future__ import annotations

#: Fondo de la ventana y de las vistas (grafito frio).
BG_VOID_HEX = "#171B20"

#: Fondo de los paneles/tarjetas/barras que se levantan sobre el fondo.
SURFACE_HEX = "#1D2227"

#: Lineas divisorias y bordes de tarjeta.
RULE_HEX = "#333A42"

#: Texto principal.
INK_PRIMARY_HEX = "#E7EAED"

#: Texto secundario (etiquetas, estados "sin datos todavia").
INK_MUTED_HEX = "#8A929B"

#: Senal de "pasa" (veredicto Codex OK, calibracion sana).
SIGNAL_PASS_HEX = "#5FBF8F"

#: Senal de "no pasa" (veredicto Codex FAIL, calibracion bloqueada).
SIGNAL_FAIL_HEX = "#E0676B"

#: Senal de aviso (calibracion degradada, medicion bloqueada, rechazo).
SIGNAL_WARN_HEX = "#D9A54B"

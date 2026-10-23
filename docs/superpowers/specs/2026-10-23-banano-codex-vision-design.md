# Sistema de calibración dimensional de banano por visión monocular

**Fecha:** 2026-10-23
**Estado:** Diseño aprobado — pendiente de plan de implementación

---

## 1. Intención y por qué

Construir un sistema de visión por computadora que mida bananos sobre una base fija y
reporte los **dos parámetros que define la norma CODEX STAN 205-1997** para banano de
exportación: longitud a lo largo de la curva exterior y calibre (diámetro de la sección
transversal), más área, volumen y peso estimado.

El proyecto es una **pieza de portafolio destinada a publicarse en LinkedIn**. Eso no es
un detalle decorativo: condiciona decisiones reales a lo largo del documento. Un sistema
que mida bien pero no se pueda enseñar en 30 segundos de video no cumple el objetivo.

Lo que hace que el proyecto se distinga de los cientos de tutoriales de "medir objetos con
ArUco":

1. **Se ancla a una norma real.** No inventa métricas; implementa las del Codex, con sus
   nombres y sus umbrales. El sector agro reconoce esos números.
2. **La norma exige exactamente lo que una bounding box no puede dar.** El Codex define la
   longitud *"a lo largo de la curva exterior"*. Eso obliga a longitud de arco sobre el
   contorno, no a un rectángulo. El proyecto está técnicamente justificado por el dominio.
3. **Reporta su propio error.** Hay tabla de validación contra pesa de cocina y se publica
   el error medio absoluto. Es lo que casi nadie hace y lo que separa ingeniería de demo.
4. **Corrige el paralaje por altura del objeto**, usando el grosor que el propio sistema
   acaba de medir. Es un bucle de autocorrección, no una constante mágica.

### Referencia normativa

CODEX STAN 205-1997 (CXS 205-1997, enmendada 2005 y 2022):

| Parámetro | Mínimo | Definición de la norma |
|---|---|---|
| Longitud | ≥ 14 cm | medida a lo largo de la **curva exterior**, del extremo floral a la base del pedicelo donde termina la pulpa comestible |
| Grado (calibre) | ≥ 2.7 cm (27 mm) | diámetro de la sección transversal, medido en el dedo de referencia |

Tolerancia de calibre admitida por la norma: 10 % en número o peso fuera del rango, siempre
que caigan en la talla inmediatamente superior o inferior.

En la práctica comercial, el calibre de los Cavendish se gradúa de ~34 mm (mínimo comercial)
a 42+ mm (fruta premium), y es la métrica que gobierna el precio por caja.

Fuentes:
- https://www.fao.org/input/download/standards/329/CXS_205e.pdf
- https://alimentosargentinos.magyp.gob.ar/contenido/marco/Codex_Alimentarius/normativa/codex/stan/205-1997.PDF
- https://www.oecd.org/content/dam/oecd/en/publications/reports/2023/07/bananas_de7c66e5/e762dec6-en-fr.pdf

---

## 2. Objetivos

- Medir, sobre banano fresco colocado en la base, en **tiempo real y en vivo**:
  - **Longitud Codex** (mm) — arco sobre la curva exterior
  - **Calibre Codex** (mm) — diámetro en el punto medio del eje
  - **Perfil de grosor** a lo largo del eje
  - **Área proyectada** (mm²)
  - **Volumen estimado** (cm³)
  - **Peso estimado** (g)
  - **Veredicto Codex** (cumple / no cumple) y grado comercial de calibre
- Medir **entre 1 y 3 bananos simultáneos**, separados entre sí, cada uno con identidad
  propia y persistente entre frames.
- Reportar cada medida con su **incertidumbre** (± desviación estándar entre frames), no
  como un número pelado.
- Hacer **visible el proceso de calibración** en la interfaz, con métricas de calidad
  propias en cada etapa.
- Producir una **tabla de validación** contra peso real medido en pesa de cocina, con error
  medio absoluto, error porcentual y R².

## 3. No-objetivos

Fuera de alcance en esta versión. No son malas ideas; son trabajo para después.

- Entrenar un modelo propio. Se usa YOLO preentrenado en COCO (clase `banana`) tal cual.
- Racimos, manos o bananos que se toquen entre sí. Requiere separación de instancias en
  contacto, que COCO no resuelve bien.
- Clasificación de madurez por color (escala Von Loesecke). Es una extensión natural y
  barata, pero se deja para una segunda fase.
- Detección de defectos superficiales (magulladuras, cicatrices).
- Despliegue en producción, multi-cámara, o cinta transportadora.
- Medición de objetos con altura grande respecto a los 30 cm de la cámara (tazas, botellas).
  El modelo de paralaje de este diseño asume objetos tendidos, de grosor pequeño frente a
  la distancia de trabajo.

## 4. Anti-objetivos

Cosas que harían fracasar el proyecto aunque funcionara técnicamente.

- **Reportar cifras sin error asociado.** Si el sistema dice "187.3 mm" sin saber cuánto se
  equivoca, el proyecto pierde su argumento central. Antes no reportar la medida que
  reportarla sin su ±.
- **Falsa precisión.** Mostrar decimales que la calibración no sostiene. La resolución
  efectiva se calcula y se respeta; si el sistema tiene 0.5 mm de error, no se imprimen
  centésimas.
- **Que la interfaz parezca un script.** Ventanas sueltas de `cv2.imshow` con texto encima
  invalidan el objetivo de portafolio.
- **Esconder las limitaciones.** El paralaje, la dependencia del fondo y el límite de los
  racimos se documentan y se dicen en la publicación. Reconocer los límites del sistema
  comunica más seniority que ocultarlos.
- **Que el video quede en segundo plano.** El panel de datos no puede crecer a costa del
  video. Decisión explícita del autor: lo que vende es el video en vivo.

---

## 5. Restricciones del montaje físico

| Elemento | Valor | Notas |
|---|---|---|
| Cámara | **ELP-USBFHD01M-BFV** | OV2710 1/2.7" CMOS; área sensible 5.856 × 3.276 mm; píxel 3.0 µm |
| Resolución de trabajo | 1920 × 1080 @ 30 fps (MJPEG) | también 1280×720@60 y 640×480@120 |
| Lente | **varifocal 2.8–12 mm, zoom y foco manuales** | ver §5.2 — restricción crítica |
| Altura | soporte de **30 cm**; lente → superficie ≈ **27 cm** | el soporte de cámara cuelga de la barra superior y resta ~3 cm. Valor aproximado del autor: **confirmar con metro antes de calibrar** |
| Marcadores | 4 × ArUco impresos en 3D, **24.0 mm el cuadrado negro completo** | coincide con el `markerLength` de OpenCV; espesor por medir |
| Diccionario ArUco | **DICT_6X6_50**, ids **1, 2, 3, 4** | medido en el montaje real con `tools/aruco_probe.py` |
| Tablero de calibración | disponible | se usa una sola vez por cada ajuste de zoom |
| Superficie de trabajo | cartulina **blanca mate**, sobre base rígida | ver §5.3 |
| Objetos | 1 a 3 bananos, separados, tendidos | |
| Instrumentos | báscula de cocina ✓, metro ✓, **pie de rey ✗** | ver §5.4 |
| Plataforma | Windows 11, Python | GPU por determinar |
| Idioma de la interfaz | **inglés** | decisión del autor: alcance en LinkedIn |

### Implicaciones de que los marcadores sean impresos en 3D

Esto no es cosmético. Tres consecuencias con efecto directo en la precisión:

1. **Tienen espesor.** La cara del marcador no está en el plano de la mesa, sino 2–3 mm por
   encima. A 300 mm de distancia de trabajo eso es ~1 % de error de escala sistemático en
   todas las medidas. Se corrige restando el espesor a la altura efectiva. **El espesor
   debe medirse con calibrador y entrar como parámetro de configuración.**
2. **El blanco no es blanco.** Es gris con textura de capas, y eso rompe la decodificación,
   no solo la detección. **Medido sobre las siete capturas reales del montaje:**

   | Parámetros | Resultado |
   |---|---|
   | Defectos puros de OpenCV (`perspectiveRemovePixelPerCell = 4`) | **se cae algún marcador en 5 de 7 capturas**, y no siempre el mismo: `[1,3]`, `[1,2,3]`, `[2,3,4]`, `[1,2]`… |
   | Parámetros ajustados pero con `perspectiveRemovePixelPerCell = 4` | el id 3 no se decodifica en **ninguna** de las capturas sobre superficie pétrea |
   | **`perspectiveRemovePixelPerCell = 8`** | **4/4 en las siete**, sobre cartulina blanca y sobre piedra |

   Es decir: el fallo con los defectos puros es errático, no específico de un marcador — una
   corrección sobre una versión anterior de este documento, que atribuía el fallo solo al
   id 3 generalizando desde un único juego de parámetros. Lo robusto del hallazgo es lo
   contrario de lo que sugería esa lectura: el ajuste recupera los cuatro marcadores incluso
   en el montaje malo que luego se descartó.

   El porqué: al decodificar, ArUco endereza el marcador a un cuadrado canónico y decide
   cada bit muestreando una celda de N×N píxeles. Con N=4 son 16 píxeles por bit, y en una
   pieza FDM una sola línea de capa o una mota puede dominar esa muestra. Con N=8 son 64 y
   la textura se promedia.

   Subir `errorCorrectionRate` a 0.8 también lo resuelve, pero es la palanca equivocada: no
   mejora la muestra, acepta más bits erróneos. Con ~50 cuadriláteros espurios por frame
   procedentes de la veta de la superficie, aflojar la validación invita a falsos positivos.
   **Muestrear mejor, no validar menos.**
3. **El PLA negro refleja.** Una luz puntual cenital produce reflejos especulares que
   destruyen esquinas y hacen parpadear la detección. **Requisito de montaje: iluminación
   difusa o lateral suave, nunca una fuente puntual directamente encima.**
4. **Necesitan zona de silencio.** El detector exige un margen claro alrededor del cuadrado
   negro. La cartulina blanca la provee de forma natural; no pegar los marcadores sobre
   superficie oscura ni al borde de la cartulina.

### 5.1 Presupuesto óptico

Con el sensor de 5.856 mm de ancho, el campo de visión horizontal a una distancia de
trabajo `H` es `W = 5.856 · H / f`. A `H = 270 mm`:

| Focal | Ancho capturado | Alto capturado | Escala |
|---|---|---|---|
| 2.8 mm | 565 mm | 316 mm | 0.294 mm/px |
| **4.0 mm** | **395 mm** | **221 mm** | **0.206 mm/px** |
| 5.0 mm | 316 mm | 177 mm | 0.165 mm/px |
| 6.0 mm | 264 mm | 148 mm | 0.137 mm/px |
| 12 mm | 132 mm | 74 mm | 0.069 mm/px |

Un banano mide 180–230 mm y hay que rodearlo con cuatro marcadores más margen. Eso pide un
área útil del orden de **300 × 180 mm**, que cae en **f ≈ 4 mm**. Consecuencias:

- **Escala de trabajo ≈ 0.2 mm/px.** Confirma el lienzo rectificado de 0.2 mm/px elegido en
  §7; el warp no inventa ni tira resolución.
- **Un ArUco de 24 mm ocupa ~116 px.** Muy por encima del mínimo de detección fiable
  (~30 px), con margen de sobra para refinamiento subpíxel de esquinas.
- **Objetivo realista de exactitud:** ±1 mm en longitud, ±0.5 mm en calibre.
- A 2.8 mm el campo sobra pero la distorsión de barril es severa y se desperdicia
  resolución. A 6 mm o más no cabe el área de trabajo. **4 mm es el punto de operación.**

Nota sobre MJPEG: a 1080p sobre USB 2.0 la única opción a 30 fps es MJPEG, que introduce
artefactos de compresión en los bordes. A 0.2 mm/px, una incertidumbre de borde de 1 px son
0.2 mm — tolerable. YUY2 sin comprimir no alcanza el ancho de banda a esta resolución.

### 5.1.b Valores medidos en el montaje real (2026-10-23)

Tomados con `tools/aruco_probe.py` sobre tres capturas del montaje:

| Magnitud | Medido | Comentario |
|---|---|---|
| Resolución concedida | 1920 × 1080 | el driver concede lo pedido |
| Lado del marcador | 119.3–122.7 px | predicho ~116 px en §5.1 |
| Escala | 0.196–0.199 mm/px | **confirma el lienzo de 0.2 mm/px** |
| Campo de visión | ~380 × 213 mm | dentro del objetivo; el zoom ya está bien puesto |
| Paralelismo cámara–plano | ratio de tamaños 0.974 | inclinación ~2°, despreciable |
| Repetibilidad de esquinas | 0.05–0.34 mm (σ entre frames) | muy por debajo del objetivo de ±1 mm |
| Falsos candidatos | ~50 por frame | veta de la superficie pétrea; desaparecen con cartulina mate |

Dos problemas abiertos del montaje:

- **Área entre marcadores: 184 × 122 mm.** Un banano mide 180–230 mm, así que **no cabe**.
  Hay sitio de sobra en el cuadro (380 × 213 mm): los marcadores están demasiado juntos. El
  límite real son las patas del soporte, que dejan ~233 mm libres en horizontal y los 213 mm
  completos en vertical. Con los marcadores abiertos a ese rectángulo, un banano largo entra
  **en diagonal** — la homografía y el eje medial son indiferentes a la orientación.
- **Reflejo especular** visible en el centro inferior del cuadro: superficie brillante más
  fuente de luz puntual. Lo resuelve la cartulina mate de §5.3 más luz difusa.

### 5.1.c Validación empírica de la tesis del proyecto (2026-10-23)

Segunda tanda de capturas, ya con cartulina blanca mate y un banano real en escena.
Marcadores abiertos a un rectángulo de 215 × 118 mm; 4/4 detectados en las cuatro capturas;
campo de visión 392 × 221 mm, prácticamente idéntico al objetivo de §5.1; cuadrilátero de
marcadores simétrico al 0.8 %, o sea cámara bien alineada; cero píxeles quemados (antes
0.3 %), el reflejo especular desapareció con la cartulina mate; candidatos espurios de ~50
a ~35. El banano cabe entero dentro del cuadrilátero calibrado.

**La segmentación por saturación funciona.** De 913 372 píxeles oscuros pero desaturados de
la escena (la sombra del banano y las patas negras del soporte), **57** acabaron dentro de
la máscara del banano: el 0.006 %. La decisión de §5.3 queda confirmada sobre datos reales.

**La tesis del proyecto, medida sobre el banano real:**

| Medida | Valor |
|---|---|
| Cuerda punta a punta (lo que da una bounding box) | 182.3 mm |
| Curva interior (cóncava) | 217.0 mm |
| **Curva exterior — longitud Codex** | **233.5 mm** |

**La norma mide 51 mm más que la cuerda: un 28 %.** Mayor incluso de lo previsto. Una caja
delimitadora no se queda corta por un detalle; se equivoca en casi un tercio.

**El escalonado de píxeles también confirmado:** la misma curva exterior da 253.2 mm sobre
la cadena cruda de píxeles y 233.5 mm tras suavizar — **8.4 % de sobrestimación**, 20 mm en
este banano. El paso de suavizado de §7 etapa 6 no es un refinamiento opcional.

**Hallazgo nuevo — dónde se mide el calibre importa mucho más de lo que parecía.** Sobre la
misma máscara, el círculo inscrito máximo da 51.9 mm y el ancho medio (área dividida por
longitud de la línea media) da 38.8 mm. Un 34 % de diferencia según el criterio. La norma
fija el punto — el medio del fruto — y por eso el diseño toma el calibre en el punto medio
del eje medial **ya recortado el pedicelo**: no es un detalle de implementación, es lo que
decide el número.

**Advertencia sobre estos valores absolutos.** Se midieron sin corregir distorsión, sin
homografía y sin corrección de paralaje, usando la escala derivada de los marcadores. Los
marcadores están a radio ~598 px del centro óptico y el banano a ~0–300 px: zonas distintas
del lente. Las **comparaciones relativas** (cuerda contra arco, crudo contra suavizado) son
sólidas porque ambos términos se midieron igual sobre la misma imagen; los **valores
absolutos son provisionales** hasta que exista la cadena de corrección completa. Es, de
paso, la demostración empírica de por qué la calibración intrínseca de §7 etapa 1 no se
puede saltar.

### 5.1.d Parpadeo de la iluminación: estetico, no metrologico (2026-10-23)

La lámpara LED cenital produce bandas horizontales visibles. La causa no es la cartulina:
el OV2710 es de **obturador rodante** y expone fila por fila, mientras el LED se apaga y
enciende cientos de veces por segundo (regulación PWM, o el rizado de 120 Hz de la red).
Cada fila pilla el LED en un punto distinto del ciclo.

**Medido: no afecta a la medición.**

| | Valor |
|---|---|
| Oscilación por bandas en saturación | 1.2–1.8 unidades (de 255) |
| Margen de la cartulina al umbral de segmentación | ~55 unidades |
| Área del banano en cuatro capturas con bandas visibles | 208 287 / 208 441 / 208 539 / 208 312 px |
| Dispersión de esa área | **0.12 %** |

El parpadeo es entre 30 y 45 veces menor que el margen de decisión, y el área segmentada
es estable al 0.12 % pese a las bandas. **La lámpara se queda.** Corregir el parpadeo es
cosmetica para el video, no un requisito de exactitud: panel del driver (`Powerline
Frequency` / anti-flicker a 60 Hz), subir el tiempo de exposición, o rebotar la luz contra
el techo en vez de iluminar directo.

**Limite anotado:** las manchas oscuras del banano bajan a saturación 45–61, por debajo del
umbral de 83. El cierre morfológico y la seleccion del componente mayor las recuperan en un
banano normal, pero un banano muy maduro, con mucha superficie parda, puede perder area.
Hay que validarlo con fruta en distintos estados de madurez.

### 5.2 Restricción crítica: bloquear el varifocal

La lente tiene dos anillos ajustables (zoom y foco) con tornillos prisioneros. **Los
parámetros intrínsecos solo son válidos para la posición exacta de zoom y foco en que se
calibró.** Si alguno de los dos anillos se mueve — incluso sin querer, al manipular el
montaje — la matriz de cámara y los coeficientes de distorsión dejan de corresponder y
todas las medidas se corrompen de forma silenciosa.

Procedimiento obligatorio:

1. Ajustar zoom a ~4 mm y enfocar nítidamente el plano de trabajo.
2. **Apretar los tornillos prisioneros de ambos anillos.**
3. Marcar la posición de los anillos con un trazo de pintura o cinta, para detectar a simple
   vista si se movieron.
4. Solo entonces ejecutar la calibración intrínseca.
5. Guardar en `config/intrinsics.json` la posición de zoom junto a los parámetros.

Si se cambia el zoom, hay que recalibrar. No hay atajo.

### 5.3 Requisito de superficie rígida y plana

Toda la metrología descansa en que el área de trabajo sea un **plano**. Una cartulina
apoyada sobre alfombra, tapete o cualquier superficie blanda no es un plano: se hunde bajo
el peso del banano y se ondula. El banano deja de estar a altura constante y la corrección
de paralaje deja de tener sentido.

**Requisito:** la cartulina va sobre una base rígida y plana — MDF, acrílico, vidrio, o
simplemente trasladar todo el montaje a una mesa. El soporte de la cámara debe apoyarse en
la misma superficie rígida, no en una blanda.

Sobre el blanco como fondo: funciona, con dos condiciones. Debe ser **mate** (el brillo
produce reflejos especulares que la segmentación confunde con el objeto), y la segmentación
debe hacerse en el **canal de saturación** de HSV, no en luminancia. La razón es la sombra:
el banano proyecta sombra sobre el blanco, y una sombra es gris — oscura en luminancia pero
igual de desaturada que el blanco. Segmentando por saturación, el amarillo del banano salta
y la sombra se queda con el fondo.

Aun así, una lámina mate azul o verde separaría también en tono y daría más margen. Cuesta
poco probarlo y conviene comparar ambas antes de fijar el método.

### 5.4 Medir el espesor de los marcadores sin pie de rey

El espesor entra en la altura efectiva y vale ~1 % de error sistemático si se ignora. Sin
calibrador hay dos caminos válidos:

- **Apilar los cuatro marcadores** y medir la pila con el metro, dividiendo entre 4. Un metro
  con resolución de 1 mm da así ~0.25 mm de precisión, suficiente.
- **Consultar el laminado del STL o el laminador** con que se imprimieron: la altura de la
  pieza es un parámetro conocido del modelo.

---

## 6. Arquitectura: la decisión central

**Detección neuronal, medición clásica.** Las dos responsabilidades se separan y no se
mezclan:

| Etapa | Herramienta | Responde |
|---|---|---|
| Detectar e identificar | YOLOv8-seg preentrenado en COCO + tracker | *qué* es y *dónde* está |
| Medir | visión clásica dentro del ROI | el contorno exacto y sus dimensiones |

**El porqué:** las máscaras de segmentación de COCO son polígonos anotados a mano con
criterio de velocidad, no contornos de precisión. Medir directamente sobre la máscara de
YOLO significa que la exactitud del sistema la define un anotador humano trabajando rápido.
YOLO aporta lo que la visión clásica no tiene — saber que *eso* es un banano, y mantener su
identidad — y la visión clásica aporta lo que la red no da: un borde preciso y repetible.

Esta separación es la decisión de diseño más defendible del proyecto y debe explicarse en
la publicación.

---

## 7. Pipeline de procesamiento

### Etapa 0 — Identificación del diccionario ArUco (una sola vez)

Script independiente que recorre todos los diccionarios predefinidos de OpenCV
(`DICT_4X4_50`, `DICT_5X5_*`, `DICT_6X6_*`, `DICT_7X7_*`, `DICT_ARUCO_ORIGINAL`, …) sobre
una foto de los cuatro marcadores, y reporta cuál produce 4 detecciones consistentes y qué
IDs devuelve. Resultado: diccionario e IDs fijos, guardados en configuración.

Es el primer entregable del proyecto y desbloquea todo lo demás.

### Etapa 1 — Calibración intrínseca (una sola vez, persistente)

~20 capturas del tablero en poses variadas (inclinado, cerca, lejos, en las cuatro esquinas
del cuadro). `cv2.calibrateCamera` → `camera_matrix`, `dist_coeffs`, error RMS de
reproyección.

Se guarda en `config/intrinsics.json`. No se vuelve a ejecutar mientras no se cambie de
lente ni de resolución.

**Criterio de aceptación:** RMS < 0.5 px. Por encima de eso, repetir la captura.

**Por qué no se puede saltar:** la lente gran angular del ELP introduce distorsión de
barril. Sin corregirla, la homografía intenta compensar con un modelo lineal algo que no lo
es, y el error crece hacia los bordes del cuadro. El síntoma es desconcertante: el mismo
banano mide distinto según dónde lo pongas, y el error no se puede atribuir a nada del
algoritmo.

En la interfaz esta etapa aparece como estado cargado (`✓ Intrínseca · RMS 0.21 px`) con un
**toggle antes/después de la corrección de distorsión**. Ese toggle, con líneas rectas
enderezándose, es el recurso visual más eficaz del proyecto y debe existir aunque solo se
use en el video.

### Etapa 2 — Calibración del plano métrico (**continua, cada frame**)

1. Deshacer la distorsión del frame con los intrínsecos.
2. Detectar los 4 ArUcos. Cada marcador aporta 4 esquinas → **16 correspondencias** entre
   píxeles y coordenadas del mundo.
3. Las posiciones de los marcadores en el mundo provienen de una **plantilla de disposición
   conocida** (ver §9). Con 16 puntos para 8 incógnitas, el sistema queda sobredeterminado.
4. `cv2.findHomography` → matriz H que mapea el plano de la base al plano de la imagen.
5. **Residual de reproyección:** reproyectar los 16 puntos del mundo a la imagen y medir el
   error en mm. Esta es la métrica de confianza de la calibración y va en pantalla.
6. Corrección por espesor del marcador: la altura efectiva de trabajo es `H_ef = H − t`,
   donde `H` es la distancia medida de la lente a la cartulina y `t` el espesor de los
   marcadores.

**La homografía se recalcula en cada frame, no una vez por sesión.** Los cuatro marcadores
están permanentemente en el campo de visión, así que no cuesta nada, y convierte la
debilidad más obvia del montaje en una fortaleza: si alguien roza el soporte, si la mesa
vibra, si el trípode cede un poco, el sistema **se recalibra solo en el siguiente frame**.
Un sistema calibrado una sola vez al arrancar quedaría corrompido en silencio por el primer
golpe.

Lo único que *no* se auto-corrige es el cambio de zoom o foco (§5.2): eso invalida los
intrínsecos, no la homografía.

**Degradación elegante:** con 3 marcadores visibles el sistema sigue funcionando con 12
puntos y lo indica (`⚠ 3/4 ArUcos`). Con menos de 3, la medición se bloquea y se avisa; no
se reportan medidas sin calibración válida.

### Etapa 3 — Rectificación

`cv2.warpPerspective` a un lienzo cenital con escala fija y conocida (p. ej. 0.2 mm/px). En
ese lienzo **1 píxel vale un número conocido de milímetros en toda la superficie**, que es
toda la razón de ser de la homografía: sin ella, un objeto en la esquina mediría distinto
que el mismo objeto en el centro.

Toda la medición ocurre en este lienzo rectificado, nunca en la imagen cruda.

### Etapa 4 — Detección y seguimiento

- `YOLOv8n-seg` con `model.track(persist=True)`, filtrando a la clase COCO `banana` (46).
- El tracker devuelve un `.id` estable por instancia entre frames.
- **Respaldo de asociación:** los objetos están quietos, así que si el tracker pierde un ID
  por una caída momentánea de confianza, se reasocia por cercanía de centroide en lugar de
  reiniciar el acumulador.
- **Guarda de validez:** si dos máscaras quedan adyacentes o fusionadas, no se reporta
  medida — se muestra el aviso "separa los bananos". Una medida falsa es peor que ninguna.

### Etapa 5 — Refinamiento del contorno

Dentro del ROI que da YOLO, sobre el lienzo rectificado:

- Segmentación clásica por **umbral sobre el canal de saturación de HSV** (ver §5.3). Sobre
  cartulina blanca es la elección correcta: el amarillo del banano está muy saturado, y
  tanto el blanco como su sombra están desaturados, así que la sombra no entra en la
  máscara. Un umbral sobre luminancia, en cambio, se tragaría la sombra entera.
- Alternativas a comparar empíricamente: canal `b` de LAB, y GrabCut sembrado con la máscara
  de YOLO.
- Limpieza morfológica (apertura y cierre) y selección del componente conexo mayor.

El resultado es una máscara binaria de borde limpio y repetible, que es lo que realmente se
mide.

### Etapa 6 — Medición geométrica

Aquí vive el núcleo del proyecto.

1. **Eje medial.** `skimage.morphology.medial_axis(mask, return_distance=True)` devuelve en
   una sola llamada el esqueleto y la transformada de distancia. El valor de distancia en
   cada punto del esqueleto es el **radio del círculo inscrito**, es decir la mitad del
   ancho local. El perfil de grosor sale gratis.
2. **Poda y camino principal.** Los esqueletos de blobs tienen ramas espurias en los
   extremos. Se construye el grafo de píxeles del esqueleto y se toma el **camino geodésico
   más largo entre dos extremos**; sus dos puntas son las puntas del banano.
3. **Curva exterior (longitud Codex).** El eje medial **no** es la curva exterior: en un
   objeto curvo el borde convexo es más largo que la línea media. Procedimiento:
   - partir el contorno en los dos puntos más cercanos a las puntas → dos cadenas
   - la cadena **más larga** es la curva exterior
   - su longitud de arco es la **longitud Codex**
4. **Corrección de escalonado.** La longitud de una cadena de píxeles 8-conexa sobrestima
   la de una curva suave por el efecto escalera. Se ajusta un spline suavizante a la cadena
   y se integra, o se simplifica con Douglas-Peucker antes de `cv2.arcLength`. **Sin esta
   corrección el sistema sobrestima sistemáticamente la longitud en varios por ciento.**
5. **Recorte del pedicelo.** La norma mide *"del extremo floral a la base del pedicelo
   donde termina la pulpa comestible"* — es decir, **el cabo no cuenta**. Pero el cabo sí
   aparece en la máscara. Si no se recorta, el sistema sobrestima la longitud en todo lo
   que mida el pedicelo.
   El perfil de grosor resuelve esto: el cabo es marcadamente más delgado que el fruto. Se
   recorre el eje medial desde cada punta hacia el centro y se corta donde `w(s)` cae por
   debajo de una fracción del ancho máximo. El umbral se ajusta empíricamente y debe ser
   asimétrico — el extremo floral también se adelgaza, pero de forma gradual, mientras que
   la transición al pedicelo es abrupta. Se recomienda detectar el corte por la **derivada**
   de `w(s)`, no por su valor absoluto.
   Es una aproximación y debe documentarse como tal.
6. **Calibre Codex.** Dos veces el valor de la transformada de distancia en el punto medio
   del eje medial, calculado **después** del recorte del pedicelo.
7. **Perfil de grosor** `w(s)` a lo largo de la abscisa curvilínea del eje.
8. **Área proyectada:** número de píxeles de la máscara × (mm/px)².
9. **Volumen**, asumiendo secciones transversales circulares:
   `V ≈ Σ π·(w(s)/2)²·Δs`

### Etapa 7 — Corrección de paralaje

El problema: la cámara está a ~270 mm y un banano tendido tiene ~35 mm de grosor, un 13 %
de la distancia de trabajo. La silueta vista desde arriba corresponde aproximadamente
a la sección más ancha, que está a media altura del objeto, no en la mesa. Si se mide
asumiendo que el objeto está en el plano de calibración, **se sobrestima en torno al 6.5 %**
— unos 12 mm en un banano de 190 mm. No es un detalle fino: es el error más grande del
sistema si no se corrige.

La corrección y lo elegante del asunto: la altura que hace falta es el grosor, que el propio
sistema acaba de medir.

```
factor = (H_ef − h/2) / H_ef        con h = calibre medido
```

Se itera: medir con plano cero → obtener `h` → corregir escala → volver a medir. Converge en
dos pasadas. En la interfaz conviene poder apagar la corrección para enseñar la diferencia.

### Etapa 8 — Agregación temporal

La segmentación tiembla entre frames: el borde se mueve uno o dos píxeles y la medida salta.
Con identidad persistente por `track_id` eso se resuelve:

- Buffer circular de las últimas N ≈ 30 mediciones por ID.
- Se reporta la **mediana** (robusta a frames malos, a diferencia de la media).
- La **desviación estándar del buffer es la incertidumbre de medición reportada.**
- La medida se "fija" cuando el buffer está lleno y la desviación cae por debajo de un
  umbral. Mientras tanto, barra de progreso (`midiendo 19/30`).

El ruido cae aproximadamente con √N, y de paso el sistema gana una métrica de metrología
real sin coste adicional.

### Etapa 9 — Evaluación Codex

- Cumple si `longitud ≥ 140 mm` **y** `calibre ≥ 27 mm`.
- Además se reporta el grado comercial de calibre (34–42+).

### Etapa 10 — Validación

Registro en CSV: identificador de muestra, todas las medidas calculadas, peso real de la
pesa.

**La densidad no se asume: se ajusta.** En lugar de tomar una densidad de tabla, se hace
regresión lineal de peso real contra volumen calculado sobre el conjunto de validación; la
pendiente es la densidad efectiva del sistema. Es más honesto y más preciso, y el R² de esa
regresión es un resultado publicable.

Métricas finales: error medio absoluto (g), error porcentual medio, R².

**Diseño experimental — separar dos preguntas distintas:**

1. *¿Es correcta la metrología?* Se valida con objetos planos de dimensión normalizada
   conocida: tarjeta ISO/IEC 7810 ID-1 (85.60 × 53.98 mm), un ArUco dejado fuera de la
   calibración (debe medir 24.0 mm), monedas. Al ser planos, aíslan el error de calibración
   del error de paralaje.
2. *¿Es correcta la segmentación del banano?* Se valida con la pesa.

Mezclar ambas impide saber de dónde viene el error.

---

## 8. Interfaz

Decisión del autor: **el video es lo que vende**. El reparto del espacio se subordina a eso.

Layout elegido — *variante 1, cenital dominante con PiP*:

```
┌──────────────────────────────────────────────────────┬──────────┐
│ ✓ Intrínseca RMS 0.21px │ ✓ 4/4 ArUcos │ 0.183 mm/px │          │  ← barra fina
├──────────────────────────────────────────────────────┤          │
│                                                      │ #1  ✓    │
│          VISTA CENITAL RECTIFICADA                   │ L 187.3  │
│          a escala real                               │ ⌀ 38.4   │
│                                                      │ 124 g    │
│     cotas dibujadas SOBRE el video,                  │          │
│     siguiendo a cada banano                          │ #2  ✓    │
│                                       ┌────────────┐ │ ...      │
│                                       │ PiP crudo  │ │          │
│                                       │ 4 ArUcos   │ │ #3       │
│                                       └────────────┘ │ ▓▓▓░ 19/30│
└──────────────────────────────────────────────────────┴──────────┘
                                                        240 px fijos
```

Requisitos de la interfaz:

- **Las cotas se dibujan sobre el frame**, no solo en el panel. Los números siguen a los
  bananos cuando se mueven con la mano. Si los números viven únicamente en la tabla, el
  video pasa a ser decorativo y la mirada se va al panel — justo lo contrario del objetivo.
- **El panel lateral mide 240 px fijos, no un porcentaje.** Al maximizar la ventana crece el
  video y el panel no.
- El **PiP se oculta con una tecla** para primeros planos limpios.
- El PiP se renderiza a resolución reducida: ya se paga un `warpPerspective` por frame y no
  conviene duplicar el coste de render.
- Cada etapa de calibración expone su métrica de calidad en la barra superior.
- La pantalla de calibración (intrínseca y de plano) es una vista aparte, no compite por
  espacio con la operación en vivo.
- Vista de validación separada, con la tabla y las métricas de error.
- **Todos los textos de la interfaz van en inglés**, por decisión del autor (alcance en
  LinkedIn). Este documento y los comentarios internos quedan en español.

### Pila tecnológica

| Capa | Elección | Razón |
|---|---|---|
| Lenguaje | Python 3.11 | |
| Visión | OpenCV (`opencv-python`) | ArUco, homografía, warp |
| Geometría | scikit-image | `medial_axis` con distancias en una llamada |
| Detección | Ultralytics YOLOv8n-seg | preentrenado COCO + tracking integrado |
| Numérico | NumPy, SciPy | splines, regresión |
| Interfaz | **PySide6 (Qt)** | video en vivo sin tirones; aspecto de software real |
| Datos | pandas | tabla de validación y CSV |

**Sobre la interfaz:** Streamlit es más rápido de escribir, pero el video en vivo va a
tirones y eso se nota muchísimo en una grabación. Dado que el objetivo del proyecto es que
el video se vea bien, Streamlit queda descartado. Ventanas de `cv2.imshow` quedan
descartadas por el anti-objetivo de aspecto amateur.

**Modelo de hilos:** hilo de captura → hilo de procesamiento → señales de Qt hacia la
interfaz. Nunca procesar en el hilo de la GUI; es la causa número uno de interfaces que se
congelan con video.

**Sobre la GPU:** aún sin determinar si el equipo tiene NVIDIA. El diseño no depende de
ello. YOLOv8n-seg es el modelo más pequeño de la familia y corre en CPU, aunque a pocos
fps. Estrategias de respaldo, en orden:

1. Ejecutar la inferencia sobre el frame **reducido a 640 px** y escalar la máscara al
   lienzo rectificado. La detección no necesita resolución completa — la medición sí, pero
   esa la hace la visión clásica a resolución plena.
2. Ejecutar YOLO **cada N frames** en vez de en todos. Los objetos están quietos; entre
   inferencias se reutiliza el ROI anterior y solo se repite la medición clásica, que es
   barata.
3. Exportar el modelo a ONNX Runtime u OpenVINO para acelerar en CPU.

Con (1) y (2) el sistema mantiene video fluido incluso sin GPU, porque lo que el usuario ve
moverse a 30 fps es el warp y el overlay, no la inferencia.

---

## 9. Disposición de los marcadores

Los cuatro ArUcos deben estar en **posiciones conocidas** para aportar las 16
correspondencias. Dos caminos:

- **Plantilla impresa (preferido):** una hoja con las cuatro posiciones marcadas, sobre la
  que se pegan los marcadores impresos. Las coordenadas del mundo son exactas por
  construcción y el residual de reproyección mide directamente la calidad del pegado.
- **Auto-levantamiento (respaldo):** si no se pueden colocar con precisión, se usa un
  marcador para fijar la escala (sus propias 4 esquinas a 24 mm), se miden las posiciones de
  los otros tres y se guardan como disposición calibrada. Más cómodo, peor condicionado:
  extrapolar escala desde un cuadrado de 24 mm hacia toda la superficie amplifica el error
  de esquina.

La disposición concreta depende de las dimensiones de la base y del campo de visión real de
la cámara a 300 mm.

---

## 10. Decisiones ya tomadas

**Del autor:**
- El objeto es banano, por relevancia del sector agro.
- YOLO preentrenado, sin entrenamiento propio.
- De 1 a 3 bananos, siempre separados; nada de racimos.
- Seguimiento con identidad persistente por banano.
- Layout: variante 1 (cenital dominante + PiP), con el video como protagonista.

**Del diseño (decisiones mías, sujetas a revisión del autor):**
- PySide6 como interfaz.
- Separación detección neuronal / medición clásica.
- Eje medial para las puntas y el calibre; partición del contorno para la longitud Codex.
- Corrección iterativa de paralaje con el grosor medido.
- Mediana temporal por `track_id`, con la desviación estándar como incertidumbre.
- Densidad ajustada por regresión, no tomada de tabla.
- Uso del tablero para calibración intrínseca única y persistente, contra la intención
  inicial de omitirlo. El motivo está en §7, etapa 1.
- Lienzo rectificado a 0.2 mm/px y ventana temporal N = 30; ambos parametrizables.

---

## 11. Qué queda a criterio de quien construya

- Estructura de módulos y organización del repositorio.
- Formato concreto de los ficheros de configuración y persistencia.
- Detalles de estilo visual de la interfaz dentro de los requisitos de §8.
- Estrategia de poda del esqueleto (varias implementaciones válidas del camino geodésico
  más largo).
- Elección entre GrabCut y umbralado por canal para el refinamiento del contorno; se decide
  empíricamente según el fondo real.
- Política de reintentos y manejo de desconexión de la cámara.
- Si la validación se automatiza con un flujo de captura guiado o se hace con registro
  manual.

---

## 12. Riesgos conocidos

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Reflejos especulares en el PLA negro | detección de ArUcos intermitente | iluminación difusa; ajuste de `DetectorParameters` |
| Fondo con contraste insuficiente | contorno impreciso → todas las medidas mal | superficie mate azul o verde; validar temprano |
| Espesor de marcador no medido | ~1 % de error sistemático | medir con calibrador y parametrizar |
| Escalonado de píxeles | sobrestimación sistemática de longitud | spline suavizante antes de integrar |
| Umbral de recorte del pedicelo mal ajustado | longitud sobrestimada (no recorta) o subestimada (recorta fruto) | ajuste empírico por derivada del perfil de grosor; validar contra medida manual |
| **Anillo de zoom o foco movido tras calibrar** | **todas las medidas corruptas en silencio** | **apretar prisioneros + marca de pintura; verificar antes de cada sesión** |
| Superficie de trabajo blanda (alfombra) | el plano deja de ser plano | base rígida obligatoria bajo la cartulina |
| Cartulina blanca brillante | reflejos confundidos con objeto | exigir acabado mate |
| Sombra del banano sobre fondo blanco | máscara inflada | segmentar por saturación, no por luminancia |
| Campo de visión insuficiente | no cabe el área de trabajo | f ≈ 4 mm según §5.1; verificar en montaje real |
| Sin GPU NVIDIA | inferencia lenta | inferencia a 640 px y cada N frames (§8) |
| Pérdida de ID del tracker | reinicio del acumulador temporal | reasociación por centroide |
| COCO falla con bananos muy verdes o muy maduros | sin detección | verificar temprano; respaldo por color si hace falta |

---

## 13. Qué se publica

La publicación debería contener, por orden de impacto:

1. El video en vivo: mover bananos con la mano y ver los números seguirlos.
2. El toggle antes/después de corregir la distorsión.
3. La frase que ancla el proyecto: *la norma Codex exige medir el banano a lo largo de su
   curva exterior; una bounding box no puede hacer eso.*
4. La tabla de validación con el error real.
5. Las limitaciones, dichas abiertamente.

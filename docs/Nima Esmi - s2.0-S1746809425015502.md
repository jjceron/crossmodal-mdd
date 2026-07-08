# Análisis técnico: "Multimodal transformer for depression detection based on EEG and interview data" (Esmi et al., 2026, BSPC)

---

## 1. Resumen conceptual del método

**Hipótesis principal.** Los autores parten de una idea bastante estándar en fusión multimodal — que EEG (marcador neurológico objetivo) y entrevista (marcador conductual subjetivo, dividido en audio paralingüístico y texto lingüístico) contienen información *complementaria* sobre depresión — pero le añaden dos hipótesis secundarias que son el verdadero motor del paper:

1. Que la relación entre canales EEG y entre EEG-audio-texto **no es estática**, sino que existe un desfase temporal (asincronía) entre la señal EEG y la señal de audio/texto de la entrevista, y que modelar explícitamente esa alineación temporal (en vez de asumir sincronía perfecta) mejora la fusión.
2. Que **no todos los canales EEG aportan la misma información discriminativa**, y que se puede aprender una priorización de canales de forma agnóstica al modelo de clasificación (es decir, antes y fuera de la red neuronal), para luego usar solo un subconjunto pequeño sin sacrificar demasiada exactitud.

**Problema que resuelve.** Dos problemas prácticos concretos, no solo "mejorar accuracy":
- **Sincronización intermodal:** en escenarios reales, EEG e interview audio pueden estar desalineados temporalmente (drift de reloj, latencias de grabación, diferencias de muestreo). Los métodos anteriores basados en GNN (MS2-GNN, G-Atten, etc.) asumen implícitamente una correspondencia temporal razonable y se degradan cuando esta falta.
- **Costo/portabilidad de hardware EEG:** los sistemas de 128+ canales son caros y poco portables. El paper aborda esto con FTSM (Flexible Temporal Sequence Matching) como método de *priorización* de canales.

**Contribución real.** Hay que ser preciso aquí porque el paper mezcla términos que suenan más novedosos de lo que realmente son:

- La arquitectura central es, en esencia, un **transformer multimodal con cross-attention + self-attention**, un patrón muy establecido (ViLBERT, LXMERT, Perceiver, etc.) aplicado a EEG-como-imagen + audio-espectrograma + texto tokenizado. No hay una innovación arquitectónica de fondo en el mecanismo de atención en sí (Ecs. 4-6 son el transformer estándar de Vaswani et al.).
- La contribución real y más defendible es doble:
  1. **FTSM para selección/priorización de canales EEG**, que en realidad es una reformulación de *shapeDTW* / DTW clásico (Algoritmo 1 es DTW puro, la referencia [51] es shapeDTW de Zhao & Itti 2018) aplicado como medida de similitud entre canales para rankear su redundancia/relevancia.
  2. **Módulo de sincronización de modalidades** (Algoritmo 2), inspirado explícitamente en trabajos de sincronización audio-visual ([58], "Unified cross-modal attention... audio-visual speech recognition"), adaptado aquí a EEG-audio. Es un mecanismo de *self-supervised offset prediction*: se simulan desplazamientos temporales, se calcula similitud coseno entre embeddings EEG y audio para cada offset candidato, y se entrena con un loss de entropía cruzada (con label smoothing) para que el offset correcto sea el más probable.

**Diferencia frente a una simple concatenación de features.** El argumento del paper es que concatenar features (early fusion naive) o promediar logits (late fusion naive) no captura *interacciones* entre modalidades — por ejemplo, que un patrón espectral EEG específico correlacione con un patrón prosódico de audio específico. El cross-attention, al generar Q desde una modalidad y K/V desde otra, permite que cada token de una modalidad "busque" activamente en la otra modalidad la información relevante, aprendiendo pesos de atención en vez de una combinación lineal fija. Es una fusión *aprendida y dependiente del contenido*, no una fusión estática. Además, el módulo de sincronización resuelve algo que la concatenación no puede resolver en absoluto: el desalineamiento temporal entre secuencias de longitud/tasa de muestreo distintas.

---

## 2. Dataset y protocolo experimental

**Datasets:**
- **MODMA** (Lanzhou University): 128 canales EEG + audio, 24 pacientes con MDD + 29 controles sanos = 53 sujetos. 20 mujeres, 33 hombres, edades 16-52. Sesiones de 20-30 min con elicitación multimodal de emociones.
- **DAIC-WOZ**: 189 sesiones (142 train / 47 test), entrevistador virtual (Wizard-of-Oz), audio 16kHz/16-bit, video 1920×1080, texto transcrito. Los autores **solo usan audio + transcripts** de DAIC-WOZ (no video ni EEG, obviamente, porque DAIC-WOZ no tiene EEG).

**Distribución de clases:** Para MODMA: 24 MDD vs 29 sanos — desbalance leve (~45%/55%), manejable. Para DAIC-WOZ el split oficial train/test ya viene definido por el corpus (142/47), con desbalance conocido en ese dataset hacia la clase no-depresiva.

**División train/val/test — punto crítico.** El paper **no especifica explícitamente** si el split en MODMA es por sujeto o por ventana/segmento. Esto es una omisión metodológica seria, porque:
- Con solo 53 sujetos y sesiones de 20-30 min, si se generan múltiples "imágenes 2D" de 224×224 por sesión (segmentando la señal en ventanas de tiempo), y luego se hace el split train/val a nivel de ventana en lugar de a nivel de sujeto, **hay fuga de datos (data leakage) por diseño**: ventanas del mismo sujeto (con la misma "firma" neurofisiológica/de voz) aparecerían tanto en train como en validación/test, inflando artificialmente la accuracy.
- La Fig. 6 muestra curvas de "Training" y "Validation" accuracy que difieren claramente (mencionan overfitting explícitamente: "hay un gap significativo entre training y validation... indicando overfitting"), lo cual sugiere que si hubiera leakage severo a nivel de ventana, el gap sería mucho menor de lo reportado — es un indicio *en contra* de leakage total, pero no lo descarta, porque el leakage parcial (algunas ventanas del mismo sujeto en train y val) igual podría inflar el resultado sin eliminar todo el gap.
- El hecho de que reporten resultados con desviación estándar sobre 10 corridas (91.20% ± 0.75%) sugiere variabilidad controlada, pero eso no dice nada sobre si el split es correcto a nivel de sujeto.

**Veredicto sobre leakage:** con 53 sujetos en MODMA, es *altamente probable* que exista algún grado de leakage por ventana, dado que no se menciona un split "subject-independent" explícito ni k-fold subject-wise. Esto es exactamente el tipo de riesgo que debes evitar en tu propio trabajo con MODMA.

**Otras decisiones experimentales relevantes:**
- Usan grid search para learning rate, batch size y épocas (Fig. 8), reportando óptimos en LR=1e-4, batch=32, épocas≈38.
- Hardware de preprocesamiento: CPU/GPU de gama media (i7, GTX 1080), entrenamiento final en Google Colab — sugiere un modelo no gigantesco en cómputo, pero con ViT de por medio (backbone pesado) para pocos datos.
- Métricas: Accuracy, Precision, Recall, F1 — estándar, correcto para problema binario.
- Test estadístico: t-tests pareados sobre 10 corridas contra cada baseline — buena práctica, aunque no reportan corrección por comparaciones múltiples (Bonferroni/Holm), lo cual sería exigible en revisión.

---

## 3. Preprocesamiento de señales

### EEG

- **Canales:** parten de 128 canales (dataset MODMA nativo), y evalúan subconjuntos de 4, 8, 16, 32, 64, 128 canales priorizados por FTSM (Fig. 3).
- **Selección de canales:** no es una selección "dura" única, sino un *ranking* de prioridad por sujeto luego consolidado por repetición entre sujetos (frecuencia con que un canal aparece como "muy similar/redundante" o relevante en el ranking FTSM de cada sujeto). Los 4 canales top son 67, 68, 93, 94.
- **Frecuencia de muestreo:** no se especifica el valor numérico original de muestreo del EEG de MODMA en el cuerpo del texto (solo se menciona "cinco muestras por segundo" en la representación 2D, ver abajo) — esto es una laguna de reporting que deberías resolver mirando el dataset original de Cai et al. 2022 (MODMA), no el paper actual.
- **Filtrado:** no se describe ningún filtrado explícito (notch 50/60Hz, band-pass, ICA para artefactos oculares/musculares) en la sección de metodología. Esta es otra omisión importante — para un dataset real de EEG-depresión esto es prácticamente obligatorio y su ausencia en el texto es una debilidad de reporting (puede que lo hayan hecho y no lo documentaron, o puede que no lo hicieran).
- **Segmentación en ventanas / representación 2D (Sección 3.2.1, Fig. 5):** Aquí está el corazón del preprocesamiento EEG. Convierten la señal en una "imagen" donde:
  - Eje vertical = canales (dimensión **espacial**), de arriba a abajo.
  - Cada canal se muestrea a "cinco muestras por segundo" convertidas a **píxeles en escala de grises** (dimensión **espectral**, según ellos — en realidad esto se parece más a una representación de amplitud/energía por muestra que a un espectro real; el término "espectral" es usado de forma laxa).
  - Eje horizontal = tiempo, generando una imagen de ~70,000 píxeles de largo.
  - Esta imagen larga se **redimensiona a 224×224** (interpolación, con pérdida de resolución temporal significativa) y se divide en **parches de 16×16**, siguiendo el esquema de patchify de ViT (Dosovitskiy et al. 2020).
  - Se aplica proyección lineal + embedding posicional sobre los tokens/parches, exactamente como en ViT.

  **Punto crítico:** esta transformación a imagen 2D + resize agresivo (de 70,000 a 224 píxeles en el eje temporal, ~300:1) implica una pérdida masiva de resolución temporal. Es una decisión de diseño cuestionable si el objetivo es capturar dinámica temporal fina del EEG — sacrifica mucha información temporal en pos de reutilizar arquitecturas de visión (ViT) preentrenadas/estándar.

- **Normalización:** no se detalla explícitamente un paso de normalización de amplitud EEG antes de la conversión a escala de grises (aunque implícitamente debe existir un mapeo min-max o z-score para llevar amplitudes a rango de píxel [0,255]; el paper no lo explicita).
- **Representación final:** tokens de imagen (parches 16×16 de una imagen 224×224), con embedding posicional, listos para atención.

### Audio

- **Tipo de señal:** grabación de audio de entrevista, con dos streams paralelos:
  1. **Stream lingüístico:** ASR → texto → tokenización subword (tokenizer preentrenado) → embeddings numéricos.
  2. **Stream paralingüístico:** STFT → espectrograma en dB (Ecs. 2-3) → features de pitch, volumen, centroide espectral.
- **Preprocesamiento de audio (3.1.2):** chequeo de integridad de archivos, normalización de volumen, resampleo a 44.1kHz.
- **STFT:** ventana de 0.125s, rango de visualización 0-5000Hz, rango de pitch de voz mapeado a 75-265 (asumiendo rango fisiológico de pitch humano 85-255Hz).
- **Segmentación temporal:** ventanas STFT solapadas (overlapping frames), tamaño de ventana 0.125s — no se especifica el hop size/overlap exacto.
- **Sincronización con EEG:** este es el punto donde entra el **Módulo de Sincronización de Modalidades (Algoritmo 2)**, que es la respuesta explícita del paper al problema de desalineamiento temporal EEG-audio (ver Sección 4 más abajo, es más un mecanismo de fusión/entrenamiento que de preprocesamiento puro).
- **Representación final:** espectrograma como "imagen" (paralingüístico) + secuencia de tokens de texto (lingüístico), ambos proyectados linealmente y con embedding posicional antes del cross-attention.

**Mecanismo de reducción de canales EEG:** sí existe, es FTSM aplicado en la etapa de *Signal Preparation* (antes de que los datos entren a la red), es decir, es una selección **fuera de línea y agnóstica al modelo**, no aprendida end-to-end dentro de la arquitectura (aunque después, en 4.5, también reportan un análisis de "channel attention" aprendido dentro del modelo vía pesos de atención — esto es un análisis post-hoc, no un mecanismo de selección arquitectónico). La motivación explícita es la portabilidad de dispositivos EEG de bajo costo (reducir de 128 a pocos electrodos).

---

## 4. Arquitectura completa del modelo

Reconstruyendo el pipeline de la Fig. 2 bloque por bloque:

### a) Encoder EEG

- **Entrada:** señal EEG cruda multicanal (128/64/32/16/8/4 canales según el experimento).
- **Transformación:** conversión a imagen 2D (canales × muestras-por-segundo × tiempo) → resize a 224×224 → patchify 16×16 → 196 parches de 16×16×1 (grises) → proyección lineal a dimensión de embedding `d_model` (no especificada numéricamente en el texto) → suma de embedding posicional.
- **Módulo:** no es una CNN ni RNN — es efectivamente un **ViT-style patch embedding + transformer encoder** (self-attention, Sección 3.4) que opera sobre los parches de la imagen EEG.
- **Captura de información espacial/espectral/temporal:** se captura *implícitamente* por construcción de la imagen (eje = canal, valor de píxel = amplitud/"espectral", eje horizontal = tiempo) y luego el self-attention sobre los parches aprende relaciones entre regiones espacio-temporales de esa imagen. No hay un módulo separado explícito para cada uno de los tres aspectos (serían "fusionados" en la imagen antes de la red); toda la separación espacial/espectral/temporal ocurre a nivel de *representación de entrada*, no a nivel de arquitectura (no hay ramas paralelas espacial/espectral/temporal dentro de la red, a diferencia de arquitecturas dual-stream tipo STSNet [54] que citan como inspiración de la representación pero no replican su arquitectura de ramas).

### b) Encoder audio

- Dos sub-encoders paralelos (aunque el texto no dice si comparten pesos):
  - Texto: embeddings de tokens (tipo BERT-like, no se especifica el tokenizer/backbone concreto).
  - Espectrograma: tratado igual que la imagen EEG, como parches para atención (aunque no se detalla si también pasa por patchify 16×16 estilo ViT — se asume análogo por consistencia con Fig. 2).
- Busca capturar **prosodia, pitch, volumen, centroide espectral** (paralingüístico) y **contenido semántico/léxico** (lingüístico).

### c) Self-attention

- Sí existe, aplicada **dentro de cada modalidad** (intra-modal), en el bloque final antes de clasificación (Fig. 2, bloque rojo "Self-attention Block × 10" — repetido 10 veces, sugiriendo 10 capas apiladas).
- Aprende dependencias de largo alcance entre tokens de la misma modalidad (ej. relaciones entre parches distantes de la imagen EEG, o entre palabras distantes del texto).
- Fórmula estándar de transformer (Ecs. 4-6):

```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) · V
Z^(l+1) = MLP(Norm(Attention(Q,K,V) + Z^l))
Q = Z^l W_Q,  K = Z^l W_K,  V = Z^l W_V
```

### d) Cross-modal attention

- Ubicada en el bloque azul "Multi-head Cross-attention × 2" (Fig. 2) — sugiere 2 capas apiladas de cross-attention, antes del self-attention block.
- **Quién genera Q, K, V:** el texto es explícito en que "Q, K, V vienen de diferentes tipos de datos tokenizados, como Q_T" — es decir, una modalidad aporta la Query y otra aporta Key/Value (patrón estándar de cross-attention tipo "modality A attends to modality B": `Q = f(A), K = V = f(B)`). El paper indica que se calculan scores de atención entre **audio e interview text**, y por separado se examina la conexión total de esas dos con las **imágenes EEG**. Esto sugiere una jerarquía de dos pasos: primero fusión audio↔texto, luego esa fusión combinada atiende (o es atendida por) EEG.
- **¿Es bidireccional?** El paper lo llama explícitamente "mecanismo de atención multimodal bidireccional" — es decir, no es solo EEG→audio sino también audio→EEG (cross-attention en ambas direcciones, cada una con su propio conjunto Q/K/V y luego se combinan/"unifican").
- **Qué significa la interacción aprendida:** cada token de una modalidad (ej. un parche EEG) obtiene una distribución de atención sobre los tokens de la otra modalidad (ej. frames de espectrograma o tokens de texto), permitiendo que el modelo aprenda, por ejemplo, qué momentos de la voz correlacionan con qué patrones EEG — sin que esa correlación esté hard-coded.

### e) Fusión multimodal

- Es **atención cruzada** (no concatenación simple, no suma simple). Tras calcular los scores de cross-attention e intra-attention, el texto dice que se "integran" ("These cross-modal attention scores are then integrated with the intra-modal attention scores to create a unified representation") — el diagrama (Fig. 2) muestra un bloque de "**Unification**" después de las ramas de multi-head cross-attention, sugiriendo una operación de combinación (posiblemente concatenación seguida de proyección lineal, o suma ponderada) antes de pasar al self-attention block conjunto.
- **Por qué esta estrategia:** los autores argumentan que la atención permite captura de relaciones *dependientes del contenido* entre modalidades, superando a concatenación estática o a fusión basada en grafos (GNN), que —según ellos— maneja peor la falta de sincronización temporal entre modalidades.
- Adicionalmente, el **Módulo de Sincronización de Modalidades (Algoritmo 2)** actúa como un mecanismo de fusión *auxiliar* con su propia función de pérdida (Sync Loss), entrenado para maximizar la similitud coseno entre EEG y audio en el offset temporal correcto:

```
similarity(A,E) = (1/T) Σ_t (A_t · E_t) / (||A_t|| ||E_t||)
p(o) = exp(similarity(o)) / Σ_o' exp(similarity(o'))
L_sync = - Σ_o label_o · log(p(o))
```

con label smoothing sobre offsets vecinos (factor ε), y augmentación por desplazamiento aleatorio de secuencias durante el entrenamiento para simular asincronía real.

### f) Clasificador final

- Recibe el "latent array" combinado tras el self-attention block.
- Capas: cabeza de clasificación lineal (no se detalla número de capas exacto) → logits para 2 clases ("normal"/"depressed") → softmax.
- Función de pérdida: **binary cross-entropy**.
- Estrategia de entrenamiento: Adam (implícito, no mencionado explícitamente el optimizador — otra omisión), LR=1e-4, batch=32, ~38 épocas, grid search para hiperparámetros.

---

## 5. Inspiración arquitectónica para tu trabajo (EEG+audio, MODMA, ~50 sujetos)

| Componente | Veredicto | Razón |
|---|---|---|
| Convertir EEG a imagen 2D + ViT patchify | **Evitar / Modificar fuertemente** | Con ~53 sujetos, un ViT desde cero es una arquitectura hambrienta de datos; el propio paper reporta overfitting notable (Fig. 6, gap train/val). Además el resize 70,000→224 px destruye resolución temporal. Para MODMA con pocos sujetos, un encoder CNN (EEGNet/ShallowConvNet/DeepConvNet, que ya usas) es más apropiado por su inductive bias espacial-temporal y menor número de parámetros. |
| Self-attention dentro de cada modalidad | **Modificar** | Útil como concepto, pero considera reemplazar 10 capas de self-attention (excesivo para pocos datos) por 1-2 capas ligeras aplicadas sobre embeddings ya comprimidos (ej. tus embeddings de 64-d), no sobre secuencias largas de tokens crudos. |
| Cross-modal attention (Q de una modalidad, K/V de otra) | **Recomendable replicar (adaptado)** | Este es el componente de mayor valor conceptual y el más transferible: aplicar cross-attention entre tu embedding EEG (64-d) y tu embedding de audio (Mel-spectrogram embedding) para modelar interacción aprendida, en vez de concatenación fija. Con embeddings de baja dimensión (64-d) el riesgo de overfitting de un cross-attention pequeño es mucho menor que aplicarlo sobre secuencias completas de tokens. |
| Módulo de sincronización temporal (offset prediction + sync loss) | **Recomendable replicar, si tu protocolo tiene desalineamiento real** | En MODMA, EEG y audio se graban simultáneamente en el mismo protocolo experimental (a diferencia de DAIC-WOZ donde no hay EEG); si tu pipeline garantiza alineación por diseño (mismo timestamp de inicio, mismo dispositivo de sincronización), este módulo pierde utilidad práctica y solo añadiría un hiperparámetro/pérdida auxiliar sin beneficio claro. Si en cambio hay jitter de grabación real, es un componente elegante y barato de incorporar como pérdida auxiliar (multi-task) sin cambiar tu arquitectura principal — puedes calcularlo directamente sobre tus embeddings de 64-d en vez de sobre secuencias crudas.|
| FTSM para selección de canales EEG | **Modificar** | La idea de priorizar canales por redundancia/similitud (FTSM ≈ DTW/shapeDTW) es razonable y barata de aplicar a 64 canales, pero no está validada contra métodos más simples y ya establecidos (ranking por correlación, PSD por banda, atención aprendida end-to-end vía Grad-CAM como Wang et al. [38]). Te conviene *comparar* FTSM contra un ranking basado en tu propia atención aprendida (ver Sección 6) en vez de asumir que FTSM es superior. |
| Clasificador final simple (MLP + softmax + BCE) | **Recomendable replicar** | Es la elección estándar y correcta; combina bien con tu esquema de linear probing sobre embeddings de 64-d. |
| Vision Transformer completo de gran tamaño como backbone | **Evitar** | Explícitamente desaconsejado por el propio paper (mencionan el gap train/val como síntoma de overfitting), y menos aún justificable con 53 sujetos MODMA cuando tú ya tienes encoders EEG específicos (EEGNet, DeepConvNet, ShallowConvNet) con muchos menos parámetros y mejor prior espacio-temporal para EEG. |

---

## 6. Análisis específico de la selección de canales EEG (FTSM)

**¿Seleccionan canales?** Sí, pero es más preciso llamarlo *priorización/ranking* que "selección" en el sentido de un método de selección de features supervisado (no usan mutual information con la etiqueta, no usan un clasificador para rankear, no usan PSO como Shen et al. [48]).

**¿Cómo lo hacen?** El Algoritmo 1 (FTSM) es, en su forma matemática, **DTW (Dynamic Time Warping) clásico** — alinean de forma no lineal dos secuencias temporales minimizando una función de costo acumulado:

```
D(i,j) = d(x_i, y_j) + min{ D(i-1,j), D(i,j-1), D(i-1,j-1) }
```

con `d(x_i,y_j) = |x_i - y_j|` (distancia absoluta). El nombre "FTSM" y la cita a Zhao & Itti (shapeDTW, 2018) sugieren que en realidad están usando una variante de DTW con features de forma local, aunque el algoritmo presentado en el texto (Algoritmo 1) es DTW estándar puro sin el paso de extracción de descriptores de forma local que shapeDTW añade sobre el DTW clásico — es decir, hay una posible inconsistencia entre lo que citan (shapeDTW) y lo que documentan (DTW simple).

Para cada sujeto, calculan la similitud FTSM entre cada canal y todos los demás canales (matriz de similitud canal×canal por sujeto), y luego **priorizan canales que se repiten como "similares" o relevantes a través de sujetos distintos** — esto es agregación de rankings intra-sujeto a un ranking poblacional, una heurística razonable pero sin justificación estadística formal (no explican si usan promedio de rankings, votación, frecuencia de aparición en el top-k, etc.).

**¿Fisiológico, ML o heurística?** Es una **heurística puramente basada en similitud de series temporales**, sin usar la etiqueta de depresión ni ningún modelo de aprendizaje automático — de ahí su ventaja declarada de ser "agnóstica al modelo de clasificación" (channel priority independiente del clasificador usado después). No incorpora conocimiento fisiológico a priori (por ejemplo, regiones frontal/temporal asociadas a depresión en literatura de asimetría frontal alfa) — es puramente data-driven sobre la forma de la señal.

**¿Antes o dentro de la red?** **Antes** — es una etapa de *Signal Preparation*, completamente desacoplada del entrenamiento del transformer. Esto contrasta con el análisis posterior de "channel attention" (Sección 4.5, Fig. 9), que sí es un análisis *dentro* del modelo entrenado (pesos de atención aprendidos), pero ese análisis es post-hoc/interpretativo, no un mecanismo de selección usado para reducir canales de entrada.

**Efecto sobre rendimiento:** Fig. 6 muestra que con 4 canales el accuracy cae de ~91% (128 canales) a ~84%, y esto se replica de forma más marcada cuando solo se usa EEG puro (curva "E") vs. cuando se combinan modalidades (E,A,T) — la caída relativa por reducción de canales es menor cuando hay más modalidades disponibles, lo cual es un argumento razonable a favor de la multimodalidad como mitigador de la pérdida de información EEG.

**Cómo aplicarlo a 64 canales:**
1. Computa la matriz de similitud canal×canal por sujeto usando DTW (o shapeDTW si quieres seguir fielmente la cita [51]) sobre ventanas cortas (ej. 2-4s) de EEG crudo o de banda filtrada.
2. Agrega el ranking entre sujetos (recomendaría usar la mediana del ranking, más robusta a outliers de sujeto que el promedio simple).
3. **Crítico:** valida el ranking FTSM contra un ranking alternativo *supervisado* (ej. importancia de canal vía gradientes/Grad-CAM de tu encoder EEGNet, o simplemente accuracy leave-one-channel-out) — el propio paper no compara FTSM con un baseline supervisado de selección de canales, lo cual es una debilidad que puedes convertir en una contribución de tu trabajo (comparar priorización no supervisada de forma de onda vs. priorización supervisada por gradiente).
4. Dado que MODMA tiene 64 canales reales, replica el experimento de 4/8/16/32/64 en vez de hasta 128 (ellos usan 128 nativos de su propio MODMA — cuidado: **el MODMA público estándar en la literatura suele reportarse con 128 canales**, así que confirma cuál versión de MODMA tienes tú, con 64 canales, antes de comparar cifras directamente).

---

## 7. Resultados experimentales

**Métricas:** Accuracy, Precision, Recall, F1 — completas y correctas para clasificación binaria.

**Comparación con baselines (Tabla 2):** Comparan contra 7 métodos, todos ellos **basados en GNN o graph pooling** (HGP-SL, AM-GCN, SAGE, CGIPool, SGP-SL, MS2-GNN, G-Atten). Esto es una limitación de la comparación: **no comparan contra ningún transformer multimodal previo de los que citan en Related Work** (TensorFormer, DepMSTAT, MTNet), a pesar de mencionarlos extensamente como estado del arte transformer-based. Esto debilita la afirmación de "supera al estado del arte" porque el estado del arte transformer-based queda fuera de la tabla comparativa — solo comparan contra arquitecturas GNN, que es una familia distinta y potencialmente menos competitiva en primer lugar.

En MODMA: pasan de 90.35% (mejor baseline, G-Atten) a 91.22% — mejora modesta (~1 punto porcentual), con p=0.0006 (significativo pero con una diferencia de accuracy pequeña en términos absolutos, y con IC 95% [90.79, 91.73] que casi se solapa con el rendimiento reportado de G-Atten si este también tuviera varianza, que no se reporta para los baselines).

En DAIC-WOZ: mejora más clara, de 92.21% (G-Atten) a 94.17%.

**Resultados unimodales vs. multimodales (Fig. 6):** EEG solo (E) tiene el rendimiento más bajo consistentemente across todos los tamaños de canal; añadir audio (E,A) ayuda más que añadir texto (E,T); la combinación completa (E,A,T) es superior en todos los casos. Esto es consistente con el hallazgo de SHAP (Fig. 7): EEG tiene el mayor impacto en la salida del modelo, seguido de audio, y luego texto — es decir, la señal paralingüística aporta más que la lingüística en este dataset, lo cual es razonable dado que MODMA usa un protocolo de elicitación emocional (no necesariamente conversación libre rica en contenido semántico como en una entrevista clínica DAIC-WOZ).

**Ablation studies (Tabla 3) — el resultado más informativo del paper:**

| Cross-att. | Sync | Self-att. | ACC% |
|---|---|---|---|
| ✗ | ✗ | ✗ | 58.52 |
| ✗ | ✗ | ✓ | 62.16 |
| ✓ | ✗ | ✗ | 64.63 |
| ✓ | ✗ | ✓ | 87.16 |
| ✓ | ✓ | ✗ | 75.08 |
| ✓ | ✓ | ✓ | 91.22 |

Interpretación cuantitativa:
- El salto más grande ocurre al **combinar cross-attention + self-attention** (64.63%→87.16%, +22.5 puntos) — esto sugiere que ninguno de los dos mecanismos por sí solo es suficiente; se necesitan ambos, lo cual valida la arquitectura de dos etapas (fusión + refinamiento intra-modal).
- El módulo de sincronización aporta una mejora adicional más modesta pero no trivial: sin sync (87.16%) vs. con sync (91.22%), +4 puntos — consistente con la afirmación del texto de que su impacto es "comparativamente menor" que los bloques de atención, pero de ninguna manera despreciable.
- Fila 5 (cross-att + sync, sin self-att) da 75.08%, **peor** que cross-att+self-att sin sync (87.16%) — esto es un dato interesante y algo contraintuitivo: sugiere que el módulo de sincronización, sin el refinamiento de self-attention posterior, puede introducir ruido/complejidad que no se aprovecha bien. Es una interacción no explicada por los autores y sería un buen punto para preguntar/profundizar si replicas esto.

**Impacto de hiperparámetros (Fig. 8):** comportamiento en U/campana esperado para LR (óptimo en 1e-4, degradación por LR alto por oscilación y por LR bajo por underfitting) y meseta esperada para batch size (mejora hasta 32, luego plateau) y épocas (mejora hasta ~38, luego plateau) — resultados estándar sin sorpresas, correctamente interpretados por los autores.

**Channel attention (Fig. 9, Sección 4.5):** análisis cualitativo (visual) de mapas de atención por canal sobre 3 sujetos sanos y 3 con MDD. Reportan que las áreas de "alto peso" son minoritarias (pocos canales relevantes), y que en sujetos con MDD los puntos de atención alta están "más dispersos", con menos actividad relevante frontal/izquierda y más en posterior/derecha. **Ojo:** esto es un análisis puramente descriptivo sobre n=6 sujetos (3 vs 3), sin ningún test estadístico — no se puede interpretar como un hallazgo neurofisiológico validado, es una observación exploratoria. Ellos mismos son honestos al decir "no significant visual distinctions were observed" en términos generales.

---

## 8. Crítica científica (como revisor)

**Fortalezas:**
1. Arquitectura conceptualmente coherente y bien motivada (fusión + sincronización + reducción de canales atacan tres problemas reales y distintos).
2. Ablation study (Tabla 3) es el elemento más sólido del paper — aísla correctamente la contribución de cada bloque.
3. Validación estadística con múltiples corridas (10 runs, IC 95%, t-tests) es más rigurosa que lo típico en este subcampo.
4. Uso de dos datasets (MODMA + DAIC-WOZ) para generalización parcial de la comparación (aunque con modalidades distintas en cada uno).
5. FTSM para reducción de canales aborda un problema práctico real de traducción a dispositivos wearables, no solo un ejercicio académico.

**Debilidades:**
1. **Comparación de baselines incompleta:** todos los baselines de la Tabla 2 son GNN; ningún transformer multimodal (TensorFormer, DepMSTAT, MTNet) — exactamente los métodos que citan como "estado del arte transformer-based" en Related Work — es incluido en la comparación cuantitativa. Esto es una omisión que un revisor exigente señalaría como sesgo de comparación (straw-man baselines).
2. **Riesgo de leakage no descartado:** no se especifica split por sujeto vs. por ventana en MODMA (53 sujetos). Con arquitecturas ViT-like hambrientas de datos y sesiones largas segmentadas en muchas ventanas/imágenes, esto es un riesgo metodológico serio no abordado explícitamente.
3. **Falta de detalles de preprocesamiento EEG:** no se reporta filtrado (banda, notch), no se reporta manejo de artefactos oculares/musculares, no se reporta la frecuencia de muestreo original ni el paso de normalización de amplitud antes de la conversión a imagen — detalles esenciales para reproducibilidad.
4. **Desproporción modelo/datos:** un backbone tipo ViT (con miles de parámetros de atención, 10 capas de self-attention según Fig. 2) entrenado sobre 53 sujetos de MODMA es una relación parámetros/sujetos muy alta. Los propios autores admiten el overfitting (Fig. 6), lo cual valida la preocupación, aunque argumentan que la multimodalidad + sincronización lo mitiga — la mitigación es parcial, no elimina el riesgo estructural de sobreajuste con tan pocos sujetos.
5. **Inconsistencia FTSM/shapeDTW:** citan shapeDTW [51] pero el Algoritmo 1 documentado es DTW clásico sin el paso de descriptores de forma local — o la descripción del algoritmo está incompleta, o el nombre "FTSM" es un rebranding de DTW estándar con una capacidad real menor a la sugerida por la cita.
6. **Análisis de channel attention (Fig. 9) es anecdótico:** n=6, sin estadística, presentado con lenguaje interpretativo neurofisiológico (front/left vs. back/right) que no está respaldado por un análisis cuantitativo formal ni por literatura de asimetría frontal citada en ese punto específico.
7. **Sin corrección por comparaciones múltiples** en los t-tests contra 7 baselines simultáneamente.
8. **Reproducibilidad limitada:** "Data will be made available on request" (no público directo), y faltan hiperparámetros clave (dimensión de embedding `d_model`, número de heads, optimizador, dropout, tamaño de vocabulario del tokenizer, backbone ASR usado).

**¿La cantidad de sujetos justifica la complejidad del modelo?** No completamente. Con 53 sujetos, un modelo con backbone ViT + 2 capas de cross-attention + 10 capas de self-attention es sobredimensionado; el propio paper documenta el síntoma (gap train/val) sin resolverlo estructuralmente (no aplican pretraining, no aplican data augmentation explícito más allá del offset shifting para sync, no usan modelos más pequeños como comparación de tamaño-controlado).

**¿Los resultados son generalizables?** Limitado. Los propios autores lo reconocen en las limitaciones ("limited generalizability beyond evaluated datasets, susceptibility to real-world noise"). La mejora sobre el mejor baseline en MODMA es de solo ~1 punto porcentual, dentro de un rango donde la varianza entre semillas/folds podría ser comparable si se reportara para los baselines también.

---

## 9. Aplicación a tu arquitectura CAMPNet

Dado tu setup actual (EEG MODMA ~53 sujetos/64 canales, audio MODMA ~52 sujetos, encoders EEGNet/ShallowConvNet/DeepConvNet, embeddings bottleneck de 64-d, evaluación por linear probing), aquí está mi propuesta concreta:

### Qué módulos agregaría

1. **Cross-modal attention ligero sobre embeddings de 64-d**, no sobre secuencias largas de tokens crudos. Concretamente:
   - Si tus encoders producen un solo vector de 64-d por ventana/trial (embedding global), primero necesitas una representación *secuencial* de menor granularidad para que cross-attention tenga sentido (atención sobre un solo token es trivial). Dos opciones:
     - (a) Extraer embeddings intermedios *por segmento temporal* (ej. dividir el trial en k sub-ventanas de 1-2s, generar k vectores de 64-d por modalidad) y aplicar cross-attention EEG↔audio sobre esos k tokens.
     - (b) Mantener el embedding global de 64-d por modalidad y aplicar un cross-attention "single-token" equivalente a un gating/bilinear interaction — más simple y con menos riesgo de overfitting dado tu tamaño de dataset.
   - Recomendación dado tu N pequeño: empieza con (b), mide si mejora sobre concatenación simple con linear probing; si hay señal, escala a (a).

2. **Un mecanismo de fusión con gating aprendido** (en vez de cross-attention completo) como alternativa más ligera y menos hambrienta de datos: `z_fused = σ(W_g[z_eeg; z_audio]) ⊙ z_eeg + (1-σ(...)) ⊙ z_audio` — mucho más barato en parámetros que un bloque de atención completo, y con menor riesgo de overfitting en régimen de pocos sujetos.

3. **Sync loss auxiliar solo si tu protocolo de grabación EEG-audio en MODMA tiene desalineamiento real conocido.** Si tus timestamps ya están alineados por diseño (protocolo MODMA sincronizado por hardware), omite este módulo — añadiría complejidad sin beneficio, y el propio ablation study del paper (fila 5 de la Tabla 3) muestra que el sync module sin buen self-attention posterior puede incluso *perjudicar* el rendimiento.

4. **Selección de canales:** en vez de adoptar FTSM tal cual, te propongo un experimento comparativo (ver más abajo) entre FTSM/DTW no supervisado y una priorización supervisada derivada de gradientes de tu propio encoder EEGNet (channel-wise Grad-CAM o ablation leave-one-channel-out), documentando cuál generaliza mejor con 64 canales.

### Dónde pondría el cross-modal attention

Justo **después** de tus encoders EEGNet/DeepConvNet/ShallowConvNet (que ya reducen la señal cruda a un embedding compacto), y **antes** de la etapa de linear probing. Es decir:

```
EEG raw → [EEGNet/DeepConvNet/ShallowConvNet] → z_eeg (64-d)
Audio raw (Mel-spec) → [CNN encoder audio] → z_audio (64-d)
[z_eeg, z_audio] → Cross-Attention / Gated Fusion → z_fused (64-d o 128-d)
z_fused → Linear probe → clasificación MDD/sano
```

Esto preserva tu pipeline de evaluación (linear probing) casi intacto — el linear probe pasa a operar sobre `z_fused` en vez de sobre `z_eeg` solo, permitiéndote comparar directamente el valor añadido de la fusión.

### Qué experimentos haría

1. Linear probing con `z_eeg` solo, `z_audio` solo, concatenación simple `[z_eeg;z_audio]`, y `z_fused` (cross-attention/gating) — replicando la lógica de la Fig. 6 del paper (E vs A vs E+A) pero en tu setup de embeddings, no de secuencias de imagen completa.
2. Ablation equivalente a la Tabla 3: con/sin cross-attention, con/sin sync loss (si aplica), con/sin self-attention adicional sobre `z_fused`.
3. Split **estrictamente por sujeto** (leave-N-subjects-out o k-fold subject-wise) para MODMA, reportando explícitamente que no hay leakage — esto es algo que el paper de Esmi et al. no garantiza claramente y que tú puedes convertir en una fortaleza metodológica de tu trabajo.
4. Comparación de priorización de canales: FTSM/DTW no supervisado vs. importancia supervisada (gradiente/Grad-CAM) vs. conocimiento fisiológico previo (regiones frontal/temporal asociadas a asimetría depresiva) — con curvas de accuracy vs. número de canales (4/8/16/32/64) análogas a la Fig. 6, pero evaluando ambos métodos de ranking.
5. Análisis SHAP (como su Fig. 7) sobre tus embeddings de 64-d para ver contribución relativa EEG vs. audio a la predicción final.

### Qué ablations serían necesarias para publicar un trabajo sólido

- **Ablation de cada componente de fusión** (concatenación vs. gating vs. cross-attention) manteniendo el mismo encoder EEG/audio — aislando el efecto puro del mecanismo de fusión, algo que el paper de Esmi et al. no hace (ellos no comparan cross-attention contra concatenación simple con la misma arquitectura base).
- **Ablation de tamaño de modelo controlado por número de parámetros** — para descartar que cualquier mejora se deba solo a "más parámetros" en vez de al mecanismo de fusión en sí (control que el paper original tampoco reporta).
- **Ablation subject-wise vs. window-wise split** — reportar ambos para demostrar cuantitativamente el tamaño del posible leakage, algo que fortalecería tu credibilidad frente al problema que detecté en la Sección 2.
- **Sensibilidad a semilla aleatoria / validación cruzada con múltiples folds subject-independent**, reportando media ± desviación estándar como hace el paper (buena práctica a mantener).
- **Ablation de número de canales EEG** (4/8/16/32/64) cruzado con presencia/ausencia de audio, replicando el hallazgo de que la multimodalidad mitiga la pérdida por reducción de canales — esto sería una contribución fuerte y directamente comparable a la Fig. 6 del paper de referencia.
- **Comparación contra al menos un transformer multimodal previo** (no solo GNN), corrigiendo la limitación más señalable del paper analizado.

---

### Nota final sobre reproducibilidad de cifras

Ten en cuenta que las cifras exactas de canales EEG usadas en el paper (128 nativos de su versión de MODMA) pueden no coincidir con la versión de MODMA que tú tienes (64 canales). Antes de comparar tu accuracy directamente contra su 91.22%, confirma diferencias de protocolo (duración de sesión, tarea de elicitación emocional vs. entrevista libre, número de canales EEG del dataset, si usan la misma partición de sujetos MDD/sanos) — no son directamente comparables sin ese control.

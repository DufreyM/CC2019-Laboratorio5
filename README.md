VIDEO: https://www.canva.com/design/DAG4be9CNRI/hNYkT3YAH_Ro9SIvFwnT8Q/watch?utm_content=DAG4be9CNRI&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h8dadb08331

Laboratorio 5

------------------------------------------------------------------------

Descripción general

La escena muestra una esfera (modelo sphere.obj) que se deforma y brilla
de forma continua, emitiendo luz variable en función del ruido y el
tiempo.
El resultado es una estrella pulsante con apariencia realista,
coloración térmica y leves destellos superficiales.

------------------------------------------------------------------------

Uniformes principales

  ------------------------------------------------------------------------
  Uniform                 Tipo           Descripción
  ----------------------- -------------- ---------------------------------
  time                    float          Controla la animación continua;
                                         se actualiza en cada frame.

  noiseScale              float          Escala espacial del ruido Perlin;
                                         define el tamaño de los patrones
                                         de turbulencia.

  noiseAmplitude          float          Intensidad del desplazamiento de
                                         vértices (flare superficial).

  vertexTwist             float          Factor de pulsación/distorsión
                                         adicional basado en seno del
                                         tiempo.

  emissionBoost           float          Multiplicador de brillo global de
                                         la estrella.

  pulseAmp                float          Amplitud de las pulsaciones
                                         periódicas de emisión.

  tempCold / tempHot      float          Representan temperaturas
                                         relativas (solo referenciales
                                         para gradiente visual).

  uModel, uView, uProj    mat4           Matrices de transformación
                                         estándar.

  viewPos                 vec3           Posición de la cámara (usada para
                                         dirección de luz básica).
  ------------------------------------------------------------------------

------------------------------------------------------------------------
Funciones clave del shader

float cnoise(vec3 P)

Implementación compacta de Classic Perlin Noise en 3D.
Genera valores pseudoaleatorios suaves basados en coordenadas, usados
para distorsionar la superficie y modular la intensidad lumínica.

-   Se usa en tres octavas (noise, n2, n3) con distintas frecuencias y
    fases.
-   Combinación ponderada: 0.6 * noise + 0.3 * n2 + 0.1 * n3.

vec3 gradient_color(float t)

Genera un gradiente térmico dinámico: - t = 0.0: naranja profundo
(estrella fría).
- t = 1.0: blanco brillante (estrella caliente).
El valor t depende de la intensidad de emisión calculada con el ruido y
el tiempo.

------------------------------------------------------------------------

Animación y realismo

-   Tiempo (time) se actualiza constantemente → produce movimiento
    fluido en el ruido y los pulsos.
-   Desplazamiento de vértices (normal * combined * noiseAmplitude) crea
    el efecto de flare dinámico.
-   Emisión variable controlada por sin(time * 3.0 + vNoise * 10.0)
    simula pulsos energéticos.
-   Gradiente de color térmico reacciona a la intensidad para imitar la
    variación de temperatura superficial.

------------------------------------------------------------------------

🧠 Cómo ejecutar

1.  Coloca un modelo de esfera en assets/sphere.obj.

2.  Ejecuta:

        cargo run --release

3.  Usa ESC para salir.

------------------------------------------------------------------------

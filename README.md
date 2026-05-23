# 🏴‍☠️ Barco Pirata — Aprendizaje por Refuerzo

**Asignatura:** Aprendizaje Automático II — Bloque RL  
**Grado:** Ciencia de Datos e Inteligencia Artificial · ETSISI UPM  
**Autores:** Brian Bedoya Piedrahita · Daniel Moraleda Sánchez · David Santiago Ruiz

---

## ¿De qué va el proyecto?

Implementamos un entorno de navegación personalizado usando la API de Gymnasium donde un agente controla un barco pirata en un mapa 8×10. El objetivo: recoger los cuatro cofres del mapa (de distintos valores) y regresar al puerto antes de agotar las provisiones, adaptando la ruta a un viento que cambia de dirección durante el episodio.

Lo que hace este entorno más interesante que los clásicos (FrozenLake, Taxi, CartPole) es que combina **cinco fuentes de complejidad simultáneas**:

- **Viento variable en el estado** — la misma acción tiene distribución de transición distinta según sople el viento. Sin `dir_viento` en el estado se viola la propiedad de Markov, así que el agente aprende a explotar viento de cola y evitar rutas contra el viento.
- **4 cofres con valores distintos** (+20, +35, +50, +70) — existen 4! = 24 órdenes posibles de recogida. La política óptima pondera riesgo y recompensa: ¿merece la pena ir a por el legendario con pocas provisiones y viento en contra?
- **Provisiones discretizadas en 4 niveles** — la política cambia cualitativamente entre niveles: con provisiones CRÍTICAS el agente debe ir al puerto ignorando cofres cercanos.
- **Corrientes marinas** — desplazamiento determinista que el agente aprende a explotar como atajo o evitar según el contexto.
- **Maelstrom** — atrapa 2 turnos con penalización fija; el agente aprende una zona de evitación absoluta incluso cuando el camino más corto pasa por él.

El espacio de estados es `8 × 10 × 2⁴ × 4 × 4 = 20.480 estados` — manejable con métodos tabulares.

---

## Mapa del entorno (8×10)

```
     Col→  0     1     2     3     4     5     6     7     8     9
Fila 0  [ ⛵  ][  ~  ][  ~  ][ 🌴  ][  ~  ][  ~  ][  ~  ][  ~  ][  ~  ][  ~  ]
Fila 1  [  ~  ][ 🌴  ][  ~  ][  ~  ][  ~  ][ 💰+35][  ~  ][ 🌴  ][  ~  ][  ~  ]
Fila 2  [  ~  ][  ~  ][  ~  ][ 🌧  ][  ~  ][  ~  ][  ~  ][  ~  ][ 🌊↓ ][  ~  ]
Fila 3  [  ~  ][  ~  ][ 🌧  ][  ~  ][  ~  ][ 🌴  ][  ~  ][  ~  ][ 🌊↓ ][  ~  ]
Fila 4  [  ~  ][ 🌴  ][  ~  ][ ⛈  ][  ~  ][  ~  ][ ⛈  ][  ~  ][  ~  ][ 💎+50]
Fila 5  [  ~  ][  ~  ][  ~  ][  ~  ][ 🌀  ][  ~  ][  ~  ][ 🌴  ][  ~  ][  ~  ]
Fila 6  [  ~  ][  ~  ][ ⛈  ][  ~  ][  ~  ][  ~  ][  ~  ][  ~  ][  ~  ][ 📦+20]
Fila 7  [ ⚓  ][  ~  ][  ~  ][  ~  ][  ~  ][  ~  ][  ~  ][  ~  ][ 👑+70][  ~  ]
```

| Símbolo | Tipo | Efecto |
|---|---|---|
| ⛵ | Inicio | Posición inicial del barco (0,0) |
| ⚓ | Puerto | Objetivo — llegar aquí con los 4 cofres da `+100` |
| 💰 📦 💎 👑 | Cofres | `+35 / +20 / +50 / +70` — se recogen al pisarlos |
| 🌴 | Isla | Celda bloqueada, el barco rebota |
| 🌧 | Tormenta moderada | Reduce `p_dir` en `−0.15` |
| ⛈ | Tormenta severa | Reduce `p_dir` en `−0.25` |
| 🌊↓ | Corriente sur | Desplazamiento determinista hacia el sur |
| 🌀 | Maelstrom | Atrapa 2 turnos · `−5` por turno · inescapable |

## Estructura del notebook

El notebook sigue exactamente las secciones que pide el enunciado. Cada sección alterna celdas markdown (análisis) con celdas de código (implementación):

| Sección | Contenido |
|---|---|
| **2.1 Motivación** | Por qué este problema es interesante para RL y no resoluble con A* |
| **2.2 Formalización MDP** | Definición completa de S, A, R, T, γ con justificación matemática |
| **2.3 Estado del arte** | Contexto en la literatura: SSP, Windy Gridworld, Cliff Walking, DQN |
| **2.4 Soluciones y experimentos** | Implementación + entrenamiento + resultados + gráficas + búsqueda HP + ablación |
| **2.5 Conclusiones** | Análisis crítico de resultados, limitaciones y trabajo futuro |

Dentro de **2.4**, el desarrollo completo es:

```
Constantes y mapa              ← tipos de celda, probabilidades de viento, mapa 8×10
PirateEnv                      ← entorno Gymnasium con step(), reset(), render() pygame
QLearningAgent                 ← off-policy, actualización por paso con max Q(s')
SARSAAgent                     ← on-policy, actualización con Q(s', A') real
MonteCarloAgent                ← every-visit, actualización al final del episodio
entrenar() + evaluar()         ← bucle genérico compatible con los tres agentes
Entrenamiento (500k ep)        ← ~40 min total, resultados guardados en .pkl
Resultados                     ← tabla resumen con métricas de evaluación
Gráficas convergencia          ← recompensa, victorias, epsilon decay
Gráficas política y valor      ← mapas de flechas y función de valor V(s)
Render pygame (en notebook)    ← animación paso a paso capturando frames SDL
Búsqueda de hiperparámetros    ← γ y ε_min para los 3 algoritmos, 12 experimentos
Análisis de ablación           ← Q-Learning sin maelstrom, validación del diseño
Celdas de referencia           ← código experimental comentado para documentación
2.5 Conclusiones               ← análisis crítico integrado de todos los resultados
```

---

## Por qué estos tres algoritmos

El enunciado pide un mínimo de dos algoritmos. Elegimos tres porque cubren los tres enfoques distintos de aprendizaje tabular y permiten un contraste experimental completo:

**Q-Learning (off-policy, TD)** es el algoritmo de referencia del bloque. Al usar `max Q(s', a')` en la actualización, aprende la política óptima π* independientemente de cómo explore — puede observar acciones exploratorias sin que contaminen la política que está aprendiendo.

**SARSA (on-policy, TD)** es el contraste natural de Q-Learning. Usa `Q(s', A')` — la acción *real* que tomará, incluyendo errores de exploración — lo que genera una política más conservadora. Este contraste on-policy vs off-policy es exactamente el experimento del **Cliff Walking** de Sutton & Barto (cap. 6.7). Las tormentas severas y el maelstrom juegan aquí el papel del acantilado.

**Monte Carlo (every-visit, sin bootstrapping)** añade una tercera dimensión: actualiza al final del episodio con el retorno real observado, eliminando el sesgo de bootstrapping a costa de mayor varianza y convergencia más lenta.

Los tres algoritmos comparten la misma interfaz (`elegir_accion`, `actualizar`, `decay`) y el mismo bucle de entrenamiento genérico.

---

## Cómo ejecutar el notebook

### Requisitos

```bash
pip install gymnasium numpy matplotlib pygame pillow
```

### Orden de ejecución completo

```
Celda 1  → Título y descripción (markdown)
Celda 2  → separador
...
Celda 9  → imports y constantes globales
Celda 10 → PirateEnv (entorno completo)
Celda 11 → QLearningAgent
Celda 12 → SARSAAgent
Celda 13 → MonteCarloAgent
Celda 14 → funciones entrenar() y evaluar()
```

Después, **dos opciones**:

**Si quieres reentrenar desde cero** (~40 min):
```
Celda de entrenamiento → entrena 500k × 3 algoritmos, guarda en resultados_entrenamiento.pkl
```

**Si ya tienes el `.pkl` en disco**:
```
Celda de carga → carga desde resultados_entrenamiento.pkl
```

Luego ejecuta el resto en orden: gráficas → render pygame → búsqueda HP → ablación.

> ⚠️ Si reinicias el kernel, ejecuta siempre las celdas de definición antes de la carga. Pickle necesita las clases en memoria para deserializar.

### Archivos `.pkl` generados

| Archivo | Contenido |
|---|---|
| `resultados_entrenamiento.pkl` | Q-tables y métricas de los 3 algoritmos (entrenamiento base) |
| `resultados_exp_qlearning.pkl` | 4 variaciones de hiperparámetros para Q-Learning |
| `resultados_exp_sarsa.pkl` | 4 variaciones de hiperparámetros para SARSA |
| `resultados_exp_montecarlo.pkl` | 4 variaciones de hiperparámetros para Monte Carlo |
| `resultados_ablacion_maelstrom.pkl` | Q-Learning entrenado en mapa sin maelstrom |

### Render pygame en Jupyter

El render usa `SDL_VIDEODRIVER=dummy` para capturar frames en memoria sin necesidad de display gráfico. La visualización se exporta como animación interactiva (jshtml) y como GIF en `plots/render_pygame.gif`.


## Cobertura del enunciado

| Requisito del enunciado | Estado | Dónde está |
|---|---|---|
| Entorno propio adaptado a Gymnasium API | ✅ | `PirateEnv` — implementación completa |
| Formalización MDP (S, A, R, T, γ) | ✅ | Sección 2.2 con justificación matemática |
| Mínimo 2 algoritmos de RL | ✅ | Q-Learning + SARSA + Monte Carlo |
| Contraste on-policy vs off-policy | ✅ | SARSA vs Q-Learning — analizado en 2.4 y 2.5 |
| Tabla de hiperparámetros justificada | ✅ | Sección 2.4 + búsqueda experimental |
| Gráficas de convergencia | ✅ | Recompensa, victorias, epsilon decay |
| Función de valor y mapa de política | ✅ | Visualizaciones por dirección de viento y nivel de provisiones |
| Render / visualización del entorno | ✅ | Pygame (frames SDL) + matplotlib animación |
| Estado del arte con ≥3 referencias | ✅ | Sección 2.3 — 6 referencias |
| Conclusiones con análisis crítico | ✅ | Sección 2.5 con resultados de ablación y HP |
| Búsqueda de hiperparámetros | ✅ *(extra)* | 12 experimentos sistemáticos |
| Análisis de ablación del entorno | ✅ *(extra)* | Eliminación del maelstrom — validación del diseño |

---

## Referencias

1. Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2ª ed.). MIT Press.
2. Watkins, C.J.C.H. & Dayan, P. (1992). Q-learning. *Machine Learning, 8*(3–4), 279–292.
3. Rummery, G.A. & Niranjan, M. (1994). *On-line Q-learning using connectionist systems.* Technical Report, Cambridge University.
4. Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature, 518*, 529–533.
5. Hart, P.E., Nilsson, N.J. & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics.*
6. Bellman, R. (1957). *Dynamic Programming.* Princeton University Press.

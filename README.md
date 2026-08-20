# 🏴‍☠️ Reinforcement Learning: Custom Pirate Ship Environment

> 🎓 **Academic Project** | *Universidad Politécnica de Madrid (UPM)*
> 👥 **Team Project:** Co-authored with my colleagues from the Data Science & AI degree.

### 📝 About the Project
This project explores tabular Reinforcement Learning (RL) by implementing a custom navigation environment from scratch using the Gymnasium API. The agent controls a pirate ship on an 8x10 grid with the objective of collecting four chests of varying values (+20, +35, +50, +70) and returning to port before running out of provisions.

Unlike standard benchmark environments, this custom gridworld combines five simultaneous sources of complexity: variable wind directions altering transition distributions, combinatorial chest collection strategies, discrete provision levels dictating policy shifts, deterministic sea currents, and a "Maelstrom" trap. The state space consists of 20,480 states, making it a highly challenging yet solvable problem for tabular methods.

### 👨‍💻 My Contribution & Role
As an active member of this team, my technical contributions included:

- **Algorithm Implementation & Comparison:** Developed and evaluated three distinct RL algorithms from scratch: Q-Learning (off-policy), SARSA (on-policy), and Monte Carlo (every-visit). Successfully achieved a 97.2% win rate using Q-Learning.
- **Environment Design:** Built the custom Gymnasium environment integrating Markov Decision Process (MDP) principles to handle stochastic transitions (variable wind) and complex reward functions.
- **Hyperparameter Optimization:** Designed a comprehensive 12-experiment search to tune the discount factor (γ) and exploration rate (ε_min) across all three algorithms, ensuring optimal convergence.
- **Ablation Studies:** Conducted an ablation analysis by removing the Maelstrom trap to validate the environment's design, proving the agent successfully learned active avoidance strategies.
- **Visual Rendering:** Integrated Pygame to capture SDL frames in memory (using a dummy video driver), generating interactive in-notebook animations and GIFs of the agent's learned policies.

---
*Note: This is a forked repository to showcase my specific contributions to this academic project. The original repository is hosted by my colleague.*

# 🏴‍☠️ Barco Pirata — Aprendizaje por Refuerzo

**Asignatura:** Aprendizaje Automático II — Bloque RL  
**Grado:** Ciencia de Datos e Inteligencia Artificial · ETSISI UPM  
**Autores:** Brian Bedoya Piedrahita · Daniel Moraleda Sánchez · David Santiago Ruiz

---

## ¿De qué va el proyecto?

Implementamos un entorno de navegación personalizado usando la API de Gymnasium donde un agente controla un barco pirata en un mapa 8x10. El objetivo: recoger los cuatro cofres del mapa (de distintos valores) y regresar al puerto antes de agotar las provisiones, adaptando la ruta a un viento que cambia de dirección durante el episodio.

Lo que hace este entorno más interesante que los clásicos (FrozenLake, Taxi, CartPole) es que combina **cinco fuentes de complejidad simultáneas**:

- **Viento variable en el estado** — la misma acción tiene distribución de transición distinta según sople el viento. Sin `dir_viento` en el estado se viola la propiedad de Markov, así que el agente aprende a explotar viento de cola y evitar rutas contra el viento.
- **4 cofres con valores distintos** (+20, +35, +50, +70) — existen 4! = 24 órdenes posibles de recogida. La política óptima pondera riesgo y recompensa: ¿merece la pena ir a por el legendario con pocas provisiones y viento en contra?
- **Provisiones discretizadas en 4 niveles** — la política cambia cualitativamente entre niveles: con provisiones CRÍTICAS el agente debe ir al puerto ignorando cofres cercanos.
- **Corrientes marinas** — desplazamiento determinista que el agente aprende a explotar como atajo o evitar según el contexto.
- **Maelstrom** — atrapa 2 turnos con penalización fija; el agente aprende una zona de evitación absoluta incluso cuando el camino más corto pasa por él.

El espacio de estados es `8 x 10 x 2⁴ x 4 x 4 = 20.480 estados` — manejable con métodos tabulares.

---

## Mapa del entorno (8x10)

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

---

## Estructura del notebook

El notebook sigue las secciones que pide el enunciado. Cada sección alterna celdas markdown (análisis teórico) con celdas de código (implementación):

| Sección | Contenido |
|---|---|
| **2.1 Motivación** | Por qué este problema es interesante para RL y no resoluble con A* |
| **2.2 Formalización MDP** | Definición completa de S, A, R, T, γ con justificación matemática |
| **2.3 Estado del arte** | Contexto en la literatura: SSP, Windy Gridworld, Cliff Walking, DQN |
| **2.4 Soluciones y experimentos** | Contraste on/off-policy · implementación · entrenamiento · resultados · gráficas · búsqueda HP · ablación |
| **2.5 Conclusiones** | Convergencia · observaciones · hiperparámetros · maelstrom · limitaciones · trabajo futuro |

Dentro de **2.4**, el desarrollo completo es:

```
Diferencias on-policy/off-policy <- análisis teórico del comportamiento esperado
Constantes y mapa                <- tipos de celda, probabilidades de viento, mapa 8x10
PirateEnv                       <- entorno Gymnasium con step(), reset(), render() pygame
QLearningAgent                  <- off-policy, actualización por paso con max Q(s')
SARSAAgent                      <- on-policy, actualización con Q(s', A') real
MonteCarloAgent                 <- every-visit, actualización al final del episodio
entrenar() + evaluar()          <- bucle genérico compatible con los tres agentes
Entrenamiento (500k ep)         <- ~30 min total, resultados guardados en .pkl
Resultados                      <- tabla resumen con métricas de evaluación
Gráficas convergencia           <- recompensa, victorias, epsilon decay
Gráficas política y valor       <- mapas de flechas y función de valor V(s)
Panel 4 vientos                 <- política aprendida para las 4 direcciones de viento
Render pygame (en notebook)     <- animación paso a paso capturando frames SDL
Búsqueda de hiperparámetros     <- γ y ε_min para los 3 algoritmos, 12 experimentos
Análisis de ablación            <- Q-Learning sin maelstrom, validación del diseño
```

---

## Por qué estos tres algoritmos

El enunciado pide un mínimo de dos algoritmos. Elegimos tres porque cubren los tres enfoques distintos de aprendizaje tabular y permiten un contraste experimental completo:

**Q-Learning (off-policy, TD)** es el algoritmo de referencia del bloque. Al usar `max Q(s', a')` en la actualización, aprende la política óptima π* independientemente de cómo explore — puede observar acciones exploratorias sin que contaminen la política que está aprendiendo.

**SARSA (on-policy, TD)** es el contraste natural de Q-Learning. Usa `Q(s', A')` — la acción *real* que tomará, incluyendo errores de exploración — lo que genera una política más conservadora. Este contraste on-policy vs off-policy es exactamente el experimento del **Cliff Walking** de Sutton & Barto (cap. 6.7). Las tormentas severas y el maelstrom juegan aquí el papel del acantilado.

**Monte Carlo (every-visit, sin bootstrapping)** añade una tercera dimensión: actualiza al final del episodio con el retorno real observado, eliminando el sesgo de bootstrapping a costa de mayor varianza y convergencia más lenta.

Los tres algoritmos comparten la misma interfaz (`elegir_accion`, `actualizar`, `decay`) y el mismo bucle de entrenamiento genérico.

---

## Resultados

### Entrenamiento base (500k episodios por algoritmo)

| Métrica | Q-Learning | SARSA | Monte Carlo |
|---|---|---|---|
| Tasa victoria (eval ε=0) | **97.2%** | 96.4% | 88.5% |
| Recompensa media evaluación | **253.3 ± 54.1** | 252.0 ± 60.7 | 215.7 ± 97.7 |
| Pasos medios evaluación | 45.3 | **45.1** | 51.1 |
| Estados visitados en Q-table | 10 120 | 10 727 | 12 112 |

### Búsqueda de hiperparámetros (12 experimentos x 500k episodios)

Se varió γ ∈ {0.95, 0.97, 0.99} y ε_min ∈ {0.01, 0.05, 0.10} para los tres algoritmos. La configuración óptima encontrada es **γ=0.97 con ε_min=0.05** para Q-Learning y SARSA, mientras que Monte Carlo se beneficia de mayor exploración residual.

### Análisis de ablación — eliminación del maelstrom

Se reentrenó Q-Learning con el mapa sin maelstrom para validar que este elemento del diseño añade dificultad genuina. La mejora de +0.4pp en tasa de victoria y la reducción de pasos medios (45.3 -> 44.8) confirman que el agente había aprendido a evitar el maelstrom: el diseño añade complejidad sin hacer el problema irresoluble.

---

## Cómo ejecutar el notebook

### Requisitos

```bash
pip install gymnasium numpy matplotlib pygame pillow
```

### Orden de ejecución

Ejecutar las celdas de definición (constantes, clases, funciones) en orden secuencial. Después:

**Si quieres reentrenar desde cero** (~30 min):
```
Celda de entrenamiento -> entrena 300k x 3 algoritmos, guarda en resultados_entrenamiento.pkl
```

**Si ya tienes el `.pkl` en disco**:
```
Celda de carga → carga desde resultados_entrenamiento.pkl
```

Luego ejecutar el resto en orden: gráficas → render pygame → búsqueda HP → ablación.

> Si reinicias el kernel, ejecuta siempre las celdas de definición antes de la carga. Pickle necesita las clases en memoria para deserializar.

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

---

## Referencias

1. Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2ª ed.). MIT Press.
2. Watkins, C.J.C.H. & Dayan, P. (1992). Q-learning. *Machine Learning, 8*(3–4), 279–292.
3. Rummery, G.A. & Niranjan, M. (1994). *On-line Q-learning using connectionist systems.* Technical Report, Cambridge University.
4. Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature, 518*, 529–533.
5. Hart, P.E., Nilsson, N.J. & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics.*
6. Bellman, R. (1957). *Dynamic Programming.* Princeton University Press.

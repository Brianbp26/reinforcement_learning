# 🏴‍☠️ Barco Pirata — Aprendizaje por Refuerzo

**Asignatura:** Aprendizaje Automático II — Bloque RL  
**Grado:** Ciencia de Datos e Inteligencia Artificial · ETSISI UPM  
**Autores:** Brian Bedoya Piedrahita · Daniel Moraleda Sánchez · David Santiago Ruiz

---

## ¿De qué va el proyecto?

Implementamos un entorno de navegación personalizado usando la API de Gymnasium donde un agente controla un barco pirata en un mapa 8×10. El objetivo: recoger los cuatro cofres del mapa (de distintos valores) y regresar al puerto antes de agotar las provisiones, adaptando la ruta a un viento que cambia de dirección durante el episodio.

Lo que hace este entorno más interesante que los clásicos (FrozenLake, Taxi, CartPole) es que combina cinco fuentes de complejidad simultáneas:

- **Viento variable en el estado** — la misma acción tiene distribución de transición distinta según sople el viento. Sin `dir_viento` en el estado se viola la propiedad de Markov, así que el agente aprende a explotar viento de cola y evitar rutas contra el viento.
- **4 cofres con valores distintos** (+20, +35, +50, +70) — existen 4! = 24 órdenes posibles de recogida. La política óptima pondera riesgo y recompensa: ¿merece la pena ir a por el legendario con pocas provisiones y viento en contra?
- **Provisiones discretizadas en 4 niveles** — la política cambia cualitativamente entre niveles: con provisiones CRÍTICAS el agente debe ir al puerto ignorando cofres cercanos.
- **Corrientes marinas** — desplazamiento determinista que el agente aprende a explotar como atajo o evitar según el contexto.
- **Maelstrom** — atrapa 2 turnos con penalización fija; el agente aprende una zona de evitación absoluta incluso cuando el camino más corto pasa por él.

El espacio de estados es `8 × 10 × 2⁴ × 4 × 4 = 20.480 estados` — manejable con métodos tabulares.

---

## Estructura del notebook

El notebook sigue exactamente las secciones que pide el enunciado. Cada sección alterna celdas markdown (análisis) con celdas de código (implementación):

| Sección | Contenido |
|---|---|
| **2.1 Motivación** | Por qué este problema es interesante para RL y no resoluble con A* |
| **2.2 Formalización MDP** | Definición completa de S, A, R, T, γ con justificación matemática |
| **2.3 Estado del arte** | Contexto en la literatura: SSP, Windy Gridworld, Cliff Walking, DQN |
| **2.4 Soluciones y experimentos** | Implementación + entrenamiento + resultados + gráficas |
| **2.5 Conclusiones** | Análisis crítico de resultados, limitaciones y trabajo futuro |

Dentro de **2.4**, el desarrollo es este:

```
Constantes y mapa         ← tipos de celda, probabilidades de viento, mapa 8×10
PirateEnv                 ← entorno Gymnasium con step(), reset(), render()
QLearningAgent            ← off-policy, actualización por paso con max Q(s')
SARSAAgent                ← on-policy, actualización con Q(s', A') real
MonteCarloAgent           ← every-visit, actualización al final del episodio
entrenar() + evaluar()    ← bucle genérico compatible con los tres agentes
Entrenamiento (500k ep)   ← ~40 min total, resultados guardados en .pkl
Resultados                ← tabla resumen con métricas de evaluación
Gráficas                  ← convergencia, victorias, epsilon, política, valor V(s)
Visualización             ← episodio paso a paso con matplotlib
```

---

## Por qué estos tres algoritmos

El enunciado pide un mínimo de dos algoritmos. Elegimos tres porque cubren los tres enfoques distintos de aprendizaje tabular y permiten un contraste experimental completo:

**Q-Learning (off-policy, TD)** es el algoritmo de referencia del bloque. Al usar `max Q(s', a')` en la actualización, aprende la política óptima π* independientemente de cómo explore — puede observar acciones exploratorias sin que contaminen la política que está aprendiendo. Es el punto de comparación base.

**SARSA (on-policy, TD)** es el contraste natural de Q-Learning. Usa `Q(s', A')` — la acción *real* que tomará, incluyendo errores de exploración — lo que genera una política más conservadora. Este contraste on-policy vs off-policy es exactamente el experimento del **Cliff Walking** de Sutton & Barto (cap. 6.7), que los apuntes de la asignatura trabajan directamente. Las tormentas severas y el maelstrom juegan aquí el papel del acantilado.

**Monte Carlo (every-visit, sin bootstrapping)** añade una tercera dimensión de comparación: en lugar de actualizar en cada paso (como los dos anteriores), actualiza al final del episodio con el retorno real observado. Esto elimina el sesgo de bootstrapping a costa de mayor varianza y convergencia más lenta. Es el punto de referencia para entender por qué los métodos TD son más eficientes en la práctica.

Los tres algoritmos comparten la misma interfaz (`elegir_accion`, `actualizar`, `decay`) y el mismo bucle de entrenamiento genérico — las diferencias están aisladas en cada clase, lo que hace el código comparativo limpio.

---

## Resultados obtenidos

Entrenamiento de 500.000 episodios por algoritmo (~13 min cada uno en CPU):

| Algoritmo | Tasa victoria (eval ε=0) | R media evaluación | Pasos medios |
|---|---|---|---|
| Q-Learning | **97.0%** | 254.2 ± 55.9 | 45.3 |
| SARSA | 96.2% | 253.0 ± 62.4 | 45.0 |
| Monte Carlo | 88.3% | 219.5 ± 99.1 | 50.1 |

Q-Learning y SARSA convergen a políticas casi equivalentes — la diferencia teórica on/off-policy se difumina cuando ε decae a 0.05 (episodio ~100k). Monte Carlo converge más lento y con mayor varianza, como predice la teoría.

---

## Cómo ejecutar el notebook

### Requisitos

```bash
pip install gymnasium numpy matplotlib pygame
```

### Orden de ejecución

```
Celda 9  → imports y constantes
Celda 11 → PirateEnv
Celda 13 → QLearningAgent
Celda 15 → SARSAAgent
Celda 17 → MonteCarloAgent
Celda 19 → funciones entrenar() y evaluar()
```

Después, dos opciones:

**Si quieres reentrenar desde cero** (~40 min):
```
Celda 21 → entrenamiento (500k × 3 algoritmos)
```

**Si ya tienes el `.pkl` en disco** (o te lo pasan los compañeros):
```
Celda 23 → carga desde resultados_entrenamiento.pkl
```

Luego ejecuta las celdas de gráficas y visualización en orden.

> ⚠️ Si reinicias el kernel, ejecuta siempre las celdas de definición (9–19) antes de la celda de carga. Pickle necesita las clases en memoria para deserializar.

### Si el entrenamiento tarda demasiado

```python
# En celda 21, cambia:
N_EPISODIOS = 200_000   # ~15 min total — calidad algo menor
```

---

## Cobertura del enunciado

| Requisito del enunciado | Dónde está en el notebook |
|---|---|
| Entorno propio o adaptado a Gymnasium API | `PirateEnv` (celdas 8–11) |
| Formalización MDP (S, A, R, T, γ) | Sección 2.2 |
| Mínimo 2 algoritmos de RL | Q-Learning + SARSA + Monte Carlo (celdas 12–17) |
| Contraste on-policy vs off-policy | SARSA vs Q-Learning — analizado en 2.4 y 2.5 |
| Tabla de hiperparámetros | Sección 2.4, inicio |
| Gráficas de convergencia | Celdas 28–31 |
| Función de valor y mapa de política | Celdas 36–39 |
| Estado del arte con ≥3 referencias | Sección 2.3 (6 referencias) |
| Conclusiones con análisis crítico | Sección 2.5 |

---

## Referencias

1. Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2ª ed.). MIT Press.
2. Watkins, C.J.C.H. & Dayan, P. (1992). Q-learning. *Machine Learning, 8*(3–4), 279–292.
3. Rummery, G.A. & Niranjan, M. (1994). *On-line Q-learning using connectionist systems.* Technical Report, Cambridge University.
4. Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature, 518*, 529–533.
5. Hart, P.E., Nilsson, N.J. & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics.*
6. Bellman, R. (1957). *Dynamic Programming.* Princeton University Press.
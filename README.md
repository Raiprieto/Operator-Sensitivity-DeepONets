# Operator-Sensitivity-DeepONets

Framework de aprendizaje de operadores neuronales con cuantificación analítica de incertidumbre (UQ) y análisis de sensibilidad física mediante **Jacobian-DeepONet**.

Este repositorio contiene la suite de código reproducible para el estudio de sensibilidad e incertidumbre en DeepONets para ecuaciones diferenciales parciales (EDPs), incluyendo:
- **Ecuación de Burgers 1D** (frentes de choque advectivos y amortiguamiento viscoso)
- **Flujo de Darcy 2D** (medios porosos con campos aleatorios gaussianos de permeabilidad)
- **Navier-Stokes Compresible 2D** (operador flow-map sobre dinámicas hiperbólicas)
- **Ecuación Biarmónica 2D** (operador elíptico de cuarto orden verificado por linealidad)

---

## 1. Fundamentos Metodológicos

1. **Prior físico espacial via Jacobiano:** En lugar de entrenar cabezas de incertidumbre con miles de parámetros o recurrir a costosos muestreos de Monte Carlo, se explota la norma del Jacobiano de la red como prior topológico de la incertidumbre.
2. **Factorización Latente O(K):** Para DeepONets Cartesianas, la Trunk Net procesa exclusivamente coordenadas espaciales y se aísla de las entradas de la Branch Net. La varianza puntual exacta se calcula mediante la matriz de Gram latente, reduciendo la complejidad computacional y de memoria de forma drástica.
3. **Calibración Asimétrica con Pinball Loss:** Se optimizan dos escalares independientes (w_lower, w_upper) bajo la función de pérdida Pinball al 90% (alpha = 0.10). Esta formulación actúa como una regularización Sobolev implícita: las derivadas de la red se alinean con la sensibilidad analítica del sistema sin requerir etiquetas de derivadas durante el entrenamiento.
4. **Rescalado Split-Conformal:** Procedimiento post-hoc riguroso sobre conjuntos de calibración independientes para comparar anchos de intervalo bajo niveles idénticos de cobertura empírica.

---

## 2. Estructura del Repositorio

- **`protocol/`**: Motor estricto de generación, verificación de solapamiento (anti-fuga), entrenamiento, evaluación y agregación reproducible. Incluye configuraciones JSON por EDP y scripts Slurm para clúster HPC.
- **`benchmarks/`**: Suite de evaluación experimental:
  - `posthoc_jacobian.py`: Cuantificación de incertidumbre post-hoc sobre operadores deterministas congelados.
  - `input_noise_propagation.py`: Propagación analítica de perturbaciones de entrada (Método Delta) frente a solvers físicos.
  - `mc_benchmark.py`: Benchmark estocástico de Monte Carlo multiescala.
  - `benchmark_jacobian_latency.py`: Protocolo estandarizado de medición de latencia en GPU/CPU.
  - `fair_eval.py`: Comparativa uniforme de modelos y rescalado conformal.
- **`deepxde-extensions/`**: Implementación de arquitecturas (`JacobianDeepONet`, `VanillaPinballDeepONet`, `QuantileDeepONet`, formulación latente y pérdidas asimétricas).
- **`src/data_generation/`**: Solvers deterministas libres de fuga y generadores de datos con control de semillas y distribuciones fuera de dominio (OOD).
- **`src/model_training/`**: Rutinas de entrenamiento procedural y serialización de modelos.

---

## 3. Instalación

Se recomienda utilizar un entorno virtual con Python 3.9 o superior:

```bash
python -m venv env_tesis
source env_tesis/bin/activate  # En Windows: env_tesis\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Uso Rápido

### Generación de Datos
Para generar conjuntos de datos certificados sin solapamiento:
```bash
python src/data_generation/darcy_pdebench.py --num_samples 1000 --N 64 --output_path data/darcy_test.h5 --seed_offset 1000000
```

### Entrenamiento
Para entrenar una Jacobian-DeepONet con calibración asimétrica:
```bash
python src/model_training/deeponet_training.py \
    --data_path data/darcy_train.h5 \
    --output_path modelos/darcy_jacobian \
    --use_cartesian_prod \
    --use_jacobian \
    --variance_activation softplus \
    --pinball_lambda 4.0 \
    --iterations 200000 \
    --batch_size 128
```

### Ejecución en Clúster (Slurm)
El protocolo automatizado encadena las etapas de generación, verificación, entrenamiento array, evaluación y agregación con dependencias afterok:
```bash
bash protocol/slurm/launch.sh darcy_small --smoke  # Prueba de humo
bash protocol/slurm/launch.sh darcy_small          # Barrido completo
```

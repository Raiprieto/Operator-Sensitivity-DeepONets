# Protocolo de entrenamiento y evaluación

Procedimiento único para entrenar y evaluar los cuatro benchmarks (Burgers,
Biharmonic, Darcy, NS2D). Reemplaza a los `scripts/*/..._sweep.sh`, que ya no
reproducían los modelos del paper y usaban tests contaminados.

**Principio:** el protocolo no advierte y sigue; **aborta** (código de salida 3)
ante cualquier violación. Si una etapa falla, las siguientes no se ejecutan.

## Uso

Desde la raíz del repo, en el nodo de login y con el código commiteado:

```bash
bash protocol/slurm/launch.sh ns2d --smoke   # prueba de humo: minutos
bash protocol/slurm/launch.sh ns2d           # barrido completo
```

Encadena, con dependencias `afterok`:

```
gen train ┐
gen val   ├─> verify ─> train (job array, 1 tarea por corrida) ─> eval ─> aggregate
gen test  ┘
```

`--from train|eval|aggregate` retoma desde una etapa; `--after JOBID` encadena la
primera etapa a un job ya encolado; `--train-gres` y `--eval-gres` cambian la GPU
(default: A100 para entrenar, A30 para evaluar). En A30 (partición MIG de 6 GB)
caben los Jacobian de Darcy y NS2D; los de Burgers y Biharmonic necesitan A100.

## Fuente de verdad

`protocol/configs/<benchmark>.json` define **todo**: generador y semilla de cada
split, tamaños, arquitectura, sigma por modelo, optimizador, validación y
barrido (modelos × λ × semillas). Ningún hiperparámetro se pasa a mano.

### Registro de semillas

| Split | Semilla base | Rango consumido |
|---|---|---|
| test | 1 000 000 | Darcy: `[s, s+n)` · Burgers/NS2D: `[s, s+4n+30000)` |
| val | 2 000 000 | ídem |
| train | 3 000 000 | ídem |

`common.validate_config` calcula el rango exacto que consume cada split y aborta
si dos se cruzan. Biharmonic usa semillas de NumPy para val/test y el archivo de
fair-sciml como train (no regenerable sin el solver FEM).

## Garantías por etapa

| Etapa | Aborta si… |
|---|---|
| todas | no corre dentro de Slurm · hay cambios sin commitear en `src/`, `deepxde-extensions/` o `protocol/` |
| `generate` | el archivo ya existe · el generador falla · escribe una cantidad de muestras distinta a la pedida |
| `verify` | falta un split o tiene otro `n` · grillas o dimensiones distintas · **alguna muestra se repite entre train/val/test** (exacto y a 1e-6) · val/test difieren en distribución del train (KS, p < 1e-4) |
| `train` | falta el manifiesto · cambió la definición de los datos (splits, semillas, dimensión de entrada) después de verificar · el md5 de train/val no coincide · la corrida ya existe · no hay GPU · NaN/∞ en validación · ningún checkpoint válido |
| `evaluate` | el md5 del test no coincide · alguna corrida del barrido no está completa o usó otros datos · la predicción cambia con el tamaño de batch · ya existe el resultado |
| `aggregate` | falta cualquier corrida o evaluación del barrido |

El entrenamiento **nunca lee el test**.

## Decisiones de diseño

1. **Validación y selección de checkpoint.** Cada `val_every` iteraciones se
   evalúa en validación y se guarda el checkpoint con menor *interval score*
   (Gneiting & Raftery, 2007). Es una regla de puntuación propia para
   intervalos: no se puede mejorar solo ensanchando (sube el ancho) ni solo
   angostando (suben las penalizaciones por no cubrir).
2. **Selección de λ solo con validación.** `aggregate` marca, por modelo, el λ de
   menor interval score medio en validación. El test no interviene en ninguna
   decisión.
3. **Todas las semillas o nada.** La tabla reporta media ± desviación estándar
   sobre todas las semillas declaradas. No se puede reportar un subconjunto.
4. **Intervalo determinista por muestra (cambio respecto del paper).** El modelo
   Jacobian normaliza la forma de la incertidumbre por la media del batch
   (paper, línea 244). Medido en los modelos originales: el ancho del intervalo
   de una misma muestra cambia en promedio 19–29 % (hasta ×4.9) según el
   tamaño del batch de inferencia, y la PICP de NS2D va de 49 % a 57 %. El
   entrenamiento no cambia, pero al terminar se fija una referencia: la media de
   la incertidumbre sobre todo el train (análogo a BatchNorm en evaluación).
   `evaluate` comprueba que la predicción no depende del batch y además reporta
   el modo original (batch = 50) solo como comparación. **Requiere actualizar la
   definición de σ̄ en el paper.**
5. **Sigma ligado al modelo.** La evaluación toma sigma, arquitectura, scalers y
   referencia de la metadata de cada corrida. Evaluar con un sigma distinto al del
   entrenamiento (lo que infló la cobertura del vanilla de NS2D) ya no es posible.
6. **Misma arquitectura para Jacobian y vanilla**, como afirma el paper.

## Salidas

```
data/protocol/<bench>/{train,val,test}.h5   (+ .provenance.json por split)
data/protocol/<bench>/MANIFEST.json         md5, leakage, KS
runs/protocol/<bench>/<modelo>_lam<λ>_seed<s>/
    metadata.json      config, commit, md5, sigma, paso elegido, métricas val, unc_ref, versiones
    best.pt            checkpoint seleccionado por validación
    scalers.npz        normalización (ajustada solo con train)
    val_history.csv    métricas de validación en cada evaluación
    loss_history.csv
    test_metrics.json
runs/protocol/<bench>/summary.{md,csv,json}
```

El modo `--smoke` escribe en `data/protocol_smoke/` y `runs/protocol_smoke/`, con
datos y entrenamientos mínimos, para probar la cadena completa en minutos.

## Usar datos externos (p. ej. los de otro colaborador)

En la config, reemplazar el generador de un split por `{"file": "ruta.h5", "n": N}`.
`verify` aplica las mismas comprobaciones (incluido el leakage) antes de permitir
entrenar con ellos.

## Costo de referencia (A100)

Un modelo Jacobian toma 3–4 h (Biharmonic λ = 20 llegó a 13.5 h); un vanilla,
~0.5 h. El barrido completo (2 modelos × 4 λ × 3 semillas) son ~50 GPU·h por
benchmark.

## Limitaciones conocidas

- **Biharmonic:** todo el dataset es `f = c·g` con `g` fija (rango 1). Aun sin
  leakage, el test es un reescalado de muestras del train: mide interpolación en
  un escalar, no generalización entre funciones de entrada.
- **Burgers:** los datos usan ν ∈ [0.001, 0.1]; el paper dice [0.001, 0.01].

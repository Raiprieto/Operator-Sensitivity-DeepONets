# Modelos, datos y configuraciones del protocolo

**Archivo generado.** Regenerar con `python protocol/index_models.py > protocol/MODELS.md` despues de cada barrido; no editar a mano.

Las rutas son relativas a la raiz del repo: `/nfs_home/rprieto/Tesis-RP` en el cluster (host `hpc`), `repositorio_tesis/` en local. Los `runs/` y los `data/` viven solo en el cluster.

## Datos

Un solo conjunto por benchmark: las variantes de modelo (`data_from`) reusan los mismos archivos, verificados por md5 contra `MANIFEST.json`.

| Benchmark de datos | Split | Ruta | n | Semilla | md5 / overrides |
|---|---|---|---|---|---|
| biharmonic | train | `data/biharmonic_equation.h5` | 3010 | - | `acbded84ae01` |
| biharmonic | val | `data/protocol/biharmonic/val.h5` | 500 | 20260922 | `a87e3bb2929a` |
| biharmonic | test | `data/protocol/biharmonic/test.h5` | 3010 | 20260921 | `86cb0bc819e9` |
| burgers | train | `data/protocol/burgers/train.h5` | 15000 | 3000000 | `314cace0bf17` |
| burgers | val | `data/protocol/burgers/val.h5` | 500 | 2000000 | `a4ba02d5eb41` |
| burgers | test | `data/protocol/burgers/test.h5` | 500 | 1000000 | `bcc55830ac24` |
| darcy | train | `data/protocol/darcy/train.h5` | 10000 | 3000000 | `18d05615a99e` |
| darcy | val | `data/protocol/darcy/val.h5` | 500 | 2000000 | `a5630c9340d2` |
| darcy | test | `data/protocol/darcy/test.h5` | 1000 | 1000000 | `06c882677a5b` |
| ns2d | train | `data/protocol/ns2d/train.h5` | 10000 | 3000000 | `aac3657f3289` |
| ns2d | val | `data/protocol/ns2d/val.h5` | 500 | 2000000 | `b61beae952ac` |
| ns2d | test | `data/protocol/ns2d/test.h5` | 500 | 1000000 | `7d061a143dee` |
| ns2d_n20k | train | `data/protocol/ns2d_n20k/train.h5` | 20000 | 4000000 | `3b3d7c812a90` |
| ns2d_n20k | val | `data/protocol/ns2d_n20k/val.h5` | 500 | 2000000 | `b61beae952ac` |
| ns2d_n20k | test | `data/protocol/ns2d_n20k/test.h5` | 500 | 1000000 | `7d061a143dee` |
| ns2d_n20k | ood:far_params | `data/protocol/ns2d_n20k/ood_far_params.h5` | 500 | 7000000 | mu_min=0.0001, mu_max=0.0005, gamma_min=2.2, gamma_max=2.6 |
| ns2d_n20k | ood:gamma_high | `data/protocol/ns2d_n20k/ood_gamma_high.h5` | 500 | 6000000 | gamma_min=2.0, gamma_max=2.2 |
| ns2d_n20k | ood:mu_low | `data/protocol/ns2d_n20k/ood_mu_low.h5` | 500 | 5000000 | mu_min=0.0005, mu_max=0.001 |
| ns2d_n20k | ood:shape_amp2 | `data/protocol/ns2d_n20k/ood_shape_amp2.h5` | 500 | 8000000 | amp_scale=2.0 |

## Modelos entrenados

Cada corrida es una carpeta con `best.pt` (checkpoint elegido por validacion), `scalers.npz`, `metadata.json`, `val_history.csv`, `loss_history.csv` y, si ya fue evaluada, `test_metrics.json` / `test_correlations.json` / `ood_<nombre>.json`.

Una corrida marcada *(running)* sin job vivo en `squeue` fue interrumpida (cancelada o caida): su carpeta esta incompleta y no se debe usar.

### `burgers` - branch 129 -> 10x256, datos de `burgers`

Ruta base: `runs/protocol/burgers/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jacobian_lam4_seed0 *(running)*` | jacobian | 4 | 0 | - | -/- | - | - | - | - | - | - | - |
| `jacobian_lam4_seed1 *(running)*` | jacobian | 4 | 1 | - | -/- | - | - | - | - | - | - | - |

### `darcy_small` - branch 1024 -> 5x100, entrada 32x32, datos de `darcy`

Ruta base: `runs/protocol/darcy_small/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jacobian_lam4_seed0` | jacobian | 4 | 0 | 183,603 | 130000/170000 | 2.782e-06 | 84.08% | 0.0029 | 0.0062 | 0.603 | - | - |
| `jacobian_lam4_seed1` | jacobian | 4 | 1 | 183,603 | 195000/200000 | 3.548e-06 | 82.47% | 0.0027 | 0.0065 | 0.614 | - | - |
| `jacobian_lam4_seed2` | jacobian | 4 | 2 | 183,603 | 190000/200000 | 2.783e-06 | 83.12% | 0.0026 | 0.0060 | 0.624 | - | - |

### `darcy_small_baselines` - branch 1024 -> 5x100, entrada 32x32, datos de `darcy`

Ruta base: `runs/protocol/darcy_small_baselines/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `quantile_ux_lam4_seed0` | quantile_ux | 4 | 0 | 203,803 | 120000/160000 | 2.843e-06 | 81.91% | 0.0027 | 0.0057 | 0.657 | - | - |
| `quantile_ux_lam4_seed1` | quantile_ux | 4 | 1 | 203,803 | 200000/200000 | 2.730e-06 | 81.29% | 0.0024 | 0.0056 | 0.661 | - | - |
| `quantile_ux_lam4_seed2` | quantile_ux | 4 | 2 | 203,803 | 195000/200000 | 2.632e-06 | 81.33% | 0.0024 | 0.0055 | 0.663 | - | - |
| `quantile_x_lam4_seed0` | quantile_x | 4 | 0 | 183,803 | 160000/200000 | 2.655e-06 | 81.50% | 0.0028 | 0.0075 | 0.437 | - | - |
| `quantile_x_lam4_seed1` | quantile_x | 4 | 1 | 183,803 | 200000/200000 | 3.009e-06 | 81.36% | 0.0028 | 0.0080 | 0.436 | - | - |
| `quantile_x_lam4_seed2` | quantile_x | 4 | 2 | 183,803 | 125000/165000 | 3.216e-06 | 82.03% | 0.0030 | 0.0080 | 0.437 | - | - |
| `vanilla_lam4_seed0` | vanilla | 4 | 0 | 183,603 | 165000/200000 | 2.619e-06 | 81.19% | 0.0027 | 0.0076 | 0.428 | - | - |
| `vanilla_lam4_seed1` | vanilla | 4 | 1 | 183,603 | 185000/200000 | 2.912e-06 | 81.03% | 0.0027 | 0.0079 | 0.430 | - | - |
| `vanilla_lam4_seed2` | vanilla | 4 | 2 | 183,603 | 200000/200000 | 2.948e-06 | 80.88% | 0.0027 | 0.0079 | 0.431 | - | - |

### `darcy_small_mse` - branch 1024 -> 5x100, entrada 32x32, datos de `darcy`

Ruta base: `runs/protocol/darcy_small_mse/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mse_lam0_seed0` | mse | 0 | 0 | - | 190000/200000 | 3.088e-06 | 0.00% | 0.0000 | 0.0180 | - | - | - |
| `mse_lam0_seed1` | mse | 0 | 1 | - | 185000/200000 | 3.062e-06 | 0.00% | 0.0000 | 0.0182 | - | - | - |
| `mse_lam0_seed2` | mse | 0 | 2 | - | 200000/200000 | 2.889e-06 | 0.00% | 0.0000 | 0.0180 | - | - | - |

### `ns2d` - branch 4098 -> 10x256, datos de `ns2d`

Ruta base: `runs/protocol/ns2d/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jacobian_lam4_seed0` | jacobian | 4 | 0 | 2,234,371 | 20000/200000 | 1.666e-03 | 79.39% | 0.0839 | 0.1693 | 0.456 | 0.503 | - |
| `jacobian_lam4_seed1` | jacobian | 4 | 1 | 2,234,371 | 20000/200000 | 1.725e-03 | 79.66% | 0.0860 | 0.1780 | 0.419 | 0.502 | - |
| `jacobian_lam4_seed2` | jacobian | 4 | 2 | 2,234,371 | 20000/200000 | 1.876e-03 | 77.40% | 0.0852 | 0.1880 | 0.439 | 0.480 | - |

### `ns2d_exp_s32` - branch 1026 -> 10x256, entrada 32x32, datos de `ns2d`

Ruta base: `runs/protocol/ns2d_exp_s32/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jacobian_lam4_seed0` | jacobian | 4 | 0 | - | 110000/150000 | 9.083e-03 | 90.12% | 0.3026 | 0.3984 | - | - | - |

### `ns2d_exp_small` - branch 4098 -> 5x100, datos de `ns2d`

Ruta base: `runs/protocol/ns2d_exp_small/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jacobian_lam4_seed0` | jacobian | 4 | 0 | 491,003 | 110000/150000 | 1.413e-03 | 84.61% | 0.0880 | 0.1382 | 0.478 | 0.526 | - |

### `ns2d_exp_small_quantile` - branch 1026 -> 5x100, entrada 32x32, datos de `ns2d`

Ruta base: `runs/protocol/ns2d_exp_small_quantile/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `quantile_ux_lam4_seed0` | quantile_ux | 4 | 0 | 204,003 | 105000/145000 | 1.288e-03 | 84.63% | 0.0851 | 0.1287 | 0.497 | 0.477 | - |
| `quantile_x_lam4_seed0` | quantile_x | 4 | 0 | 184,003 | 20000/60000 | 1.580e-03 | 86.49% | 0.1046 | 0.1903 | 0.006 | 0.002 | - |

### `ns2d_exp_small_s32` - branch 1026 -> 5x100, entrada 32x32, datos de `ns2d`

Ruta base: `runs/protocol/ns2d_exp_small_s32/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jacobian_lam1_seed0` | jacobian | 1 | 0 | 183,803 | 110000/150000 | 1.323e-03 | 84.56% | 0.0843 | 0.1332 | 0.492 | 0.523 | - |
| `jacobian_lam20_seed0` | jacobian | 20 | 0 | 183,803 | 155000/195000 | 1.258e-03 | 85.63% | 0.0830 | 0.1241 | 0.521 | 0.511 | - |
| `jacobian_lam4_seed0` | jacobian | 4 | 0 | 183,803 | 155000/195000 | 1.265e-03 | 85.35% | 0.0827 | 0.1276 | 0.502 | 0.511 | - |
| `jacobian_lam8_seed0` | jacobian | 8 | 0 | 183,803 | 155000/195000 | 1.273e-03 | 85.32% | 0.0829 | 0.1265 | 0.513 | 0.516 | - |
| `vanilla_lam1_seed0` | vanilla | 1 | 0 | 183,803 | 40000/80000 | 1.424e-03 | 86.46% | 0.0981 | 0.1814 | 0.006 | 0.004 | - |
| `vanilla_lam20_seed0` | vanilla | 20 | 0 | 183,803 | 35000/75000 | 1.536e-03 | 85.98% | 0.1005 | 0.1879 | -0.003 | 0.004 | - |
| `vanilla_lam4_seed0` | vanilla | 4 | 0 | 183,803 | 25000/65000 | 1.477e-03 | 85.82% | 0.0976 | 0.1853 | -0.003 | 0.004 | - |
| `vanilla_lam8_seed0` | vanilla | 8 | 0 | 183,803 | 35000/75000 | 1.499e-03 | 86.07% | 0.0997 | 0.1855 | -0.004 | 0.004 | - |

### `ns2d_n20k` - branch 1026 -> 5x100, entrada 32x32, datos de `ns2d_n20k`

Ruta base: `runs/protocol/ns2d_n20k/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jacobian_lam4_seed0` | jacobian | 4 | 0 | 183,803 | 195000/200000 | 1.067e-03 | 88.03% | 0.0822 | 0.1113 | 0.515 | 0.524 | far_params, gamma_high, mu_low, shape_amp2 |
| `jacobian_lam4_seed1` | jacobian | 4 | 1 | 183,803 | 195000/200000 | 1.068e-03 | 88.36% | 0.0821 | 0.1112 | 0.517 | 0.526 | - |
| `jacobian_lam4_seed2` | jacobian | 4 | 2 | 183,803 | 180000/200000 | 1.094e-03 | 87.98% | 0.0827 | 0.1120 | 0.524 | 0.524 | - |

### `ns2d_n20k_baselines` - branch 1026 -> 5x100, entrada 32x32, datos de `ns2d_n20k`

Ruta base: `runs/protocol/ns2d_n20k_baselines/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `quantile_ux_lam4_seed0` | quantile_ux | 4 | 0 | 204,003 | 200000/200000 | 1.072e-03 | 87.45% | 0.0825 | 0.1115 | 0.517 | 0.477 | far_params, gamma_high, mu_low, shape_amp2 |
| `quantile_ux_lam4_seed1` | quantile_ux | 4 | 1 | 204,003 | 185000/200000 | 1.078e-03 | 87.35% | 0.0826 | 0.1120 | 0.512 | 0.481 | - |
| `quantile_ux_lam4_seed2` | quantile_ux | 4 | 2 | 204,003 | 185000/200000 | 1.092e-03 | 87.62% | 0.0839 | 0.1127 | 0.513 | 0.479 | - |
| `quantile_x_lam4_seed0` | quantile_x | 4 | 0 | 184,003 | 200000/200000 | 1.145e-03 | 87.72% | 0.0910 | 0.1622 | 0.005 | 0.001 | far_params, gamma_high, mu_low, shape_amp2 |
| `quantile_x_lam4_seed1` | quantile_x | 4 | 1 | 184,003 | 175000/200000 | 1.133e-03 | 87.95% | 0.0913 | 0.1608 | 0.002 | 0.002 | - |
| `quantile_x_lam4_seed2` | quantile_x | 4 | 2 | 184,003 | 195000/200000 | 1.142e-03 | 87.85% | 0.0915 | 0.1620 | 0.005 | 0.004 | - |
| `vanilla_lam4_seed0` | vanilla | 4 | 0 | 183,803 | 170000/200000 | 1.162e-03 | 87.73% | 0.0914 | 0.1630 | 0.000 | 0.008 | far_params, gamma_high, mu_low, shape_amp2 |
| `vanilla_lam4_seed1` | vanilla | 4 | 1 | 183,803 | 195000/200000 | 1.142e-03 | 87.85% | 0.0909 | 0.1620 | -0.002 | 0.008 | - |
| `vanilla_lam4_seed2` | vanilla | 4 | 2 | 183,803 | 180000/200000 | 1.132e-03 | 87.85% | 0.0908 | 0.1610 | 0.006 | 0.008 | - |

### `ns2d_n20k_hybrid` - branch 1026 -> 5x100, entrada 32x32, datos de `ns2d_n20k`

Ruta base: `runs/protocol/ns2d_n20k_hybrid/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jacobian_quantile_lam4_seed0` | jacobian_quantile | 4 | 0 | 203,803 | 200000/200000 | 1.082e-03 | 88.56% | 0.0849 | 0.1103 | 0.525 | 0.500 | - |

### `ns2d_n20k_hybrid_c` - branch 1026 -> 5x100, entrada 32x32, datos de `ns2d_n20k`

Ruta base: `runs/protocol/ns2d_n20k_hybrid_c/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jacobian_quantile_c_lam4_seed0` | jacobian_quantile_c | 4 | 0 | - | 200000/200000 | 1.107e-03 | 87.19% | 0.0819 | 0.1129 | - | - | - |

### `ns2d_n20k_hybrid_cx` - branch 1026 -> 5x100, entrada 32x32, datos de `ns2d_n20k`

Ruta base: `runs/protocol/ns2d_n20k_hybrid_cx/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jacobian_quantile_cx_lam4_seed0` | jacobian_quantile_cx | 4 | 0 | - | 185000/200000 | 1.087e-03 | 87.94% | 0.0830 | 0.1137 | - | - | - |

### `ns2d_n20k_hybrid_x` - branch 1026 -> 5x100, entrada 32x32, datos de `ns2d_n20k`

Ruta base: `runs/protocol/ns2d_n20k_hybrid_x/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jacobian_quantile_x_lam4_seed0` | jacobian_quantile_x | 4 | 0 | 184,003 | 195000/200000 | 1.057e-03 | 88.50% | 0.0826 | 0.1102 | 0.519 | 0.529 | - |

### `ns2d_n20k_mse` - branch 1026 -> 5x100, entrada 32x32, datos de `ns2d_n20k`

Ruta base: `runs/protocol/ns2d_n20k_mse/`

| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | IS | rhoErr | rhoSens | OOD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mse_lam0_seed0` | mse | 0 | 0 | - | 175000/200000 | 1.101e-03 | 0.00% | 0.0000 | 0.4398 | - | - | - |
| `mse_lam0_seed1` | mse | 0 | 1 | - | 195000/200000 | 1.085e-03 | 0.00% | 0.0000 | 0.4370 | - | - | - |
| `mse_lam0_seed2` | mse | 0 | 2 | - | 195000/200000 | 1.097e-03 | 0.00% | 0.0000 | 0.4390 | - | - | - |

## Configuraciones

Fuente de verdad: `protocol/configs/<benchmark>.json`. Cada corrida guarda ademas una copia literal de la config con la que se entreno en su `metadata.json`.

| Config | Modelos del barrido | lambda | Semillas | n_train | Arquitectura | sigma | Entrada | Optimizador |
|---|---|---|---|---|---|---|---|---|
| `biharmonic` | jacobian | 4 | 0, 1, 2 | 3010 | 10x256, K=256, swish | jacobian=0.025 | 4225 directo | lr 0.0001, decay 0.8/50000, batch 512, 200000 it, patience None |
| `burgers` | jacobian | 4 | 0, 1, 2 | 15000 | 10x256, K=256, swish | jacobian=0.025 | 129 directo | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience None |
| `darcy` | jacobian, vanilla | 1, 4, 8, 20 | 0, 1, 2 | 10000 | 4x256, K=256, swish | jacobian=0.025, vanilla=0.025 | 4096 directo | lr 0.001, decay 0.9/20000, batch 128, 200000 it, patience None |
| `darcy_small` | jacobian | 4 | 0, 1, 2 | 10000 | 5x100, K=100, swish | jacobian=0.025 | 64x64 stride 2 | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |
| `darcy_small_baselines` | vanilla, quantile_x, quantile_ux | 4 | 0, 1, 2 | 10000 | 5x100, K=100, swish | vanilla=0.025 | 64x64 stride 2 | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |
| `darcy_small_mse` | mse | 0 | 0, 1, 2 | 10000 | 5x100, K=100, swish | - | 64x64 stride 2 | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |
| `ns2d` | jacobian | 4 | 0, 1, 2 | 10000 | 10x256, K=256, swish | jacobian=0.05 | 4098 directo | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience None |
| `ns2d_exp_s32` | jacobian | 4 | 0 | 10000 | 10x256, K=256, swish | jacobian=0.05 | 64x64 stride 2 | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |
| `ns2d_exp_small` | jacobian | 4 | 0 | 10000 | 5x100, K=100, swish | jacobian=0.05 | 4098 directo | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |
| `ns2d_exp_small_quantile` | quantile_x, quantile_ux | 4 | 0 | 10000 | 5x100, K=100, swish | - | 64x64 stride 2 | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |
| `ns2d_exp_small_s32` | jacobian, vanilla | 1, 4, 8, 20 | 0 | 10000 | 5x100, K=100, swish | jacobian=0.05, vanilla=0.025 | 64x64 stride 2 | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |
| `ns2d_n20k` | jacobian | 4 | 0, 1, 2 | 20000 | 5x100, K=100, swish | jacobian=0.05 | 64x64 stride 2 | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |
| `ns2d_n20k_baselines` | vanilla, quantile_x, quantile_ux | 4 | 0, 1, 2 | 20000 | 5x100, K=100, swish | vanilla=0.025 | 64x64 stride 2 | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |
| `ns2d_n20k_hybrid` | jacobian_quantile | 4 | 0 | 20000 | 5x100, K=100, swish | jacobian_quantile=0.05 | 64x64 stride 2 | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |
| `ns2d_n20k_hybrid_c` | jacobian_quantile_c | 4 | 0 | 20000 | 5x100, K=100, swish | jacobian_quantile_c=0.05 | 64x64 stride 2 | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |
| `ns2d_n20k_hybrid_cx` | jacobian_quantile_cx | 4 | 0 | 20000 | 5x100, K=100, swish | jacobian_quantile_cx=0.05 | 64x64 stride 2 | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |
| `ns2d_n20k_hybrid_x` | jacobian_quantile_x | 4 | 0 | 20000 | 5x100, K=100, swish | jacobian_quantile_x=0.05 | 64x64 stride 2 | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |
| `ns2d_n20k_mse` | mse | 0 | 0, 1, 2 | 20000 | 5x100, K=100, swish | - | 64x64 stride 2 | lr 0.001, decay 0.7/50000, batch 128, 200000 it, patience 8 |

Que es cada config:

- **`biharmonic`** - 2D biarmonica (fair-sciml). Entrada: f = c*g en 4225 nodos. Salida: u en 4225 nodos.
- **`burgers`** - 1D viscous Burgers. Entrada: u0 (128) + nu. Salida: u(x,t) en 128x100.
- **`darcy`** - 2D Darcy estacionario. Entrada: permeabilidad K (64x64). Salida: presion (64x64).
- **`darcy_small`** - EXPERIMENTO (Darcy): misma compactacion que en NS2D. Arquitectura chica 5x100 y entrada submuestreada 32x32 (stride 2 sobre la permeabilidad 64x64), en vez del branch 4x256 sobre 4096 entradas. Una semilla y lambda=4 como prueba inicial. Datos de darcy.json.
- **`darcy_small_baselines`** - EXPERIMENTO (Darcy): baselines con la MISMA arquitectura chica, datos, lambda y semilla que el Jacobian de darcy_small. Escalera de ablacion del ancho: constante (vanilla) -> espacial (quantile_x) -> condicionado en la muestra (quantile_ux).
- **`darcy_small_mse`** - EXPERIMENTO (Darcy): DeepONet determinista (MSE puro, sin bandas) con la MISMA arquitectura chica, datos y semillas que darcy_small. Es el modelo base de los benchmarks Monte Carlo y del metodo Delta, donde la incertidumbre se construye despues perturbando la entrada.
- **`ns2d`** - 2D Navier-Stokes compresible (estado en t=0.5). Entrada: rho0 (64x64) + (gamma, mu). Salida: rho (64x64).
- **`ns2d_exp_s32`** - EXPERIMENTO (NS2D, sobreajuste): DeepThin 10x256 (K=256), entrada submuestreada 32x32.
- **`ns2d_exp_small`** - EXPERIMENTO (NS2D, sobreajuste): 5x100 en branch y trunk (K=100), entrada completa 64x64.
- **`ns2d_exp_small_quantile`** - EXPERIMENTO (NS2D): baselines de quantile de la rama fair-quantile-baseline de Paul (quantile_x espacial y quantile_ux condicional) con la misma arquitectura chica (5x100 + entrada 32x32), mismos datos y lambda que ns2d_exp_small_s32.
- **`ns2d_exp_small_s32`** - EXPERIMENTO (NS2D, sobreajuste): 5x100 (K=100) + entrada 32x32.
- **`ns2d_n20k`** - EXPERIMENTO (NS2D): mismo generador y distribucion que ns2d, pero train de 20000 muestras con semilla propia. Val y test usan las mismas semillas que ns2d, asi que son identicos (se verifica por md5). Arquitectura chica 5x100 + entrada 32x32.
- **`ns2d_n20k_baselines`** - EXPERIMENTO (NS2D, train de 20000): baselines con la MISMA arquitectura chica, datos, lambda y semilla que el Jacobian de ns2d_n20k. Escalera de ablacion del ancho: constante -> espacial (quantile_x) -> condicionado en la muestra (quantile_ux) -> Jacobian.
- **`ns2d_n20k_hybrid`** - EXPERIMENTO (NS2D, 20000): hibrido jacobiano + cabezas con capacidad. La forma de la incertidumbre la fija ||J_branch|| y unas cabezas tipo quantile_ux aprenden una correccion multiplicativa, inicializada en 1 (identico al Jacobian en el paso 0).
- **`ns2d_n20k_hybrid_c`** - EXPERIMENTO (NS2D, 20000): hibrido jacobiano + correccion centrada por muestra y punto (~20000 params): solo puede redistribuir el ancho, no escalarlo.
- **`ns2d_n20k_hybrid_cx`** - EXPERIMENTO (NS2D, 20000): hibrido jacobiano + correccion centrada solo espacial (~200 params): repite el experimento que quedo degenerado.
- **`ns2d_n20k_hybrid_x`** - EXPERIMENTO (NS2D, 20000): hibrido con correccion SOLO ESPACIAL. El prior fisico aporta toda la dependencia de la muestra (||J(u)||) y las cabezas solo corrigen un perfil espacial (~200 parametros, dos ordenes de magnitud menos que el hibrido completo).
- **`ns2d_n20k_mse`** - EXPERIMENTO (NS2D, train de 20000): DeepONet determinista (MSE puro, sin bandas) con la MISMA arquitectura chica, datos y semillas que ns2d_n20k. Es el modelo base de los benchmarks Monte Carlo y del metodo Delta, donde la incertidumbre se construye despues perturbando la entrada.

## Modelos anteriores al protocolo

Entrenados con `scripts/*/*_sweep.sh` sobre los datos con leakage de semillas. Se conservan como referencia historica de los resultados del paper; **no usar para resultados nuevos**.

| Carpeta | Checkpoints `.pt` |
|---|---|
| `modelos/biharmonic/` | 10 |
| `modelos/biharmonic_jacobian_form/` | 1 |
| `modelos/burgers/` | 17 |
| `modelos/darcy/` | 8 |
| `modelos/ns2d/` | 8 |
| `modelos/tuning/` | 4 |

## Cargar una corrida

```python
import json, sys, numpy as np, torch; sys.path.insert(0, 'protocol')
import common as C
d = 'runs/protocol/ns2d_n20k/jacobian_lam4_seed0'
m = json.load(open(d + '/metadata.json'))
net = C.build_net(m['architecture'], m['model'], m['sigma']).cuda()
net.load_state_dict(torch.load(d + '/best.pt')['model_state_dict']); net.eval()
net.unc_ref = m['unc_ref']        # referencia fija: el intervalo no depende del batch
sc = dict(np.load(d + '/scalers.npz')); cfg = m['config']
```

La arquitectura, el sigma, los scalers y la transformacion de entrada salen de `metadata.json`, nunca de la linea de comandos: por construccion es imposible evaluar una corrida con hiperparametros distintos a los de su entrenamiento.

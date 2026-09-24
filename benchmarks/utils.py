import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import time
from typing import Tuple, Dict
import sys
import os
import h5py
import argparse



def guardar_historial_csv(losshistory, nombre_archivo):
    """
    Extrae el historial de entrenamiento de DeepXDE y lo guarda en un CSV.
    """
    pasos = np.array(losshistory.steps)
    perdida_train = np.array(losshistory.loss_train)
    
    if perdida_train.ndim > 1:
        perdida_total = np.sum(perdida_train, axis=1)
    else:
        perdida_total = perdida_train
        
    datos = {
        "Iteracion": pasos,
        "Perdida_Train_Total": perdida_total
    }
    
    
    if len(losshistory.metrics_test) > 0:
        metricas = np.array(losshistory.metrics_test)
        if metricas.size > 0 and metricas.ndim > 1 and metricas.shape[1] > 0:
            datos["Error_L2_Relativo"] = metricas[:, 0]
    df = pd.DataFrame(datos)
    df.to_csv(nombre_archivo, index=False)
    print(f"Historial guardado correctamente en: {nombre_archivo}")





#--------------------------------------------------------------------------------   



def get_physical_pred(model, data: Tuple[np.ndarray, np.ndarray], scalers: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Realiza la predicción de un punto y transforma el resultado del espacio normalizado al espacio físico.
    """
    pred_scaled = model.predict(data)
    if isinstance(pred_scaled, tuple) or isinstance(pred_scaled, list):
        pred_scaled = pred_scaled[0]
    return (pred_scaled * scalers['Y_std']) + scalers['Y_mean']


#--------------------------------------------------------------------------------


def montecarlo(
    model, 
    branch_raw: np.ndarray, 
    trunk_raw: np.ndarray, 
    scalers: Dict[str, np.ndarray], 
    param_base: np.ndarray = None,
    param_idx: int = None,
    alpha: float = 0.05, 
    n_mc: int = 1000,
    architecture: str = "cartesian" 
) -> Dict[str, np.ndarray]:
    """
    Función abstracta generalizada de Monte Carlo para validación de Sensibilidad.
    Sirve para modelos de pareo iterativo (pointwise) y CartesianProd.
    
    Args:
        model: Modelo DeepONet en memoria listo para hacer predict().
        branch_raw, trunk_raw: Datos físicos puros de validación.
        scalers: Diccionario con llaves genéricas de std/mean.
        param_base: Vector 1D (Opcional) con la constante que genera las distribuciones. 
        param_idx: Índice entero (Opcional). Si se especifica, se asume que branch_raw es un vector 
                   y solo se perturba ese índice (Ej: sensor de viscosidad).
        alpha: Perturbación en % para el parámetro.
        n_mc: Simulaciones.
        architecture: "cartesian" u "pointwise".
    """
    num_points = branch_raw.shape[0]
    
    # 1. Escalamiento universal
    b_scaled = (branch_raw - scalers['branch_mean']) / scalers['branch_std']
    t_scaled = (trunk_raw - scalers['trunk_mean']) / scalers['trunk_std']
    
    # 2. Inferencia Base (Sin Ruido)
    t_start = time.time()
    # Si la arquitectura es Pointwise y las dimensiones no cuadran, se debe predecir por chunks o forzoso. 
    # El predict asume data tuple standard compatible con DeepXDE.
    p_base_scaled = model.predict((b_scaled, t_scaled))
    p_base_scaled = p_base_scaled[0] if isinstance(p_base_scaled, (tuple, list)) else p_base_scaled
    u_base_phys = (p_base_scaled * scalers['Y_std']) + scalers['Y_mean']
    
    if u_base_phys.shape[-1] > 1 and "jacob" in str(type(model)).lower():
        n_n = u_base_phys.shape[-1] // 3 if u_base_phys.shape[-1] >= 3 else u_base_phys.shape[-1]
        u_base_phys = u_base_phys[:, 0:n_n]
        
    t_inf = (time.time() - t_start) / max(1, num_points)
    
    # 3. Preparación de Arrays para la salida
    sensitivities = np.zeros((num_points, u_base_phys.shape[1]))
    s_lower = np.zeros_like(sensitivities)
    s_upper = np.zeros_like(sensitivities)
    y_lower = np.zeros_like(u_base_phys)
    y_upper = np.zeros_like(u_base_phys)
    
    t_mc_start = time.time()
    
    for i in tqdm(range(num_points), desc=f"Monte Carlo ({architecture})"):
        val_base = branch_raw[i]
        
        # Determinar la base del generador
        if param_idx is not None:
            gen_base = val_base[param_idx]
        elif param_base is not None:
            gen_base = param_base[i]
        else:
            gen_base = val_base
        
        # Muestreo normal
        noise = np.random.normal(0, 1, (n_mc, 1))
        
        # Perturbación. Si gen_base es escalar, gen_pert_array también.
        gen_pert_array = gen_base * (1 + alpha * noise) 
        delta_p = gen_pert_array - gen_base
        
        # Como delta_p podría tener varios ceros (improbable pero posible), fijar epsilon iterativo.
        delta_p = np.where(delta_p == 0, 1e-12, delta_p)
        
        # Propagación de la perturbación al Branch
        if param_idx is not None:
            branch_pert_raw = np.tile(val_base, (n_mc, 1))
            branch_pert_raw[:, param_idx] = gen_pert_array.flatten()
        elif param_base is not None:
            branch_pert_raw = val_base * (1 + alpha * noise)
        else:
            branch_pert_raw = gen_pert_array
        branch_pert_scaled = (branch_pert_raw - scalers['branch_mean']) / scalers['branch_std']
        
        # Ajuste de dimensión del Trunk 
        if architecture == "pointwise":
            trunk_iter = np.tile(t_scaled[i], (n_mc, 1))
        elif architecture == "cartesian":
            trunk_iter = t_scaled # En CartesianProd, la misma cuadrícula X evalúa contra todos los Branch
        else:
            raise ValueError("Arquitectura no sorportada. Usa 'cartesian' o 'pointwise'.")
            
        # Inferencia de iteración
        preds_scaled = model.predict((branch_pert_scaled, trunk_iter))
        preds_scaled = preds_scaled[0] if isinstance(preds_scaled, (tuple, list)) else preds_scaled
        preds_phys = (preds_scaled * scalers['Y_std']) + scalers['Y_mean']
        
        if preds_phys.shape[-1] > 1 and "jacob" in str(type(model)).lower():
            n_n = preds_phys.shape[-1] // 3 if preds_phys.shape[-1] >= 3 else preds_phys.shape[-1]
            preds_phys = preds_phys[:, 0:n_n]
            
        # Obtención de Percentiles (Cobertura Espacial u Outliers)
        y_lower[i] = np.percentile(preds_phys, 5, axis=0).flatten()
        y_upper[i] = np.percentile(preds_phys, 95, axis=0).flatten()
        
        # Cálculo de Sensibilidad Clásica MC: dt = ΔU / ΔParámetro Funcional
        # Importante: delta_p en biarmónico es escalar shape(n_mc, 1). Se broadcasteará en u_base_phys.
        samples_sens = (abstract_flatten(preds_phys, u_base_phys.shape[1], n_mc) - u_base_phys[i].flatten()) / delta_p
        sensitivities[i] = np.mean(samples_sens, axis=0)
        s_lower[i] = np.percentile(samples_sens, 2.5, axis=0)
        s_upper[i] = np.percentile(samples_sens, 97.5, axis=0)

    t_mc_end = time.time()
    
    return {
        "y_base": u_base_phys,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "sensitivities": sensitivities,
        "s_lower": s_lower,
        "s_upper": s_upper,
        "time_per_point": (t_mc_end - t_mc_start) / max(1, num_points),
        "time_inference_only": t_inf
    }



#------------------------------------------------------------------------

def abstract_flatten(pred_array, expected_size, n_mc):
    """ Función auxiliar para evitar fallos de shape originados en DeepXDE Tuple Outputs """
    if pred_array.shape == (n_mc, expected_size):
        return pred_array
    return pred_array.reshape(n_mc, expected_size)




#------------------------------------------------------------------------

def eval_jacobian(
    model_pytorch, 
    branch_raw: np.ndarray, 
    trunk_raw: np.ndarray, 
    scalers_j: Dict[str, np.ndarray], 
    param_base: np.ndarray = None,
    param_idx: int = None,
    architecture: str = "cartesian",
    device: str = "cpu"
) -> Dict[str, np.ndarray]:
    """
    Evalúa la sensibilidad analítica espacial exacta de un JacobianDeepONet (PyTorch puro).
    Utiliza el Jacobian-Vector Product (Push-Forward) para evitar el colapso dimensional 
    de `autograd.grad` masivo en mallas cartesianas.
    """
    import torch
    
    num_points = branch_raw.shape[0]
    
    # 1. Normalización
    jb_scaled = (branch_raw - scalers_j['branch_mean']) / scalers_j['branch_std']
    jt_scaled = (trunk_raw - scalers_j['trunk_mean']) / scalers_j['trunk_std']
    
    sensitivities = []
    y_center = []
    y_lower = []
    y_upper = []
    
    t_start = time.time()

    # Precómputo de constantes invariantes al bucle:
    # BaseClass y jtt (trunk cartesiano) no cambian entre iteraciones.
    # Definir func_to_jvp UNA SOLA VEZ permite que torch.autograd.functional.jvp
    # reuse el grafo trazado en la primera llamada, eliminando el overhead de
    # re-tracing en cada iteración.
    if architecture == "cartesian":
        jtt_const = torch.tensor(jt_scaled, dtype=torch.float32).to(device)
    
    BaseClass = model_pytorch.__class__.__bases__[0]

    def func_to_jvp(b):
        # Para la JVP solo necesitamos la predicción central (sin bandas de incertidumbre).
        # Llamamos al forward de DeepONetCartesianProd directamente para evitar
        # que el Jacobiano interno de JacobianDeepONet se evalue O(K) veces extra.
        return BaseClass.forward(model_pytorch, (b, jtt_const))

    for i in tqdm(range(num_points), desc=f"Eval Analytic Jacob ({architecture})"):
        vb_scaled_i = jb_scaled[i:i+1] # Shape (1, branch_dim)
        val_base = branch_raw[i]       # Shape (branch_dim)
        
        if param_idx is not None:
            gen_base = val_base[param_idx]
        elif param_base is not None:
            gen_base = param_base[i]
        else:
            gen_base = val_base
            
        # Arquitectura del trunk
        if architecture == "pointwise":
            jtt_const = torch.tensor(jt_scaled[i:i+1], dtype=torch.float32).to(device)
        # En cartesian, jtt_const ya está definido arriba y no cambia.
            
        jtb = torch.tensor(vb_scaled_i, dtype=torch.float32, requires_grad=True).to(device)
        
        # Inferencia Central Analítica
        pred_j_scaled = model_pytorch((jtb, jtt_const))
        pred_j_np = pred_j_scaled.detach().cpu().numpy()
        
        if pred_j_np.shape[-1] == scalers_j['Y_std'].shape[-1] * 3:
            y_std_ext = np.tile(scalers_j['Y_std'], 3)
            y_mean_ext = np.tile(scalers_j['Y_mean'], 3)
            pred_j_phys = (pred_j_np * y_std_ext) + y_mean_ext
        else:
            pred_j_phys = (pred_j_np * scalers_j['Y_std']) + scalers_j['Y_mean']
            
        num_nodes_j = pred_j_phys.shape[-1] // 3 if pred_j_phys.shape[-1] >= 3 else pred_j_phys.shape[-1]
            
        y_center.append(pred_j_phys[:, 0:num_nodes_j].flatten())
        if pred_j_phys.shape[-1] >= 3:
            y_lower.append(pred_j_phys[:, num_nodes_j:2*num_nodes_j].flatten())
            y_upper.append(pred_j_phys[:, 2*num_nodes_j:].flatten())

        # Push-forward del Jacobiano
        if param_idx is not None:
            v_dir_raw = np.zeros_like(val_base)
            v_dir_raw[param_idx] = 1.0
        elif param_base is not None:
            c_val = gen_base if gen_base != 0 else 1e-12
            v_dir_raw = val_base / c_val
        else:
            v_dir_raw = np.ones_like(val_base)
            
        v_dir_scaled = v_dir_raw / scalers_j['branch_std']
        v_tensor = torch.tensor(v_dir_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Ejecución JVP PyTorch (reutiliza el grafo de func_to_jvp compilado en i=0)
        _, dy_scaled = torch.autograd.functional.jvp(func_to_jvp, (jtb,), (v_tensor,))
        
        # Llevando la diferencial computada al mundo físico
        s_jacob_phys = dy_scaled.detach().cpu().numpy() * scalers_j['Y_std']
        sensitivities.append(s_jacob_phys.flatten())

    t_end = time.time()
    
    has_bounds = len(y_lower) > 0
    
    return {
        "y_base": np.array(y_center),
        "y_lower": np.array(y_lower) if has_bounds else None,
        "y_upper": np.array(y_upper) if has_bounds else None,
        "sensitivities": np.array(sensitivities),
        "time_per_point": (t_end - t_start) / max(1, num_points)
    }





#------------------------------------------------------------------------------

def evaluar_malla_completa(ruta_datos, ruta_modelo, ruta_scalers, args_modelo):
    """
    Evalúa un modelo DeepONet entrenado sobre la malla física completa (11,881 nodos)
    usando un submuestreo fijo para la entrada de la Branch Network.
    
    Args:
        ruta_datos (str): Ruta al archivo .h5 con los datos generados por FEniCS.
        ruta_modelo (str): Ruta exacta al checkpoint del modelo entrenado (.pt).
        ruta_scalers (str): Ruta al archivo .npz con los parámetros de normalización Z-score.
        args_modelo (argparse.Namespace): Configuración de arquitectura idéntica a la usada en entrenamiento.
        
    Returns:
        tuple: (y_true_full, y_pred_real, error_l2) conteniendo el ground truth,
               la predicción en escala física y el error relativo L2.
    """
    # 1. Recuperar índices fijos del entrenamiento para la Branch Network
    # Asegurar acceso a las funciones procedurales del pipeline de entrenamiento dinámicamente
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'model_training')))
    from deeponet_training import crear_red
    
    np.random.seed(42)
    num_sensors = args_modelo.num_branch_sensors
    total_nodes = 11881
    indices_fijos = np.random.choice(total_nodes, num_sensors, replace=False)
    indices_fijos.sort()
    # 2. Extraer tensores de la primera simulación
    with h5py.File(ruta_datos, 'r') as f:
        session_key = list(f.keys())[0]
        sim_keys = list(f[session_key].keys())
        sim = f[session_key][sim_keys[0]]
        
        trunk_inputs_full = sim['coordinates'][:, :2]
        y_true_full = sim['values'][:]
        branch_inputs_full = sim['field_input_f'][:]
        branch_inputs_sub = branch_inputs_full[indices_fijos].reshape(1, -1)
    # 3. Normalización basada en la distribución del entrenamiento original
    scalers = np.load(ruta_scalers)
    branch_inputs_norm = (branch_inputs_sub - scalers['branch_mean']) / scalers['branch_std']
    trunk_inputs_norm = (trunk_inputs_full - scalers['trunk_mean']) / scalers['trunk_std']
    # 4. Reconstrucción instanciada de la red
    net = crear_red(args_modelo)
    
    # 5. Contenedor Data "Dummy" (requerido por DeepXDE para instanciar Model en fase de inferencia)
    dummy_y = np.zeros((1, trunk_inputs_full.shape[0]))
    data = dde.data.TripleCartesianProd(
        X_train=(branch_inputs_norm, trunk_inputs_norm), y_train=dummy_y,
        X_test=(branch_inputs_norm, trunk_inputs_norm), y_test=dummy_y
    )
    
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    model.restore(ruta_modelo, verbose=1, device="cpu")
    # 6. Predicción y des-normalización
    y_pred_norm = model.predict((branch_inputs_norm, trunk_inputs_norm))
    y_pred_real = y_pred_norm[0] * scalers['Y_std'] + scalers['Y_mean']
    # 7. Cálculo del Error
    error_l2 = np.linalg.norm(y_true_full - y_pred_real) / np.linalg.norm(y_true_full)
    
    return y_true_full, y_pred_real, error_l2

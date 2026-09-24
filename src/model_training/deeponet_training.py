import h5py
import argparse 
import numpy as np
import deepxde as dde
import torch
from functools import partial
from utils import BarraProgreso, guardar_historial_csv 
import pandas as pd
import sys
import os

class GradientClipCallback(dde.callbacks.Callback):
    """Aplica gradient clipping despues de cada backward pass.
    Previene explosion del Hessiano cuando se usa lambda adaptativo.
    """
    def __init__(self, max_norm=1.0):
        super().__init__()
        self.max_norm = max_norm

    def on_train_begin(self):
        original_step = self.model.opt.step
        max_norm = self.max_norm
        net = self.model.net

        def clipped_step(closure=None):
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            return original_step(closure)

        self.model.opt.step = clipped_step

# Permitir cargar módulos desde deepxde-extensions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "deepxde-extensions")))
try:
    from jacobian_deeponet import create_jacobian_deeponet, pinball_loss
    from jacobian_deeponet_softplus import create_jacobian_deeponet_softplus, pinball_loss_softplus, AdaptivePinballLossSoftplus
    from vanilla_pinball_deeponet import create_vanilla_pinball_deeponet
    from quantile_deeponet import create_spatial_quantile_deeponet
except ImportError:
    print("Warning: jacobian_deeponet o extensiones no se pudieron cargar completamente")

def crear_red(args):
    """Centraliza la arquitectura paramétrica para funcionar con cualquier ecuación"""
    is_cartesian = getattr(args, "use_cartesian_prod", False)
    
    if getattr(args, "use_jacobian", False):
        if getattr(args, "variance_activation", "exp") == "softplus":
            JacobianNetClass = create_jacobian_deeponet_softplus(is_cartesian=is_cartesian)
            return JacobianNetClass(
                layer_sizes_branch=args.layer_sizes_branch,
                layer_sizes_trunk=args.layer_sizes_trunk,
                activation=args.activation,
                kernel_initializer=args.kernel_initializer,
                jacobian_type=getattr(args, "jacobian_type", "parameter"),
                sigma=getattr(args, "sigma", 0.05),
                scale_mode=getattr(args, "jacobian_scale", "batch")
            )
        else:
            JacobianNetClass = create_jacobian_deeponet(is_cartesian=is_cartesian)
            return JacobianNetClass(
                layer_sizes_branch=args.layer_sizes_branch,
                layer_sizes_trunk=args.layer_sizes_trunk,
                activation=args.activation,
                kernel_initializer=args.kernel_initializer,
                jacobian_type=getattr(args, "jacobian_type", "parameter"),
                sigma=getattr(args, "sigma", 0.05)
            )

    if getattr(args, "use_quantile_heads", None):
        QuantileClass = create_spatial_quantile_deeponet(is_cartesian=is_cartesian)
        return QuantileClass(
            layer_sizes_branch=args.layer_sizes_branch,
            layer_sizes_trunk=args.layer_sizes_trunk,
            activation=args.activation,
            kernel_initializer=args.kernel_initializer,
            conditioning=args.use_quantile_heads,
        )

    if getattr(args, "use_vanilla_pinball", False):
        VanillaPinballClass = create_vanilla_pinball_deeponet(is_cartesian=is_cartesian)
        return VanillaPinballClass(
            layer_sizes_branch=args.layer_sizes_branch,
            layer_sizes_trunk=args.layer_sizes_trunk,
            activation=args.activation,
            kernel_initializer=args.kernel_initializer,
            sigma=getattr(args, "sigma", 0.05)
        )
        
    if is_cartesian:
        return dde.nn.DeepONetCartesianProd(
            layer_sizes_branch=args.layer_sizes_branch,
            layer_sizes_trunk=args.layer_sizes_trunk,
            activation=args.activation,
            kernel_initializer=args.kernel_initializer
        )
    return dde.nn.DeepONet(
        layer_sizes_branch=args.layer_sizes_branch,
        layer_sizes_trunk=args.layer_sizes_trunk,
        activation=args.activation,
        kernel_initializer=args.kernel_initializer
    )

def read_data(args):
    path = args.data_path
    num_sensors = getattr(args, "num_branch_sensors", 0)

    with h5py.File(path, "r") as f:
        # Autodetectar si es el formato antiguo estructurado
        if "branch_inputs" in f.keys():
            branch_inputs = f["branch_inputs"][:]
            trunk_inputs = f["trunk_inputs"][:]
            Y = f["targets"][:]
            
            # Si existe el coeficiente paramétrico (ej. Viscosidad), lo concatenamos a la Branch
            if "params_coefficient" in f.keys():
                nu = f["params_coefficient"][:]
                if len(nu.shape) == 1:
                    nu = nu[:, None]
                branch_inputs = np.hstack([branch_inputs, nu])
                print(f"Concatenado coeficiente paramétrico. Nueva forma de Branch: {branch_inputs.shape}")
                
            return branch_inputs, trunk_inputs, Y
        else:
            # Formato jerárquico FEniCS (v1) - Cartesian
            branch_data = []
            y_data = []
            trunk_coords = None
            
            for session_id in f.keys():
                session = f[session_id]
                for sim_id in session.keys():
                    sim = session[sim_id]
                    branch_data.append(sim['field_input_f'][:])
                    if trunk_coords is None:
                        trunk_coords = sim['coordinates'][:, :2] 
                    y_data.append(sim['values'][:].squeeze())
                    
            branch_inputs = np.array(branch_data, dtype=np.float64)
            trunk_inputs = np.array(trunk_coords, dtype=np.float64)
            Y = np.array(y_data, dtype=np.float64)

            # --- SUBMUESTREO FIJO DE SENSORES ---
            if num_sensors > 0 and num_sensors < branch_inputs.shape[1]:
                print(f"Submuestreando sensores de Branch de {branch_inputs.shape[1]} a {num_sensors}...")
                # Usar una semilla FIJA (42) es vital para asegurar que la red aprenda
                # siempre de las mismas posiciones espaciales en todas las ejecuciones.
                np.random.seed(42)
                indices_fijos = np.random.choice(branch_inputs.shape[1], num_sensors, replace=False)
                indices_fijos.sort() # Mantener coherencia espacial
                branch_inputs = branch_inputs[:, indices_fijos]

            return branch_inputs, trunk_inputs, Y


def train_adam(branch_inputs, trunk_inputs, Y, args):
    # Dependiendo de si es formato FEniCS cartesiano inflado o acoplado punto a punto
    if getattr(args, "use_cartesian_prod", False):
        # En cartesiano se pasa el dataset directo y sin duplicaciones
        data = dde.data.TripleCartesianProd(
            X_train=(branch_inputs, trunk_inputs),
            y_train=Y,
            # Limitamos el test set al tamaño del batch para evitar OOM al evaluar el Jacobiano completo
            X_test=(branch_inputs[:args.batch_size], trunk_inputs),
            y_test=Y[:args.batch_size],
        )
    else:
        # Corrección de nombres (plural) para datasets emparejados tradicionales
        data = dde.data.Triple(
            X_train=(branch_inputs, trunk_inputs),
            y_train=Y,
            # Limitamos el test set al tamaño del batch para evitar OOM
            X_test=(branch_inputs[:args.batch_size], trunk_inputs[:args.batch_size]), 
            y_test=Y[:args.batch_size],
        )
    
    print(f"Branch input shape: {branch_inputs.shape}")
    print(f"Trunk input shape: {trunk_inputs.shape}")

    net = crear_red(args)
    model = dde.Model(data, net)

    # Definir la funcion de perdida
    uses_bands = (getattr(args, "use_jacobian", False) 
                  or getattr(args, "use_vanilla_pinball", False) 
                  or getattr(args, "use_quantile_heads", None) is not None)
    
    if uses_bands:
        pinball_lambda = getattr(args, "pinball_lambda", 1.0)
        if getattr(args, "decoupled_pinball", False):
            # En v7 este flag activa el "Adaptive Pinball con Amortiguacion Logaritmica"
            loss_fn = AdaptivePinballLossSoftplus(
                net,
                warmup_iters=args.warmup_iters,
                calibration_iters=10000,
                pinball_lambda=pinball_lambda
            )
        elif (getattr(args, "variance_activation", "exp") == "softplus" 
              or getattr(args, "use_vanilla_pinball", False)
              or getattr(args, "use_quantile_heads", None)):
            loss_fn = partial(pinball_loss_softplus, pinball_lambda=pinball_lambda)
        else:
            loss_fn = pinball_loss
    else:
        loss_fn = args.loss_type

    # Métrica adaptativa para ignorar los cuantiles del jacobiano al medir error
    def l2_error(y_true, y_pred):
        y_center = y_pred[..., :y_true.shape[-1]]
        return np.linalg.norm(y_true - y_center) / np.linalg.norm(y_true)
        
    def mse_center_metric(y_true, y_pred):
        y_center = y_pred[..., :y_true.shape[-1]]
        return np.mean((y_true - y_center)**2)
        
    def pinball_metric(y_true, y_pred):
        alpha = 0.1
        num_nodes = y_true.shape[-1]
        y_lower = y_pred[..., num_nodes:2*num_nodes]
        y_upper = y_pred[..., 2*num_nodes:]
        
        diff_lower = y_true - y_lower
        L_lower = np.mean(np.maximum((alpha / 2) * diff_lower, (alpha / 2 - 1) * diff_lower))
        
        diff_upper = y_true - y_upper
        L_upper = np.mean(np.maximum((1 - alpha / 2) * diff_upper, ((1 - alpha / 2) - 1) * diff_upper))
        
        return L_lower + L_upper
        
    metrics = [l2_error, mse_center_metric, pinball_metric] if uses_bands else ["l2 relative error"]

    if args.training == "single":
        # Abstracción segura del scheduler
        decay_tuple = ("step", args.decay_steps, args.decay_rate)
        model.compile("adam", lr=args.learning_rate, loss=loss_fn, decay=decay_tuple, metrics=metrics)

    elif args.training == "online":
        ruta_modelo_adam = args.model_path
        model.compile("adam", lr=args.learning_rate, loss=loss_fn, metrics=metrics)
        model.restore(ruta_modelo_adam, verbose=1)
        print(f"Modelo de Adam restaurado desde {ruta_modelo_adam}")

    callbacks = []
    if getattr(args, "ckpt_every", 0) > 0:
        callbacks.append(dde.callbacks.ModelCheckpoint(args.output_path, save_better_only=False, period=args.ckpt_every))
        print(f"Checkpoint periodico cada {args.ckpt_every} iteraciones: {args.output_path}-<step>.pt")
    if getattr(args, "decoupled_pinball", False) and getattr(args, "grad_clip", 0) > 0:
        callbacks.append(GradientClipCallback(max_norm=args.grad_clip))
        print(f"Gradient clipping activado: max_norm={args.grad_clip}")

    print(f"Entrenando con Adam por {args.iterations} iteraciones...")
    
    losshistory, train_state = model.train(
        iterations=args.iterations, 
        batch_size=args.batch_size, 
        display_every=args.display_every,
        callbacks=callbacks if callbacks else None
    )

    guardar_historial_csv(losshistory, args.output_path + "_history.csv")
    model.save(args.output_path)


def train_lbfgs(branch_inputs, trunk_inputs, Y, args):
    print("Sub-muestreando datos para L-BFGS")
    num_datos_totales = len(Y)
    n_samples = min(args.lbfgs_samples, num_datos_totales)
    indices_lbfgs = np.random.choice(num_datos_totales, size=n_samples, replace=False)

    branch_inputs_lbfgs = branch_inputs[indices_lbfgs]
    Y_lbfgs = Y[indices_lbfgs]

    if getattr(args, "use_cartesian_prod", False):
        trunk_inputs_lbfgs = trunk_inputs # Trunk (coordenadas) no deben perder forma espacial
        data_lbfgs = dde.data.TripleCartesianProd(
            X_train=(branch_inputs_lbfgs, trunk_inputs_lbfgs),
            y_train=Y_lbfgs,
            X_test=(branch_inputs, trunk_inputs), 
            y_test=Y,
        ) 
    else:
        trunk_inputs_lbfgs = trunk_inputs[indices_lbfgs]
        data_lbfgs = dde.data.Triple(
            X_train=(branch_inputs_lbfgs, trunk_inputs_lbfgs),
            y_train=Y_lbfgs,
            X_test=(branch_inputs[:2000], trunk_inputs[:2000]), 
            y_test=Y[:2000],
        )   
    
    net = crear_red(args)
    model = dde.Model(data_lbfgs, net)

    dde.optimizers.config.set_LBFGS_options(
        maxiter=args.iterations,    
        maxfun=int(args.iterations * 1.5), # Corrección: Debe ser entero
        maxcor=50,  # Reducir el history_size (maxcor) para ahorrar RAM
        ftol=1e-12,      
        gtol=1e-12      
    )

    # Definir la función de pérdida si usa jacobiano
    if getattr(args, "use_jacobian", False):
        loss_fn = pinball_loss_softplus if getattr(args, "variance_activation", "exp") == "softplus" else pinball_loss
    else:
        loss_fn = args.loss_type

    model.compile("L-BFGS", loss=loss_fn)

    # Restauración limpia de pesos
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Pesos cargados correctamente desde {args.model_path}")
    
    del checkpoint
    torch.cuda.empty_cache()

    print("Iniciando Fase L-BFGS. La consola no se actualizará paso a paso...")
    losshistory, train_state = model.train(
        iterations=args.iterations, 
        display_every=args.display_every,
    )
    model.save(args.output_path)
    guardar_historial_csv(losshistory, args.output_path + "_lbfgs_history.csv")
    


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de DeepONet para ecuación de calor")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "lbfgs"])
    parser.add_argument("--training", type=str, default="single", choices=["single", "online"])
    parser.add_argument("--model_path", type=str, default="", help="Ruta al modelo para reanudar (.pt)")
    
    # Arquitectura de red parametrizada
    parser.add_argument("--layer_sizes_branch", type=int, nargs="+", default=[1, 512, 512, 512, 512, 256], help="Tamaño de capas del Branch Net")
    parser.add_argument("--layer_sizes_trunk", type=int, nargs="+", default=[2, 512, 512, 512, 512, 256], help="Tamaño de capas del Trunk Net")
    parser.add_argument("--activation", type=str, default="tanh", help="Función de activación")
    parser.add_argument("--kernel_initializer", type=str, default="Glorot normal", help="Inicializador de los pesos")
    
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "mae", "huber"])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--decay_steps", type=int, default=10000)
    parser.add_argument("--decay_rate", type=float, default=0.5)
    
    parser.add_argument("--lbfgs_samples", type=int, default=1000)
    parser.add_argument("--display_every", type=int, default=500)
    parser.add_argument("--use_cartesian_prod", action="store_true", help="Utilizar DeepONetCartesianProd para datasets no expandidos (ej. Biarmónica)")
    
    # Opciones de Incertidumbre y Jacobiano
    parser.add_argument("--use_jacobian", action="store_true", help="Utilizar arquitectura JacobianDeepONet con incertidumbre")
    parser.add_argument("--variance_activation", type=str, choices=["exp", "softplus"], default="exp", help="Función para la varianza")
    parser.add_argument("--jacobian_type", type=str, default="parameter", choices=["parameter", "spatial"], help="Tipo de jacobiano a calcular")
    parser.add_argument("--sigma", type=float, default=0.05, help="Nivel de incertidumbre (sigma)")
    parser.add_argument("--num_branch_sensors", type=int, default=0, help="Número de sensores fijos a muestrear para la Branch Network")
    parser.add_argument("--decoupled_pinball", action="store_true", help="Activar Pinball desacoplado para evitar inestabilidad del Hessiano")
    parser.add_argument("--warmup_iters", type=int, default=30000, help="Iteraciones de warmup (solo MSE, sin Jacobiano)")
    parser.add_argument("--grad_clip", type=float, default=0, help="Max norm para gradient clipping (0 = desactivado)")
    parser.add_argument("--pinball_lambda", type=float, default=1.0, help="Multiplicador de la perdida Pinball (default: 1.0). Valores altos fuerzan bandas mas anchas.")
    parser.add_argument("--use_vanilla_pinball", action="store_true", help="Usar Pinball clasico sin Jacobiano (3 Branch heads)")
    parser.add_argument("--use_quantile_heads", type=str, default=None, choices=["x", "ux"], help="Baseline de quantile regression espacial sin Jacobiano ('x' o 'ux')")
    parser.add_argument("--jacobian_scale", type=str, default="batch", choices=["batch", "ema"], help="Normalizacion de sensibilidad: 'batch' (solo forma) o 'ema' (escala congelada)")
    parser.add_argument("--n_train", type=int, default=0, help="Usar solo las primeras N trayectorias del train set (0 = todas)")
    parser.add_argument("--ckpt_every", type=int, default=0, help="Guardar checkpoint cada N iteraciones (0 = solo al final)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla global aleatoria")
    
    args = parser.parse_args()

    # OBLIGATORIO: Usar float32 para compatibilidad con GPU y datos JAX
    dde.config.set_default_float("float32")
    dde.config.set_random_seed(args.seed)

    branch_inputs, trunk_inputs, Y = read_data(args)
    if getattr(args, "n_train", 0) and args.n_train < len(Y):
        branch_inputs, Y = branch_inputs[:args.n_train], Y[:args.n_train]
        print(f"Regimen low-data: utilizando las primeras {args.n_train} trayectorias de entrenamiento")
    
    # Asegurar que los datos sean float32 para evitar errores de mat1/mat2
    branch_inputs = branch_inputs.astype(np.float32)
    trunk_inputs = trunk_inputs.astype(np.float32)
    Y = Y.astype(np.float32)

    # ----------------- NORMALIZACIÓN (Z-SCORE) -----------------
    # Branch
    branch_mean = np.mean(branch_inputs, axis=0)
    branch_std = np.std(branch_inputs, axis=0) + 1e-8
    branch_inputs = (branch_inputs - branch_mean) / branch_std

    # Trunk
    trunk_mean = np.mean(trunk_inputs, axis=0)
    trunk_std = np.std(trunk_inputs, axis=0) + 1e-8
    trunk_inputs = (trunk_inputs - trunk_mean) / trunk_std

    # Salida (Targets)
    Y_mean = np.mean(Y, axis=0)
    Y_std = np.std(Y, axis=0) + 1e-8
    Y = (Y - Y_mean) / Y_std

    # Exportar los argumentos y escaladores para revertir el proceso durante la inferencia
    import json
    with open(args.output_path + "_args.json", "w") as fh:
        json.dump(vars(args), fh, indent=2)

    scaler_path = args.output_path + "_scalers.npz"
    np.savez(
        scaler_path, 
        branch_mean=branch_mean, branch_std=branch_std,
        trunk_mean=trunk_mean, trunk_std=trunk_std,
        Y_mean=Y_mean, Y_std=Y_std
    )
    print(f"Normalización aplicada. Parámetros de escalado guardados en: {scaler_path}")
    # -----------------------------------------------------------
    
    if args.optimizer == "adam":
        train_adam(branch_inputs, trunk_inputs, Y, args)
    elif args.optimizer == "lbfgs":
        train_lbfgs(branch_inputs, trunk_inputs, Y, args)

if __name__ == "__main__":
    main()
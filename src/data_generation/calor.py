import numpy as np
from scipy.stats import qmc
import argparse
import h5py

def solucion_analitica_scalar(x, t, L, alpha, T_inicial, num_terminos=10):
    """Solución analitica escalar u(x,t) = T"""
    T = 0.0
    for n in range(1, num_terminos + 1, 2):  # odd terms only
        An = (4 * T_inicial) / (n * np.pi)
        T += An * np.sin(n * np.pi * x / L) * np.exp(-alpha * (n * np.pi / L)**2 * t)
    return T


def generar_datos_calor(num_muestras, limites_inferiores, 
                        limites_superiores, L, alpha, data_type="float64"):

    """
    Genera un conjunto de datos para el problema de la ecuación de calor utilizando Latin Hypercube Sampling (LHS).

    Args:
        num_muestras (int): Número de puntos de datos a generar.
        limites_inferiores (array_like): Límites inferiores para el escalado de las variables de entrada.
        limites_superiores (array_like): Límites superiores para el escalado de las variables de entrada.
        L (float): Longitud del dominio espacial.
        alpha (float): Coeficiente de difusividad térmica.
        data_type (str, optional): Tipo de dato para los arreglos de numpy. Por defecto "float64".

    Returns:
        tuple: Una tupla conteniendo los puntos de entrada (x, t) y la solución analítica calculada.
    """

    if data_type == "float64":
        data_type = np.float64
    elif data_type == "float32":
        data_type = np.float32
    else:
        raise ValueError("Tipo de dato no soportado")

    dimensiones = 3

    # Generar muestras con Latin Hypercube Sampling
    sampler = qmc.LatinHypercube(d=dimensiones)
    muestras_base = sampler.random(n=num_muestras)

    #Se escalan los datos
    muestras_escaladas = qmc.scale(
        muestras_base, 
        limites_inferiores, 
        limites_superiores
    ).astype(data_type)

    # Extraer los vectores columna (forma: N, 1)
    T0_flat = muestras_escaladas[:, 0:1]
    X_flat  = muestras_escaladas[:, 1:2]
    t_flat  = muestras_escaladas[:, 2:3]

    Y = solucion_analitica_scalar(X_flat, t_flat, L, alpha, T0_flat).astype(data_type)

    branch_input = T0_flat
    trunk_input = np.hstack([X_flat, t_flat]).astype(data_type)

    return branch_input, trunk_input, Y 

def generar_archivos(branch_inputs, trunk_inputs, Y, args):
    """
    Genera un archivo HDF5 para almacenar los datos generados.

    Args:
        branch_inputs (np.ndarray): Arreglo de entrada para la rama.
        trunk_inputs (np.ndarray): Arreglo de entrada para el tronco.
        Y (np.ndarray): Arreglo de salida (solución analítica).
        args (argparse.Namespace): Objeto con los argumentos del programa.
    """
    with h5py.File(args.output_path, "w") as f:
        
        f.create_dataset("branch_inputs", data=branch_inputs, compression="gzip")
        f.create_dataset("trunk_inputs", data=trunk_inputs, compression="gzip")
        f.create_dataset("targets", data=Y, compression="gzip")
        

        f.attrs["T0_min"] = args.T0_min
        f.attrs["T0_max"] = args.T0_max
        f.attrs["x_min"] = args.x_min
        f.attrs["x_max"] = args.x_max
        f.attrs["t_min"] = args.t_min
        f.attrs["t_max"] = args.t_max
        f.attrs["L"] = args.L
        f.attrs["alpha"] = args.alpha
        f.attrs["data_type"] = args.data_type
        f.attrs["num_muestras"] = args.num_muestras
    print(f"✅ Proceso finalizado. Archivo creado en: {args.output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generador de datos LHS/Simulaciones para HPC y almacenamiento en HDF5."
    )

    parser.add_argument("--T0_min", type=float, default=50, 
                        help="Límite inferior para el coeficiente")
    parser.add_argument("--T0_max", type=float, default=55, 
                        help="Límite superior para el coeficiente")

    parser.add_argument("--x_min", type=float, default=0.0, 
                        help="Límite inferior para el coeficiente")
    parser.add_argument("--x_max", type=float, default=1.0, 
                        help="Límite superior para el coeficiente")

    parser.add_argument("--t_min", type=float, default=0.0, 
                        help="Límite inferior para el coeficiente")
    parser.add_argument("--t_max", type=float, default=1.0, 
                        help="Límite superior para el coeficiente")

    parser.add_argument("--num_muestras", type=int, default=20000, 
                        help="Cantidad de muestras a generar")
    parser.add_argument("--L", type=float, default=1.0, 
                        help="Longitud del dominio espacial")
    parser.add_argument("--alpha", type=float, default=0.1, 
                        help="Coeficiente de difusividad térmica")
    parser.add_argument("--data_type", type=str, default="float64", 
                        help="Tipo de dato para los arreglos de numpy")    
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Ruta de destino para el archivo .h5")


    args = parser.parse_args()

    print("=== Iniciando Generación de Datos ===")
    print(f"Rango de T0 : [{args.T0_min}, {args.T0_max}]")
    print(f"Rango de X : [{args.x_min}, {args.x_max}]")
    print(f"Rango de t : [{args.t_min}, {args.t_max}]")
    print(f"Longitud del dominio espacial : {args.L}")
    print(f"Coeficiente de difusividad térmica : {args.alpha}")
    print(f"Tipo de dato : {args.data_type}")
    print(f"Simulaciones totales : {args.num_muestras}")
    print(f"Archivo de salida    : {args.output_path}")


    branch_inputs, trunk_inputs, Y = generar_datos_calor(
        args.num_muestras, 
        np.array([args.T0_min, args.x_min, args.t_min]), 
        np.array([args.T0_max, args.x_max, args.t_max]), 
        args.L, 
        args.alpha, 
        args.data_type
    )


    print("\nGuardando resultados en formato HDF5...")
    generar_archivos(branch_inputs, trunk_inputs, Y, args)


if __name__ == "__main__":
    main()
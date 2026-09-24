import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def guardar_historial_csv(losshistory, nombre_archivo):
    """
    Extrae el historial de entrenamiento de DeepXDE y lo guarda en un CSV.
    """
    # 1. Extraer los pasos (iteraciones)
    pasos = np.array(losshistory.steps)
    
    # 2. Extraer la pérdida de entrenamiento
    perdida_train = np.array(losshistory.loss_train)
    
    # Si la pérdida tiene múltiples componentes, calculamos la suma total
    if perdida_train.ndim > 1:
        perdida_total = np.sum(perdida_train, axis=1)
    else:
        perdida_total = perdida_train
        
    # 3. Construir el diccionario base para el DataFrame
    datos = {
        "Iteracion": pasos,
        "Perdida_Train_Total": perdida_total
    }
    
    # 4. Extraer las métricas adicionales si existen (ej. L2 relative error)
    if len(losshistory.metrics_test) > 0:
        metricas = np.array(losshistory.metrics_test)
        if metricas.size > 0 and metricas.ndim > 1 and metricas.shape[1] > 0:
            # Asumimos que la primera métrica compilada es el L2 relativo
            datos["Error_L2_Relativo"] = metricas[:, 0]
        
    # 5. Crear el DataFrame y exportar a CSV
    df = pd.DataFrame(datos)
    df.to_csv(nombre_archivo, index=False)
    print(f"Historial guardado correctamente en: {nombre_archivo}")

# Definimos el Callback personalizado
class BarraProgreso(dde.callbacks.Callback):
    def __init__(self, total_iteraciones):
        super().__init__()
        self.total_iteraciones = total_iteraciones
        
    def on_train_begin(self):
        # Inicia la barra cuando arranca el entrenamiento
        self.pbar = tqdm(total=self.total_iteraciones, desc="Entrenando DeepONet")
        
    def on_epoch_end(self):
        # Actualiza la barra en cada iteración
        self.pbar.update(1)
        
    def on_train_end(self):
        # Cierra la barra al terminar
        self.pbar.close()


def graficar_curva_entrenamiento(losshistory):
    # 1. Extraer los pasos (iteraciones)
    pasos = np.array(losshistory.steps)
    
    # 2. Extraer pérdida de entrenamiento y test
    # Si hay múltiples salidas, DeepXDE guarda una matriz. Sumamos para obtener la pérdida total.
    perdida_train = np.array(losshistory.loss_train)
    if perdida_train.ndim > 1:
        perdida_train = np.sum(perdida_train, axis=1)
        
    perdida_test = np.array(losshistory.loss_test)
    if perdida_test.ndim > 1:
        perdida_test = np.sum(perdida_test, axis=1)

    # 3. Crear la figura
    plt.figure(figsize=(10, 6))
    
    # Graficar en escala logarítmica (semilogy)
    plt.semilogy(pasos, perdida_train, label='Pérdida Entrenamiento (MSE)', color='blue', linewidth=2)
    plt.semilogy(pasos, perdida_test, label='Pérdida Validación (MSE)', color='orange', linewidth=2, linestyle='--')
    
    # 4. Graficar métricas adicionales (ej. L2 relative error) si existen
    if len(losshistory.metrics_test) > 0:
        metricas = np.array(losshistory.metrics_test)
        error_l2 = metricas[:, 0]  # Asumimos que el L2 es la primera métrica
        plt.semilogy(pasos, error_l2, label='Error Relativo L2 (Test)', color='green', linewidth=2, linestyle=':')

    # 5. Formateo del gráfico
    plt.title('Curva de Entrenamiento de la DeepONet', fontsize=14)
    plt.xlabel('Iteraciones', fontsize=12)
    plt.ylabel('Pérdida (Escala Logarítmica)', fontsize=12)
    
    # Cuadrícula fina para leer mejor los órdenes de magnitud
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Mostrar el gráfico
    plt.show()


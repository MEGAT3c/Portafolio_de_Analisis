import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv('medical_insurance.csv')

# Preprocesamiento: eliminamos la columna 'region' y convertimos las categorías a valores numéricos
data = data.drop(columns=['region'])
data['sex'] = data['sex'].map({'male': 1, 'female': 0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})

# Definir características (X) y etiquetas (y)
X = data.drop('charges', axis=1).values  # X contiene las características
y = data['charges'].values  # y contiene las etiquetas (valores de los cargos médicos)

# Dividimos los datos en entrenamiento (60%), validación (20%) y prueba (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Función de distancia euclidiana para calcular el vecino más cercano
def euclidean_distance(x1, x2):
    """
    Calcula la distancia euclidiana entre dos puntos.

    Parameters:
    x1, x2: vectores de las características.

    Returns:
    La distancia euclidiana entre x1 y x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Función de predicción usando el vecino más cercano (algoritmo KNN con k=1)
def predict(X_train, y_train, x_test):
    """
    Predice el valor de cargo médico para un nuevo conjunto de características usando el algoritmo KNN con k=1.

    Parameters:
    X_train: Conjunto de características de entrenamiento.
    y_train: Etiquetas correspondientes al conjunto de entrenamiento.
    x_test: Un nuevo conjunto de características para hacer la predicción.

    Returns:
    El valor de la predicción (cargo médico).
    """
    distances = np.array([euclidean_distance(x_test, x_train) for x_train in X_train])
    min_index = np.argmin(distances)  # Encuentra el índice del vecino más cercano
    return y_train[min_index]

# Función de predicción desde la interfaz gráfica
def make_prediction():
    """
    Recoge los datos ingresados por el usuario desde la interfaz gráfica, los procesa
    y realiza la predicción del costo médico usando el modelo entrenado.
    """
    try:
        # Obtener los valores de entrada del usuario desde los campos de entrada
        inputs = [
            float(entry_age.get()),
            1 if var_sex.get() == 'male' else 0,
            float(entry_bmi.get()),
            int(entry_children.get()),
            1 if var_smoker.get() == 'yes' else 0
        ]
        
        x_test = np.array(inputs)
        result = predict(X_train, y_train, x_test)  # Predicción del costo
        messagebox.showinfo("Resultado de la Predicción", f"El costo estimado es: ${result:.2f}")

    except ValueError:
        messagebox.showerror("Error", "Por favor, ingresa valores válidos en todos los campos.")

# Métricas de evaluación
def evaluate_model():
    """
    Evalúa el modelo de vecino más cercano (KNN con k=1) en el conjunto de prueba y
    muestra la precisión del modelo usando precision_score.
    """
    # Predicción en el conjunto de prueba
    y_pred = [predict(X_train, y_train, x_test) for x_test in X_test]

    # Convertimos los valores continuos a clases basadas en percentiles
    y_pred_classes = np.digitize(y_pred, bins=np.percentile(y_train, [25, 50, 75]))
    y_test_classes = np.digitize(y_test, bins=np.percentile(y_train, [25, 50, 75]))

    # Calcular precisión usando precision_score
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')

    # Mostrar la precisión
    messagebox.showinfo("Precisión del Modelo", f"La precisión del modelo es: {precision:.2f}")

# Configuración de la interfaz gráfica con tkinter
root = tk.Tk()
root.title("Predicción de Cargos Médicos")

tk.Label(root, text="Edad:").grid(row=0, column=0)
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1)

tk.Label(root, text="Sexo:").grid(row=1, column=0)
var_sex = tk.StringVar(value='male')
tk.OptionMenu(root, var_sex, 'male', 'female').grid(row=1, column=1)

tk.Label(root, text="BMI:").grid(row=2, column=0)
entry_bmi = tk.Entry(root)
entry_bmi.grid(row=2, column=1)

tk.Label(root, text="Número de Hijos:").grid(row=3, column=0)
entry_children = tk.Entry(root)
entry_children.grid(row=3, column=1)

tk.Label(root, text="Fumador:").grid(row=4, column=0)
var_smoker = tk.StringVar(value='no')
tk.OptionMenu(root, var_smoker, 'yes', 'no').grid(row=4, column=1)

predict_button = tk.Button(root, text="Predecir Cargos", command=make_prediction)
predict_button.grid(row=5, column=0, columnspan=2)

# Botón para evaluar el modelo
eval_button = tk.Button(root, text="Evaluar Modelo", command=evaluate_model)
eval_button.grid(row=6, column=0, columnspan=2)

root.mainloop()

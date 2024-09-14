import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import messagebox

# Cargar el conjunto de datos
data = pd.read_csv('heart_2020_cleaned.csv')

# Convertir la variable objetivo a binaria (1 = Yes, 0 = No)
data['HeartDisease'] = data['HeartDisease'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convertir variables categóricas a numéricas con Label Encoding
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separar características (X) y variable objetivo (y)
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Dividir los datos en entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Entrenar el modelo de regresión logística
#Se selecciono un número máximo de iterasiones de mil ya que se realizaron varias pruebas previas con diferentes iteraciónes y la mejor forma
# de entrenar el modelo evitando el overfiting y mejorar la regresión fue de 1000 iteraciónes, el random state usado fue justamente para evitar
# que el modelo se entrenara de la misma manera en cada ejecución del programa y no generar overfiting ya que nuestro data set es de 319795 valores
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Función para predecir la enfermedad cardíaca
def predict_heart_disease():
    try:
        # Recopilar entradas del usuario
        input_data = {
            'BMI': float(entry_bmi.get()),
            'PhysicalHealth': float(entry_physical_health.get()),
            'MentalHealth': float(entry_mental_health.get()),
            'SleepTime': float(entry_sleep_time.get())
        }

        # Asignar valores para variables categóricas
        for col, encoder in label_encoders.items():
            input_data[col] = encoder.transform([combos[col].get()])[0]

        # Crear un DataFrame con los valores ingresados
        user_data = pd.DataFrame([input_data])
        
        # Asegurar que las columnas estén en el mismo orden que durante el entrenamiento
        user_data = user_data[X.columns]

        # Realizar la predicción
        prediction = model.predict(user_data)
        result_text = "Enfermedad cardíaca" if prediction[0] == 1 else "Sin enfermedad cardíaca"
        
        # Mostrar el resultado en un cuadro de mensaje
        messagebox.showinfo("Resultado de la Predicción", f"Resultado: {result_text}")
    
    except Exception as e:
        messagebox.showerror("Error", f"Error en la entrada: {e}")

# Crear la ventana principal
root = tk.Tk()
root.title("Predicción de Enfermedad Cardíaca")

# Etiquetas y entradas para variables numéricas
tk.Label(root, text="BMI:").grid(row=0, column=0)
entry_bmi = tk.Entry(root)
entry_bmi.grid(row=0, column=1)

tk.Label(root, text="Physical Health (días malos):").grid(row=1, column=0)
entry_physical_health = tk.Entry(root)
entry_physical_health.grid(row=1, column=1)

tk.Label(root, text="Mental Health (días malos):").grid(row=2, column=0)
entry_mental_health = tk.Entry(root)
entry_mental_health.grid(row=2, column=1)

tk.Label(root, text="Sleep Time (horas):").grid(row=3, column=0)
entry_sleep_time = tk.Entry(root)
entry_sleep_time.grid(row=3, column=1)

# Comboboxes para variables categóricas
combos = {}
row_index = 4
for col, encoder in label_encoders.items():
    tk.Label(root, text=f"{col}:").grid(row=row_index, column=0)
    combo = tk.StringVar()
    tk.OptionMenu(root, combo, *encoder.classes_).grid(row=row_index, column=1)
    combos[col] = combo
    row_index += 1

# Botón de predicción
tk.Button(root, text="Predecir", command=predict_heart_disease).grid(row=row_index, columnspan=2)

# Iniciar la aplicación
root.mainloop()

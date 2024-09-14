# Portafolio_de_Análisis

Diagnóstico Inicial del Modelo


- Grado de Sesgo (Bias): Medio-Alto


  - El modelo tiene una alta precisión para la clase "0" (pacientes sin enfermedad cardíaca), con un F1-score de 96%, pero un rendimiento pobre para la clase "1" (pacientes con enfermedad cardíaca), con un F1-score de solo 16%. Esto indica que el modelo está subestimando la clase minoritaria (enfermedad cardíaca), lo que es un signo de sesgo alto hacia la clase mayoritaria.El sesgo alto significa que el modelo tiene dificultades para generalizar correctamente para la clase positiva. Esto sugiere que el modelo está subestimando patrones importantes en los datos para predecir la clase minoritaria.
- Grado de Varianza: Bajo

  - El modelo tiene un accuracy del 91.46% en el conjunto de validación, lo que sugiere que no hay una discrepancia significativa entre los resultados del entrenamiento y validación. Esto indica que el modelo no está sobreajustando a los datos de entrenamiento. Un grado bajo de varianza indica que el modelo es bastante estable y no muestra grandes fluctuaciones entre el conjunto de entrenamiento y el de validación. Es decir, el modelo no está demasiado sensible a los cambios en los datos.
- Nivel de Ajuste del Modelo: Underfitting (Subajuste)
  - Dado que el modelo tiene un bajo rendimiento en la clase minoritaria ("1") y parece que no está capturando adecuadamente las relaciones entre las características y el objetivo, parece que el modelo está subajustado. No está aprendiendo lo suficiente de los datos para predecir correctamente los casos de enfermedad cardíaca. El subajuste (underfitting) ocurre cuando el modelo es demasiado simple o no entrena suficientemente en los datos para capturar patrones importantes. Esto parece ser el caso aquí, ya que aunque el modelo tiene una buena precisión general, falla en detectar correctamente la clase minoritaria.


## Mejora del Modelo:


Basado en este diagnóstico, vamos a aplicar algunas técnicas para mejorar el rendimiento del modelo:

- Balanceo de clases:

    El conjunto de datos está desbalanceado (más pacientes sin enfermedad cardíaca que con), lo que está afectando el rendimiento en la predicción de la clase minoritaria. Vamos a aplicar una técnica de re-muestreo o utilizar el parámetro class_weight='balanced' en el modelo de regresión logística.


- Regularización:

  El modelo pasado está usando regularización con el parámetro C=1.0 (por defecto). Dado que estamos viendo un subajuste, disminuiremos el nivel de regularización (aumentando el valor de C) para permitir que el modelo capture más relaciones.

  
- Ajuste de hiperparámetros:

  Vamos a ajustar los hiperparámetros, en particular el parámetro C, para ver si un valor mayor mejora el rendimiento del modelo en la clase minoritaria.


En el nuevo código se cuenta con dos modelos nuevos usando GridSearch y el segundo usando GridSearch y Cross Validation, Teniendo las siguientes mejoras en su implementación.


- Modelo 1:
  Mejor parámetro C: {'C': 100}
Accuracy en validación: 0.7387062477850278
Reporte de clasificación:
              precision    recall  f1-score   support

           0       0.97      0.74      0.84     43863
           1       0.21      0.77      0.33      4106

    accuracy                           0.74     47969
   macro avg       0.59      0.75      0.59     47969
weighted avg       0.91      0.74      0.79     47969


- Modelo 2:
  Mejor parámetro C: {'C': 100}
Accuracy en validación: 0.7387062477850278
Reporte de clasificación:
              precision    recall  f1-score   support

           0       0.97      0.74      0.84     43863
           1       0.21      0.77      0.33      4106

    accuracy                           0.74     47969
   macro avg       0.59      0.75      0.59     47969
weighted avg       0.91      0.74      0.79     47969

Puntuaciones de validación cruzada (F1-score): [0.33698662 0.33765935 0.33316242 0.33862799 0.335942  ]
Promedio de F1-score en validación cruzada: 0.33647567652478105

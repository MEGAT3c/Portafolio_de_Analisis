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

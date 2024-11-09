# Se importan las librerías para el template y los renders
from django.shortcuts import render
import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------------
def main(request):
    # *** Plantilla ***
    return render(request, 'index.html', context={})

def prediccion(request):
    # Definir los datos de entrada y salida para la compuerta lógica XOR
    datos_entrada = np.array([
                              [0, 0],
                              [0, 1],
                              [1, 0],
                              [1, 1],
                            ], "float32")
    
    datos_salida = np.array([
             [0],
             [1],
             [1],
             [0],
             ], "float32")
    
    # Definir el modelo de la red neuronal
    model = Sequential()
    # Capa de entrada (2 entradas)
    model.add(Dense(8, input_dim=2, activation='relu'))
    # Capa de salida (1 salida)
    model.add(Dense(1, activation='sigmoid'))
    
    # Compilar el modelo
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

    # Entrenamiento del modelo
    history = model.fit(datos_entrada, datos_salida, epochs=1000, verbose=0)

    # Obtener las predicciones del modelo
    predicciones = model.predict(datos_entrada).round()  # Redondear las predicciones a 0 o 1

    # Pasar las predicciones al contexto
    prediccion_resultado = []
    for i, pred in enumerate(predicciones):
        prediccion_resultado.append({
            'entrada': [int(x) for x in datos_entrada[i]],  # Entrada (X1, X2)
            'prediccion': int(pred[0]),   # Convertir la predicción a entero (0 o 1)
            'salida': int(datos_salida[i][0]),  # Salida real como entero
            'loss': history.history['loss'][-1]  # Último valor de "loss"
        })

    # El contexto debe ser un diccionario
    context = {'prediccion': prediccion_resultado}

    # *** Plantilla ***
    return render(request, 'index.html', context=context)

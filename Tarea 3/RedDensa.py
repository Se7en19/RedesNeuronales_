
"""

Esta tarea es para entrenar una red neuronal con una capa densa
que identifique el numero de una imagen de un digito escrito a mano
utilizando las librerias de keras.

"""

''' Importar librerias '''
from calendar import EPOCH
from keras.engine import keras_tensor
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.distribute.values import SyncOnReadVariable



""" Inicializamos WandB """

import wandb
from wandb.integration.keras import WandbMetricsLogger,WandbModelCheckpoint
wandb.login()

"""

EXPERIMENTO 3 con regularización L1

En este tercer experimento los parametros a modificar serán:

batch_size = 60 -> 120
optimizer = RMSprop -> SGD
neu_densa = 100 -> 512

funcion de costo = binary_crossentropy -> mean_squared_error 
"""


""" Variables iniciales """
learning_rate = 3.0
epochs = 30
batch_size = 120
neu_entra = 784 # Numero de neuronas en la capa de entrada
neu_densa = 512 # Numero de neuronas en la capa densa 
l1 = 0.01 # Regularizador L1
l2 = 0.01 # Regularizador L2
l1_l2_l1 = 0.005 # Factor L1 para L1L2
l1_l2_l2 = 0.005 # Factor L2 para L1L2



wandb.init(
        project = 'Red-Densa-MNIST-Tarea_3',
        config={
            "learning_rate": learning_rate,
            "epoch": epochs,
            "batch_size": batch_size,
            "loss_function": 'mean_squared_error',
            "optimizer": "SGD",  # Solo el nombre del optimizador
            "metrics": ["accuracy"],
            "N_entra": neu_entra,
            "N_densa": neu_densa,
            "l1": l1,
            "l2": l2,
            "l1_l2_l1": l1_l2_l1,
            "l1_l2_l2": l1_l2_l2
        }
    )


config = wandb.config 


""" Cargar los datos """
dataset = mnist.load_data()
dat = np.array(dataset)

""" Separar los datos en entrenamiento y prueba """
(X_train, y_train),(X_test, y_test) = dat 

""" Normalizar los datos """
X_trainv = X_train.reshape(X_train.shape[0], 784)
X_testv = X_test.reshape(X_test.shape[0], 784)
X_trainv = X_trainv.astype('float32')
X_testv = X_testv.astype('float32')
X_trainv /= 255
X_testv /= 255

""" Codificamos los datos de salida"""
num_clases = 10
y_trainc = keras.utils.to_categorical(y_train, 
                                      num_clases)
y_testc = keras.utils.to_categorical(y_test, 
                                     num_clases)

""" Creamos la red neuronal"""
model = Sequential()

# Agregamos la capa densa
model.add(Dense(config.N_densa,
                activation='sigmoid', 
                input_shape=(config.N_entra,),
                kernel_regularizer=regularizers.l1_l2(l1=config.l1_l2_l1, l2=config.l1_l2_l2)))

# Agregamos la capa de salida
model.add(Dense(num_clases, 
                activation='softmax',
                kernel_regularizer=regularizers.l1_l2(l1=config.l1_l2_l1, l2=config.l1_l2_l2)))

# Resumen de la red
model.summary()
print('\n')
print('\n')
""" Compilamos la red """
# Crear el optimizador basado en la configuración
if config.optimizer == "RMSprop":
    optimizer = RMSprop(learning_rate=config.learning_rate)
elif config.optimizer == "SGD":
    optimizer = SGD(learning_rate=config.learning_rate)
elif config.optimizer == "Adam":
    optimizer = Adam(learning_rate=config.learning_rate)
else:
    optimizer = RMSprop(learning_rate=config.learning_rate)  # Por defecto

model.compile(loss=config.loss_function, 
             optimizer=optimizer,
             metrics = config.metrics
             )
print('\n')
print('\n')
""" Entrenamos la red """
history = model.fit(
                X_trainv,
                y_trainc,
                batch_size=config.batch_size,
                epochs=config.epoch,
                verbose=1,
                validation_data=(X_testv, y_testc),
                callbacks=[
                      WandbMetricsLogger(log_freq=5)
                      # WandbModelCheckpoint("models")  # Comentado para evitar apertura de ventanas
                    ]
)
print('\n')
print('\n')

score = model.evaluate(X_testv, y_testc, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
a = model.predict(X_testv)
print(f"Resultado de la red: {a[1]}")
print(f"Resultado verdadero: {y_testc[1]}")


"""

Esta tarea es para entrenar una red neuronal con una capa densa
que identifique el numero de una imagen de un digito escrito a mano
utilizando las librerias de keras.

"""

''' Importar librerias '''
from calendar import EPOCH
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.distribute.values import SyncOnReadVariable

""" Variables iniciales """
learning_rate = 3.0
epochs = 30
batch_size = 10

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
y_trainc = keras.utils.to_categorical(y_train, num_clases)
y_testc = keras.utils.to_categorical(y_test, num_clases)

""" Creamos la red neuronal"""
model = Sequential()
# Agregamos la capa densa
model.add(Dense(30,activation='sigmoid', input_shape=(784,)))
# Agregamos la capa de salida
model.add(Dense(num_clases, activation='sigmoid'))
# Resumen de la red
model.summary()

""" Compilamos la red """
model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=learning_rate), metrics = ['accuracy'])

""" Entrenamos la red """
history = model.fit(
                X_trainv,
                y_trainc,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(X_testv, y_testc)
)

score = model.evaluate(X_testv, y_testc, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
a = model.predict(X_testv)
print("Resultado de la red: ",a[1])
print("Resultado verdadero: ",y_testc[1])

"""
Este script entrena una red neuronal para reproducir funciones en el intervalo [-1,1].

Funciones a utilizar:
a) f(x) = 3sin(pi*x)
b) f(x) = 1+2x+4x^3

"""

""" PROGRAMA PRINCIPAL """
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np
import math
import wandb
from wandb.integration.keras import WandbCallback

loss_tracker = keras.metrics.Mean(name='loss')

class Funsol(keras.Model):
    @property
    def metrics(self):
        return [loss_tracker]

    def train_step(self, data):
        batch_size = 10
        x = tf.random.uniform((batch_size, 1), minval=-1, maxval=1)
        eq = 3.0*tf.math.sin(tf.constant(np.pi,x.dtype)*x)

        with tf.GradientTape() as tape:
            ypred = self(x, training=True)
            loss = tf.reduce_mean(keras.losses.mean_squared_error(eq, ypred))

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        loss_tracker.update_state(loss)

        return {'loss': loss_tracker.result()}

class SinTransform(keras.layers.Layer):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        # ❗ CORREGIDO: frecuencias π·k (no π + k)
        self.freq = np.pi * tf.range(1., num_outputs + 1)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=[self.num_outputs],
            initializer="glorot_uniform", trainable=True
        )

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=self.dtype)
        x = tf.reshape(x, (-1, 1))                  # (N,1)
        arg = x * self.freq[tf.newaxis, :]          # (N,K)
        modes = tf.sin(arg)                         # (N,K)
        y = modes @ self.kernel[:, tf.newaxis]      # (N,1)
        return y

class PlotCallback(keras.callbacks.Callback):
    def __init__(self, x_plot):
        super().__init__()
        self.x_plot = x_plot.astype("float32").reshape(-1, 1)

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_plot, verbose=0).squeeze()
        x_np = self.x_plot.squeeze()
        y_true = 3*np.sin(np.pi*x_np)

        plt.figure()
        plt.plot(x_np, y_true, label="3 sin(πx)")
        plt.plot(x_np, y_pred, "--", label="Red")
        plt.title(f"Epoch {epoch+1}")
        plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.grid(True, alpha=0.3)

        wandb.log({
            "epoch": epoch + 1,
            "curve": wandb.Image(plt)   # sube la figura
        })
        plt.close()


wandb.init(
    project="aprox-funciones",     # <-- cambia al nombre de tu proyecto
    name="3*sin-pi-x",
    config={
        "model": "Funsol+SinTransform",
        "num_outputs": 5,
        "optimizer": "SGD",
        "lr": 0.2,
        "epochs": 200,
        "internal_batch": 10,      # el que usas dentro de train_step
        "intervalo": "[-1,1]"
    }
)

config = wandb.config 

inputs = keras.Input(shape=(1,))
outputs = SinTransform(5)(inputs)
model = Funsol(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.2))

model.summary()

x_dummy = tf.linspace(-1, 1, 100)  # no se usa en tu train_step, pero sirve para contar steps
callbacks = [
    WandbCallback(log_weights=True, save_model=True),  # registra métricas y pesos
    PlotCallback(x_plot=np.linspace(-1, 1, 400))       # sube la curva por época
]

history = model.fit(
    x_dummy,
    epochs=wandb.config.epochs,
    verbose=1,
    callbacks=callbacks
)


x_test = np.linspace(-1, 1, 400).astype("float32").reshape(-1,1)
y_hat = model.predict(x_test, verbose=0).squeeze()
y_true = 1.0+2.0*x_test.squeeze()+4.0*(x_test.squeeze()**3)

x_test = np.linspace(-1, 1, 400).astype("float32").reshape(-1,1)
y_hat = model.predict(x_test, verbose=0).squeeze()
y_true = 3*np.sin(np.pi*x_test.squeeze())

plt.figure()
plt.plot(x_test.squeeze(), y_true, label="3 sin(πx)")
plt.plot(x_test.squeeze(), y_hat, "--", label="Red")
plt.legend(); plt.grid(True, alpha=0.3); plt.title("Resultado final")
wandb.log({"final_curve": wandb.Image(plt)})
plt.show()

wandb.finish()
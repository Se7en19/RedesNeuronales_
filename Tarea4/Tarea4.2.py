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
        eq = 1.0+2.0*x+4.0*(x**3)

        with tf.GradientTape() as tape:
            ypred = self(x, training=True)
            loss = tf.reduce_mean(keras.losses.mean_squared_error(eq, ypred))

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        loss_tracker.update_state(loss)

        return {'loss': loss_tracker.result()}

class PolyTransform(keras.layers.Layer):
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree

    def build(self, input_shape):
        # pesos para [1, x, x^2, ..., x^degree]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.degree + 1,),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=self.dtype)
        x = tf.reshape(x, (-1, 1))  # (N,1)

        # construye Phi = [1, x, x^2, ..., x^degree]
        feats = [tf.ones_like(x)]
        for d in range(1, self.degree + 1):
            feats.append(feats[-1] * x)
        Phi = tf.concat(feats, axis=1)              # (N, degree+1)

        y = Phi @ self.kernel[:, tf.newaxis]        # (N,1)
        return y

class PlotCallback(keras.callbacks.Callback):
    def __init__(self, x_plot):
        super().__init__()
        self.x_plot = x_plot.astype("float32").reshape(-1, 1)

    def on_epoch_end(self, epoch, logs=None):
        # --- todo este bloque debe ir indentado con 4 espacios ---
        x_np = self.x_plot.squeeze()

        # Función objetivo 1 + 2x + 4x^3
        y_true = 1.0 + 2.0*x_np + 4.0*(x_np**3)

        # Predicción del modelo
        y_pred = self.model.predict(self.x_plot, verbose=0).squeeze()

        # Gráfica y logging a W&B
        plt.figure()
        plt.plot(x_np, y_true, label="1 + 2x + 4x^3")
        plt.plot(x_np, y_pred, "--", label="Predicción")
        plt.title(f"Epoch {epoch+1}")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(); plt.grid(True, alpha=0.3)

        wandb.log({
            "epoch": epoch + 1,
            "curve": wandb.Image(plt)
        })
        plt.close()
wandb.init(
    project="aprox-funciones",     
    name="1+2x+4x^3",
    config={
        "basis": "PolyTransform",
        "degree": 3,
        "num_features":4,
        "optimizer": "SGD",
        "lr": 0.2,
        "epochs": 200,
        "internal_batch": 10,      
        "intervalo": "[-1,1]"
    }
)

config = wandb.config 

inputs = keras.Input(shape=(1,))
outputs = PolyTransform(degree=config.degree)(inputs)
model = Funsol(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.SGD(learning_rate=config.lr))

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

plt.figure()
plt.plot(x_test.squeeze(), y_true, label="1 + 2x + 4x^3")
plt.plot(x_test.squeeze(), y_hat, "--", label="Predicción")
plt.legend(); plt.grid(True, alpha=0.3); plt.title("Resultado final")
wandb.log({"final_curve": wandb.Image(plt)})
plt.show()


wandb.finish()
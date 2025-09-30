"""
Capa polinómica grado 3 entrenable para aproximar f(x) = cos(2x) en [-1,1].
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
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

        eq = tf.math.cos(2.0 * x)

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
        # Pesos a0..a3 para [1, x, x^2, x^3]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.degree + 1,),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=self.dtype)
        x = tf.reshape(x, (-1, 1)) 
        feats = [tf.ones_like(x)]
        for _ in range(1, self.degree + 1):
            feats.append(feats[-1] * x)
        Phi = tf.concat(feats, axis=1)             
        y = Phi @ self.kernel[:, tf.newaxis]        
        return y

class PlotCallback(keras.callbacks.Callback):
    def __init__(self, x_plot):
        super().__init__()
        self.x_plot = x_plot.astype("float32").reshape(-1, 1)

    def on_epoch_end(self, epoch, logs=None):
        x_np = self.x_plot.squeeze()
        
        y_true = np.cos(2.0 * x_np)
        
        y_pred = self.model.predict(self.x_plot, verbose=0).squeeze()

        
        plt.figure()
        plt.plot(x_np, y_true, label="cos(2x)")
        plt.plot(x_np, y_pred, "--", label="Predicción")
        plt.title(f"Epoch {epoch+1}")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(); plt.grid(True, alpha=0.3)

       
        coef = self.model.layers[1].kernel.numpy() 
        wandb.log({
            "epoch": epoch + 1,
            "curve": wandb.Image(plt),
            "a0": float(coef[0]),
            "a1": float(coef[1]),
            "a2": float(coef[2]),
            "a3": float(coef[3]),
        })
        plt.close()


wandb.init(
    project="aprox-funciones",
    name="poly-fit-cos2x",
    config={
        "basis": "PolyTransform",
        "degree": 3,
        "num_features": 4,   
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
model.compile(optimizer=SGD(learning_rate=config.lr))
model.summary()


x_dummy = tf.linspace(-1, 1, 100)  
callbacks = [
    WandbCallback(log_weights=True, save_model=True),
    PlotCallback(x_plot=np.linspace(-1, 1, 400))
]
history = model.fit(
    x_dummy,
    epochs=config.epochs,
    verbose=1,
    callbacks=callbacks
)


x_test = np.linspace(-1, 1, 400).astype("float32").reshape(-1,1)
y_hat = model.predict(x_test, verbose=0).squeeze()
y_true = np.cos(2.0 * x_test.squeeze())

plt.figure()
plt.plot(x_test.squeeze(), y_true, label="cos(2x)")
plt.plot(x_test.squeeze(), y_hat, "--", label="Predicción")
plt.legend(); plt.grid(True, alpha=0.3); plt.title("Resultado final")
wandb.log({"final_curve": wandb.Image(plt)})
plt.show()


coef = model.layers[1].kernel.numpy()
print(f"Coeficientes aprendidos: a0={coef[0]:.4f}, a1={coef[1]:.4f}, a2={coef[2]:.4f}, a3={coef[3]:.4f}")

wandb.finish()

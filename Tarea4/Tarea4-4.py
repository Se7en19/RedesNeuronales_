"""

Problema 4:
(a) x y' + y = x^2 cos x,   y(0)=0
(b) y'' = -y,               y(0)=1, y'(0)=-0.5

intervalo [-5,5]
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import wandb


def build_mlp(hidden=(64, 64, 64), act="tanh", seed=0):
    tf.keras.utils.set_random_seed(seed)
    m = keras.Sequential([keras.layers.Input(shape=(1,))])
    for h in hidden:
        m.add(keras.layers.Dense(h, activation=act))
    m.add(keras.layers.Dense(1)) 
    return m


def grad_y(model, x):
    with tf.GradientTape() as t:
        t.watch(x)
        y = model(x)
    dy = t.gradient(y, x)
    return y, dy

def grad2_y(model, x):
    with tf.GradientTape() as t2:
        t2.watch(x)
        with tf.GradientTape() as t1:
            t1.watch(x)
            y = model(x)
        dy = t1.gradient(y, x)
    d2y = t2.gradient(dy, x)
    return y, dy, d2y


def loss_a(model, x_col, x0, lam_bc=10.0):
    # (a) x y' + y = x^2 cos x, y(0)=0
    y, dy = grad_y(model, x_col)
    res = x_col * dy + y - (x_col**2) * tf.cos(x_col)
    pde = tf.reduce_mean(tf.square(res))

    y0 = model(x0)
    bc = tf.reduce_mean(tf.square(y0 - 0.0))
    return pde + lam_bc * bc, (pde, bc)

def loss_b(model, x_col, x0, lam_bc=10.0):
    # (b) y'' = -y, y(0)=1, y'(0)=-0.5
    y, dy, d2y = grad2_y(model, x_col)
    res = d2y + y
    pde = tf.reduce_mean(tf.square(res))

    y0, dy0 = grad_y(model, x0)
    bc = tf.reduce_mean(tf.square(y0 - 1.0)) + tf.reduce_mean(tf.square(dy0 + 0.5))
    return pde + lam_bc * bc, (pde, bc)


def train_step(model, opt, x_col, x0, which, lam_bc):
    with tf.GradientTape() as tape:
        if which == "a":
            L, (pde, bc) = loss_a(model, x_col, x0, lam_bc)
        else:
            L, (pde, bc) = loss_b(model, x_col, x0, lam_bc)
    grads = tape.gradient(L, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return L, pde, bc


def train_pinn(which="a", steps=4000, lr=2e-3, lam_bc=20.0, seed=0,
               log_every=50, project="aprox-funciones", run_name=None):

    wandb.init(project=project,
               name=run_name or f"pinn-edo-{which}",
               config={"which": which, "steps": steps, "lr": lr,
                       "lam_bc": lam_bc, "domain": "[-5,5]"})
    model = build_mlp(hidden=(64,64,64), act="tanh", seed=seed)
    opt = keras.optimizers.Adam(lr)

    # Construye modelo y estado del optimizador ANTES del tf.function
    x0 = tf.zeros((1,1), dtype=tf.float32)
    _ = model(x0)                          # fuerza build del modelo
    opt.build(model.trainable_variables)   # crea slot variables FUERA del grafo

    @tf.function  # grafo nuevo por RUN; cierra sobre model/opt/x0/which
    def step(x_col):
        with tf.GradientTape() as tape:
            if which == "a":
                L, (pde, bc) = loss_a(model, x_col, x0, lam_bc)
            else:
                L, (pde, bc) = loss_b(model, x_col, x0, lam_bc)
        grads = tape.gradient(L, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return L, pde, bc

    for i in range(1, steps + 1):
        x_col = tf.random.uniform((512,1), -5.0, 5.0, dtype=tf.float32)
        L, pde, bc = step(x_col)
        if i % log_every == 0:
            print(f"{which} | step {i:5d} | loss={float(L):.4e} | pde={float(pde):.4e} | bc={float(bc):.4e}")
            wandb.log({"step": i, "loss": float(L), "pde": float(pde), "bc": float(bc)})

    return model



def y_true_a(x):
    # y = x sin x + 2 cos x - 2 sin x / x, con y(0)=0 por límite
    x = np.array(x, dtype=np.float64)
    sinx_over_x = np.where(np.abs(x) < 1e-12, 1.0, np.sin(x)/x)
    return x*np.sin(x) + 2*np.cos(x) - 2*sinx_over_x

def y_true_b(x):
    # y = cos x - 0.5 sin x
    return np.cos(x) - 0.5*np.sin(x)


def plot_and_log(which, model, xs):
    xs1 = xs.squeeze()
    y_pred = model.predict(xs, verbose=0).squeeze()
    if which == "a":
        y_ref = y_true_a(xs1)
        titulo = r"(a)  $x y' + y = x^2 \cos x,\ y(0)=0$"
    else:
        y_ref = y_true_b(xs1)
        titulo = r"(b)  $y'' = -y,\ y(0)=1,\ y'(0)=-0.5$"

    fig, ax = plt.subplots()
    ax.plot(xs1, y_ref, label="Analítica")
    ax.plot(xs1, y_pred, "--", label="PINN")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(titulo)
    ax.legend(); ax.grid(True, alpha=0.3)

    
    wandb.log({f"final_curve_{which}": wandb.Image(fig)})
    plt.close(fig)   


if __name__ == "__main__":
    xs = np.linspace(-5, 5, 801).astype("float32").reshape(-1,1)

    
    model_a = train_pinn(which="a", steps=4000, lr=2e-3, lam_bc=20.0,
                         seed=1, log_every=50, project="aprox-funciones", run_name="edo-a")
    plot_and_log("a", model_a, xs)
    wandb.finish()

    tf.keras.backend.clear_session()

   
    model_b = train_pinn(which="b", steps=4000, lr=2e-3, lam_bc=20.0,
                         seed=2, log_every=50, project="aprox-funciones", run_name="edo-b")
    plot_and_log("b", model_b, xs)
    wandb.finish()

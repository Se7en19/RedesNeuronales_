""" Este programa recibe imagenes RGB y las 
    convierte a imagenes en escala de grises
    utilizando una red neuronal """

""" Se utiliza un pequeño conjunto del dataset MNIST el cual contiene imagenes de numeros
    escritos a mano a color. 
    

    ACCESO AL DATASET: https://www.kaggle.com/datasets/youssifhisham/colored-mnist-dataset/data
    """

""" Importamos librerias"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer
import math    
import os 
import cv2
import random 

# Cargamos los datos
def load_images_from_folder(Datadir,class_names):
    X_train=[]
    y_train=[]
    for class_name in class_names:
        path=os.path.join(Datadir,class_name) #path to our class_name
        class_num=class_names.index(class_name) #converting string class_name to numerical class_num
        for img in os.listdir(path):
            img_array=cv2.imread(os.path.join(path,img))
            X_train.append(img_array)
            y_train.append(class_num)
    return X_train , y_train

def draw_samples(X_train,y_train,class_names):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i])
   
        plt.xlabel(class_names[y_train[i]])
    plt.show()



#loading trainning dataset
Datadir=r"/home/relka/Escritorio/RedesNeuronales_/Tarea4/Dataset_MNIST_Colored/colorized-MNIST-master/training"
class_names = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']
X,y = load_images_from_folder(Datadir,class_names)

#shuffling
S=c = list(zip(X, y))
random.shuffle(S)
X, y = zip(*c)

X=np.array(X)
y=np.array(y)


#visualize some training samples
print(y.shape)
print(X.shape)
draw_samples(X,y,class_names)
        
class RGB2Gray(Layer):
    def __init__(self, **kwargs):
        super(RGB2Gray, self).__init__(**kwargs)

    def call(self, inputs):
        # Fórmula estándar: 0.299*R + 0.587*G + 0.114*B
        gray = tf.image.rgb_to_grayscale(inputs)
        return gray

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)

# Crear instancia de la capa
rgb2gray_layer = RGB2Gray()

# Tomamos algunas imágenes del dataset
sample_imgs = X[:5]  # primeras 5 imágenes

# Normalizamos a rango [0,1] si es necesario
sample_imgs = sample_imgs.astype("float32") / 255.0  

# Pasamos por la capa
gray_imgs = rgb2gray_layer(sample_imgs).numpy()

# Dibujar las imágenes originales y sus grises
plt.figure(figsize=(10,4))
for i in range(5):
    # Original
    plt.subplot(2,5,i+1)
    plt.imshow(sample_imgs[i])
    plt.axis("off")
    plt.title("RGB")
    
    # Escala de grises
    plt.subplot(2,5,i+6)
    gray3 = np.repeat(gray_imgs[i], 3, axis=-1)  # (h,w,3)
    plt.imshow(gray3)
    plt.axis("off")
    plt.title("Gris")

plt.show()
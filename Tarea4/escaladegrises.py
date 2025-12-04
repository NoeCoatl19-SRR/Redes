import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

## Aquí diseñamos la capa con la clase "Capa Grises" heredada de keras.layers.Layer
class CapaGrises(keras.layers.Layer):
    def __init__(self):
        super(CapaGrises,self).__init__() # Lamamos al constructor de la clase padre, 
                                          # pasando los argumentos necesarios

    def call(self, inputs): # En call definimos los pasos que va a ejecutar la capa al recibir datos
        # En este caso la capa espera datos en forma de tensor float.
        # Los inputs tienen la forma (batch, d1, d2, 3)
        # donde batch son el número de muestras.
        # d1 y d2 son las dimensiones de la imagen (32x32) y 3 son los canales rgb.

        # Extraemos canales R, G, B
        r = inputs[:, :, :, 0]
        g = inputs[:, :, :, 1]
        b = inputs[:, :, :, 2]
        # Aplicamos la fórmula de conversión a grises (luminancia)
        grises = 0.299 * r + 0.587 * g + 0.114 * b
        # Expandimos las dimensiones de (batch,d1,d2) a (batch,d1,d2,1)
        grises = tf.expand_dims(grises, axis=-1)
        return grises
    
# Cargamos el dataset de CIFAR-10
dataset=cifar10.load_data()
(x_train, _), _ = dataset
# Tomamos una sola imagen de 32x32 y normalizamos a [0,1] el valor de los pixeles
img_rgb = x_train[10].astype(np.float32) / 255.0

# Expandimos la dimensión para obtener el batch: (32, 32, 3) -> (1, 32, 32, 3)
img_rgb_batch = tf.expand_dims(img_rgb, axis=0)

# Aplicamos la capa a la imagen
layer = CapaGrises()
output = layer(img_rgb_batch)


# Ahora graficamos y comparamos las imágenes
plt.figure(figsize=(8, 4))


plt.subplot(1, 2, 1)
plt.title("Imagen original RGB")
plt.imshow(img_rgb)
plt.axis('off')


output_np = output.numpy()[0, ..., 0]  # Convertimos el tensor en un array de numpy
plt.subplot(1, 2, 2)
plt.title("Imagen en escala de Grises")
plt.imshow(output_np, cmap='grey')
plt.axis('off')

plt.show()
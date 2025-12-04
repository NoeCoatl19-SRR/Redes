"""
Este código es un script simple para cargar los datos de MNIST, crear una red neuronal, 
entrenarla con SGD y guardar la red entrenada en un archivo pickle.
"""

# Importamos los módulos necesarios
import mnist_loader
import network
import pickle

# Cargamos los datos de entrenamiento, validación y prueba usando la función wrapper
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Convertimos los datos de entrenamiento y prueba a listas para su uso en la red
training_data = list(training_data)
test_data = list(test_data)

# Creamos la red neuronal con capas de 784 neuronas de entrada, 15 ocultas y 10 de salida
net = network.Network([784, 15, 10])

# Entrenamos la red con SGD: 10 épocas, mini batches de 5, tasa de aprendizaje 0.1, usando test_data para evaluación
net.SGD(training_data, 10, 5, 0.1, test_data=test_data)  # Épocas=10, Mini_batch_size=5, learning rate=0.1

# Salimos del script
exit()
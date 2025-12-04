"""
mnist_loader
~~~~~~~~~~~~

El proposito de este código es cargar los datos de MNIST, que son imagenes de numeros escritos a mano. 
Para detalles de las estructuras devueltas, mira los comentarios de load_data y load_data_wrapper. 
En practica, load_data_wrapper es la que se usa usualmente en el código de redes neuronales.
"""

#### Librerías
import pickle
import gzip
import numpy as np

def load_data():
    """
    Devuelve los datos de MNIST como tupla con training_data, validation_data y test_data.
    Donde training_data es una tupla con dos entradas: las imagenes  y los números.
    El validation_data y test_data son similares, pero con 10,000 imagenes cada uno.

    Para redes neuronales, es util modificar training_data un poco, lo que hace load_data_wrapper
    al aplanar las imagenes para su fácil procesamiento. 
    """
    f = gzip.open(r"C:\VSCodeCodigos\Redes Neuronales\Tarea1\mnist.pkl.gz", 'rb')
    training_data, validation_data, test_data = pickle.load(f,encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    Devuelve (training_data, validation_data, test_data) en formato conveniente para redes neuronales.
    training_data: lista de 50,000 tuplas (x, y), donde x es arreglo 784x1 con la imagen, 
    y es vector 10x1 one-hot para el digito.

    validation_data y test_data: listas de 10,000 tuplas (x, y), con y como entero del digito.
    es decir que no utilizamos la tecnica de one-hot encoding .
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """
    Devuelve vector unitario de 10 dimensiones con 1.0 en posición j (para el valor uno tenemos j=1)
    y con ceros en el resto de valores. 
    Convierte digito (0-9) en salida deseada para la red.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
#Primer codigo para entrenar la red neuronal
#Este codigo esta basado en el repositorio de MichalDanielDobrzanski
#librerias a utilizar 
import mnist_loader
import network
import pickle
#Parte del codigo que permite importar el archivo de imagenes para entrenar
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
#Damos el mensaje que se cargaron los datos
print(" Datos cargados correctamente")

#Parte del codigo que permite entrenar a la red
net = network.Network([784, 30, 10])
net.SGD(training_data, 15, 10, 0.5, test_data=test_data)
#guardamos nuestra red
with open("red_prueba.pkl", "wb") as f:
    pickle.dump(net, f)
#damos el mensaje que despues de todas las epocas la red ha sido entrenada
print("Entrenamiento terminado y red guardada en red_prueba.pkl")
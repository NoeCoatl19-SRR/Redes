import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt


loss_tracker = keras.metrics.Mean(name="loss")

class Polinomios(keras.Model):
    @property
    def metrics(self):
        return [loss_tracker] #igual cambia el loss_tracker

    def train_step(self, data):
        batch_size =50 #Calibra la resolucion de la ec.dif
        x = tf.random.uniform((batch_size,), minval=-1, maxval=1)
        eq = tf.math.cos(2*x) 

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            mse_loss = keras.losses.MeanSquaredError()
            loss = mse_loss(eq, y_pred)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        #actualiza metricas
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}
    
class Polinomio_Capa(tf.keras.layers.Layer):
    def __init__(self, max_potencia=3):
        super(Polinomio_Capa,self).__init__()
        self.max_potencia = max_potencia # Definimos la potencia máxima a usar

        self.kernel = self.add_weight(
        shape=[self.max_potencia+1])  # inicializamos un vector de pesos de tamaño 3+1
                                      # estos representan los coeficientes del polinomio

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        if (inputs.shape == ()):
            inputs=(inputs,)
        elif (len(inputs.shape)==1):
            inputs=tf.expand_dims(inputs, axis=1)
        # Creamos un tensor para las potencias [0,1,2,3]
        potencias = tf.range(0, self.max_potencia + 1, dtype=tf.float32) 
        potencias = tf.expand_dims(potencias, axis=0) #Expandimos sus dimensiones 
        x_potencias = tf.pow(inputs, potencias)  #Calculamos los inputs elevados a 0,1,2,3

        return tf.tensordot(x_potencias,self.kernel,axes=[[-1], [0]]) # Calculamos el polinomio de grado 3 con los coeficientes obtenidos
    

inputs = keras.Input(shape=(1,))
print(inputs)
x = Polinomio_Capa(3)(inputs)
model = Polinomios(inputs=inputs,outputs=x)
model.summary()


model.compile(optimizer=Adam(learning_rate=0.02), metrics=['loss'])

x=tf.linspace(-1,1,100)
history = model.fit(x,epochs=100,verbose=1)


x_testv = tf.linspace(-1,1,100)
a=model.predict(x_testv)

plt.plot(x_testv,a,color='blue')
plt.plot(x_testv, tf.math.cos(2 * x_testv),color='red' )
plt.show()

#### Esta parte es para graficar la perdida en los datos de entrenamiento"
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
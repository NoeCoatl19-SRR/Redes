import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np

class ODEsolver(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = tf.keras.losses.MeanSquaredError()

    @property
    def metrics(self):
      return [self.loss_tracker]

    def train_step(self, data):
         batch_size = tf.shape(data)[0]
         min = tf.cast(tf.reduce_min(data),tf.float32)
         max = tf.cast(tf.reduce_max(data),tf.float32)
         x = tf.random.uniform((batch_size,1), minval=min, maxval=max)

         with tf.GradientTape(persistent=True) as tape:
             tape.watch(x)  # Esta línea es para obtener 2ª derivada respecto a x y evitar errores
             with tf.GradientTape() as tape2:
                 tape2.watch(x)
                 y_pred = self(x, training=True)
             dy = tape2.gradient(y_pred, x) #derivada del modelo con respecto a entradas x
             ydd = tape.gradient(dy, x) #derivada segunda del modelo con respecto a entradas x
             x_o = tf.zeros((batch_size,1)) #valor de x en condicion inicial x_0=0

             # Evaluamos y(x0) y y'(x0) para imponer condiciones iniciales en la ec.dif de segundo orden
             # usamos un tape separado para obtener la derivada en x_o
             with tf.GradientTape() as tape_ic:
                 tape_ic.watch(x_o)
                 y_o = self(x_o, training=True)
             dy_o = tape_ic.gradient(y_o, x_o)

             y_o = self(x_o,training=True) #valor del modelo en en x_0
             eq = ydd + y_pred #Ecuacion diferencial evaluada en el modelo. Queremos que sea muy pequeno
             ic = 1. #valor que queremos para la condicion inicial o el modelo en x_0
             # condiciones iniciales: y(0)=1, y'(0)=-0.5
             loss = self.mse(0., eq) + self.mse(y_o,ic) + self.mse(dy_o, -0.5)

        # Apply grads
         grads = tape.gradient(loss, self.trainable_variables)
         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
         self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
         return {"loss": self.loss_tracker.result()}

model = ODEsolver()

model.add(Dense(50, activation='tanh', input_shape=(1,)))
model.add(Dense(50, activation='tanh'))
model.add(Dense(1))


model.summary()


model.compile(optimizer=Adam(learning_rate=0.005),metrics=['loss'])

x=tf.linspace(-5,5,100)
history = model.fit(x,epochs=400,verbose=1)

x_testv = tf.linspace(-5,5,100)
a=model.predict(x_testv)
plt.plot(x_testv,a,label="aprox")
plt.plot(x_testv,(np.cos(x_testv) - 0.5*np.sin(x_testv)),label="exact")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
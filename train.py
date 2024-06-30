import numpy as np
import tensorflow as tf

# Simple dataset
X = np.random.rand(100, 3)
y = np.random.rand(100, 1)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(3,))
])

model.compile(optimizer='sgd', loss='mse')

# Training the model and capturing weights
class WeightHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append(self.model.get_weights())

weight_history = WeightHistory()
model.fit(X, y, epochs=100, callbacks=[weight_history])

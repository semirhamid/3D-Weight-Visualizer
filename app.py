from flask import Flask, render_template, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

class WeightHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append([w.tolist() for w in self.model.get_weights()])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/weights')
def weights():
    return jsonify(weight_history.weights)

if __name__ == '__main__':
    # Generate a simple dataset
    X = np.random.rand(100, 3)
    y = np.random.rand(100, 1)

    # Define a simple one-layer model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(3,))
    ])

    model.compile(optimizer='sgd', loss='mse')

    # Capture weight updates during training
    weight_history = WeightHistory()
    model.fit(X, y, epochs=100, callbacks=[weight_history])

    # Start the Flask server
    app.run(debug=True)

import tensorflow as tf
import numpy as np
import pandas as pd
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.99:
            print('\nReached 99% Accuracy, Cancelling Training\n')
            self.model.stop_training = True

class MNIST:
    def __init__(self):
        self.mnist = tf.keras.datasets.mnist
    def loadData(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.mnist.load_data()
    def normalize(self):
        self.X_train , self.X_test = self.X_train / 255.0, self.X_test / 255.0
    def model_train(self):

        callback = MyCallback()
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
        model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

        model.compile(optimizer=tf.optimizers.Adam(),
                      loss = 'sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(self.X_train, self.y_train, epochs=10, callbacks=[callback])

        model.evaluate(self.X_test, self.y_test)

        pred = model.predict(self.X_test)
        print(self.y_test[13])
        y = pred[13].tolist()
        print(y.index(max(y)))


if __name__ == "__main__":
    m = MNIST()
    m.loadData()
    m.normalize()
    m.model_train()


# Lint as: python3
"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for mnist.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of mnist.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import context
from tensorflow.keras import datasets,optimizers,layers


class AE(keras.Model):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(20)])
        self.decoder = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(784)])

    def call(self, inputs, training=None):
        h = self.encoder(inputs)
        x_hat = self.decoder(h)
        return x_hat


def create_db():
    (x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
    x = tf.cast(x, dtype=tf.float32) / 255.0
    x = tf.reshape(x, [-1, 784])
    x_test = tf.cast(x_test, dtype=tf.float32) / 255.0
    x_test = tf.reshape(x_test, [-1, 784])
    train_db = tf.data.Dataset.from_tensor_slices(x)
    train_db = train_db.shuffle(1000).batch(128)
    test_db = tf.data.Dataset.from_tensor_slices(x_test)
    test_db = test_db.batch(128)
    return train_db, test_db


def main():
    train_db, test_db = create_db()
    model = AE()
    optimizer = optimizers.Adam(lr=0.0001)
    for epoch in range(100):
        for step, x in enumerate(train_db):
            with tf.GradientTape() as tape:
                x_rec = model(x)
                rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec)
                rec_loss = tf.reduce_mean(rec_loss)
            grads = tape.gradient(rec_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, rec_loss.numpy())



if __name__ == '__main__':
    main()

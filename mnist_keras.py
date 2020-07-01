# Lint as: python3
"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for mnist.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of mnist.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,optimizers,layers


class mnist_model(object):
  def __init__(self):
    self.network = keras.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10)])
    self.network.compile(optimizer=optimizers.Adam(lr=0.01),
                         loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])


def create_db(x, y):
  def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.0
    x = tf.reshape(x, [-1, 784])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y

  db = tf.data.Dataset.from_tensor_slices((x, y))
  db = db.shuffle(10000).batch(128)
  db = db.map(preprocess)
  return db


def train(train_db, eval_db):
  model = mnist_model()
  history = model.network.fit(train_db, epochs=10, validation_data=eval_db,
                              validation_freq=5)
  print(model.network.summary())
  print(history.history)


def main():
  (x, y), (x_test, y_test) = datasets.mnist.load_data()
  print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)

  train_db = create_db(x, y)
  eval_db = create_db(x_test, y_test)

  train(train_db, eval_db)


if __name__ == '__main__':
  main()

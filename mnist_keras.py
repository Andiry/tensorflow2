# Lint as: python3
"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for mnist.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of mnist.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import context
from tensorflow.keras import datasets,optimizers,layers


class mnist_dense_model(object):
  def __init__(self):
    self.network = keras.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dropout(rate=0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10)])
    self.network.compile(optimizer=optimizers.Adam(lr=0.01),
                         loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])


class mnist_conv2d_model(object):
  def __init__(self):
    self.network = keras.Sequential([
        layers.Conv2D(6, kernel_size=3, strides=1),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.ReLU(),
        layers.Conv2D(16, kernel_size=3, strides=1),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)])
    self.network.compile(optimizer=optimizers.Adam(lr=0.01),
                         loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])


def create_db(x, y, using_conv2d):
  def preprocess_conv2d(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.0
    x = tf.expand_dims(x, axis=3)
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y

  def preprocess_dense(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.0
    x = tf.reshape(x, [-1, 784])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y

  db = tf.data.Dataset.from_tensor_slices((x, y))
  db = db.shuffle(10000).batch(128)
  if using_conv2d:
    db = db.map(preprocess_conv2d)
  else:
    db = db.map(preprocess_dense)

  return db


def train(train_db, eval_db, using_conv2d):
  if using_conv2d:
    model = mnist_conv2d_model()
  else:
    model = mnist_dense_model()
  context.enable_run_metadata()
  history = model.network.fit(train_db, epochs=10, validation_data=eval_db,
                              validation_freq=5)
  run_metadata = context.export_run_metadata()
  context.disable_run_metadata()
  print(model.network.summary())
  print(history.history)
  tf.saved_model.save(model.network, 'saved_model')
  print("StepStats: ", run_metadata.step_stats)


def main():
  (x, y), (x_test, y_test) = datasets.mnist.load_data()
  print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)

  using_conv2d = True
  train_db = create_db(x, y, using_conv2d=using_conv2d)
  eval_db = create_db(x_test, y_test, using_conv2d=using_conv2d)

  train(train_db, eval_db, using_conv2d=using_conv2d)


if __name__ == '__main__':
  main()

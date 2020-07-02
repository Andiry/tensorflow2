# Lint as: python3
"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for mnist.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of mnist.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import context
from tensorflow.keras import datasets,optimizers,layers


class vgg13_model(object):
  def __init__(self):
    self.network = keras.Sequential([
        layers.Conv2D(64, kernel_size=3, padding='same', strides=1, activation='relu'),
        layers.Conv2D(64, kernel_size=3, padding='same', strides=1, activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),

        layers.Conv2D(128, kernel_size=3, padding='same', strides=1, activation='relu'),
        layers.Conv2D(128, kernel_size=3, padding='same', strides=1, activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),

        layers.Conv2D(256, kernel_size=3, padding='same', strides=1, activation='relu'),
        layers.Conv2D(256, kernel_size=3, padding='same', strides=1, activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),

        layers.Conv2D(512, kernel_size=3, padding='same', strides=1, activation='relu'),
        layers.Conv2D(512, kernel_size=3, padding='same', strides=1, activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),

        layers.Conv2D(512, kernel_size=3, padding='same', strides=1, activation='relu'),
        layers.Conv2D(512, kernel_size=3, padding='same', strides=1, activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)])
    self.network.compile(optimizer=optimizers.Adam(lr=0.0001),
                         loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])


def create_db(x, y):
  def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.0
    y = tf.squeeze(y, axis=1)
    y = tf.one_hot(y, depth=10)
    return x, y

  db = tf.data.Dataset.from_tensor_slices((x, y))
  db = db.shuffle(10000).batch(128)
  db = db.map(preprocess)
  return db


def train(train_db, eval_db):
  model = vgg13_model()
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
  (x, y), (x_test, y_test) = datasets.cifar10.load_data()
  print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)

  train_db = create_db(x, y)
  eval_db = create_db(x_test, y_test)

  train(train_db, eval_db)


if __name__ == '__main__':
  main()

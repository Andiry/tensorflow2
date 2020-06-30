# Lint as: python3
"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for mnist.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of mnist.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets


class mnist_model(object):
  def __init__(self):
    self.w1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1))
    self.b1 = tf.Variable(tf.zeros([256]))
    self.w2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1))
    self.b2 = tf.Variable(tf.zeros([128]))
    self.w3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1))
    self.b3 = tf.Variable(tf.zeros([10]))
    self.lr = 0.01

  def train(self, x, y):
    with tf.GradientTape() as tape:
      h1 = tf.nn.relu(x @ self.w1 + self.b1)
      h2 = tf.nn.relu(h1 @ self.w2 + self.b2)
      out = h2 @ self.w3 + self.b3
      loss = keras.losses.categorical_crossentropy(y, out, from_logits=True)
      loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
    self.w1.assign_sub(self.lr * grads[0])
    self.b1.assign_sub(self.lr * grads[1])
    self.w2.assign_sub(self.lr * grads[2])
    self.b2.assign_sub(self.lr * grads[3])
    self.w3.assign_sub(self.lr * grads[4])
    self.b3.assign_sub(self.lr * grads[5])
    return out, loss

  def eval(self, eval_db):
    total = 0
    total_correct = 0
    for (x, y) in eval_db:
      h1 = tf.nn.relu(x @ self.w1 + self.b1)
      h2 = tf.nn.relu(h1 @ self.w2 + self.b2)
      out = tf.nn.softmax(h2 @ self.w3 + self.b3, axis=1)

      pred = tf.argmax(out, axis=1)
      y = tf.argmax(y, axis=1)
      correct = tf.equal(pred, y)
      total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
      total += tf.size(y).numpy()

    accuracy = total_correct * 1.0 / total
    print('Evaluate accuracy:', accuracy)


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
  for epoch in range(20):
    for step, (x, y) in enumerate(train_db):
      out, loss = model.train(x, y)
      if step % 100 == 0:
        print('Step ', step, ' loss: ', loss.numpy())
      if step % 500 == 0:
        model.eval(eval_db)


def main():
  (x, y), (x_test, y_test) = datasets.mnist.load_data()
  print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)

  train_db = create_db(x, y)
  eval_db = create_db(x_test, y_test)

  train(train_db, eval_db)


if __name__ == '__main__':
  main()

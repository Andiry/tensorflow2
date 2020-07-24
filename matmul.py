import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import context
from tensorflow.keras import datasets,optimizers,layers,losses


class MatMulModel(keras.Model):
    def __init__(self, dimension):
        super(MatMulModel, self).__init__()
        self.b = tf.random.uniform([dimension, dimension], dtype=tf.float16)

    @tf.function
    def call(self, inputs, training=None):
        x = inputs
        x = x @ self.b
        x = x @ self.b
        x = x @ self.b
        x = x @ self.b
        x = x @ self.b
        return x


def main():
    dimension = 1024
    x = tf.random.uniform([dimension, dimension], dtype=tf.float16)

    model = MatMulModel(dimension)
    model.build(input_shape=[dimension, dimension])
    model.call(x)
    print(model.summary())


if __name__ == '__main__':
    main()

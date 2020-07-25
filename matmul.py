import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import context
from tensorflow.keras import datasets,optimizers,layers,losses


class MatMulModel(keras.Model):
    def __init__(self, dimension):
        super(MatMulModel, self).__init__()
#        self.w = tf.random.uniform([dimension, dimension], dtype=tf.float16)
        self.w = tf.zeros([dimension, dimension], dtype=tf.float16)

    @tf.function
    def call(self, inputs, training=None):
        x = tf.cast(inputs, dtype=tf.float16)
        x = x @ self.w
        x = x @ self.w
        x = x @ self.w
        x = x @ self.w
        x = x @ self.w
        return x


def main():
    dim = 2048
    x = tf.random.uniform([dim, dim], dtype=tf.float16)

    model = MatMulModel(dim)

    logdir = '/tmp/matmul'
    summary_writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(profiler=True)
    with summary_writer.as_default():
        z = model.call(x)
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)

    print(z.numpy())

if __name__ == '__main__':
    main()

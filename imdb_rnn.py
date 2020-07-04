import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import context
from tensorflow.keras import datasets,optimizers,layers,losses


bs = 128
total_words = 10000
max_len = 80
embedding_len = 100


def create_dataset():
    (x, y), (x_test, y_test) = keras.datasets.imdb.load_data(num_words = total_words)
    x = keras.preprocessing.sequence.pad_sequences(x, maxlen = max_len)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen = max_len)

    db_train = tf.data.Dataset.from_tensor_slices((x, y))
    db_train = db_train.shuffle(1000).batch(bs, drop_remainder=True)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.batch(bs, drop_remainder=True)

    print('x_train shape: ', x.shape)
    print('x_test shape: ', x_test.shape)
    return db_train, db_test


class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        # [b, 64]
        self.state0 = [tf.zeros([bs, units])]
        self.state1 = [tf.zeros([bs, units])]
        # [b, 80] -> [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                input_length = max_len)
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout = 0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout = 0.5)
        # [b, 80, 100] -> [b, 64] -> [b, 1]
        self.out_layer = layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs # [b, 80]
        x = self.embedding(x) # [b, 80, 100]
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1): # [b, 100]
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
        x = self.out_layer(out1)
        prob = tf.sigmoid(x)
        return prob


def main():
    db_train, db_test = create_dataset()
    units = 64
    epochs = 20

    model = MyRNN(units)
    model.compile(optimizer = optimizers.Adam(0.001),
            loss = losses.BinaryCrossentropy(),
            metrics=['accuracy'])
    model.fit(db_train, epochs = epochs, validation_data = db_test,
            validation_freq = 5)
    model.evaluate(db_test)
    print(model.summary())


if __name__ == '__main__':
    main()



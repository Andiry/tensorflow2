import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds


embedding_layer = layers.Embedding(1000, 5)

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k', 
    split = (tfds.Split.TRAIN, tfds.Split.TEST), 
    with_info=True, as_supervised=True)

encoder = info.features['text'].encoder
train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes=([None], [None]))
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes=([None], [None]))

embedding_dim = 16
model = keras.Sequential([
    layers.Embedding(encoder.vocab_size, embedding_layer),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.summary()
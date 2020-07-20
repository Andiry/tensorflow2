# Lint as: python3
"""TODO(andiryxu): DO NOT SUBMIT without one-line documentation for mnist.

TODO(andiryxu): DO NOT SUBMIT without a detailed description of mnist.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import context
from tensorflow.keras import datasets,optimizers,layers


class BasicBlock(layers.Layer):
  def __init__(self, filter_num, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = layers.Conv2D(filter_num, (3,3), strides=stride, padding='same')
    self.bn1 = layers.BatchNormalization()
    self.relu = layers.ReLU()
    
    self.conv2 = layers.Conv2D(filter_num, (3,3), strides=1, padding='same')
    self.bn2 = layers.BatchNormalization()

    if stride != 1:
      self.downsample = keras.Sequential()
      self.downsample.add(layers.Conv2D(filter_num, (1,1), strides=stride))
    else:
      self.downsample = lambda x:x

  def call(self, inputs, training=None):
    # [b, h, w, c]
    out = self.conv1(inputs)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    identity = self.downsample(inputs)
    output = layers.add([out, identity])
    output = self.relu(output)
    return output


class ResNet(keras.Model):
  def build_resblock(self, filter_num, blocks, stride=1):
    res_blocks = keras.Sequential()
    res_blocks.add(BasicBlock(filter_num, stride))

    for _ in range(1, blocks):
      res_blocks.add(BasicBlock(filter_num, stride=1))

    return res_blocks

  def __init__(self, layer_dims, num_classes=10):
    super(ResNet, self).__init__()
    self.stem = keras.Sequential([
        layers.Conv2D(64, kernel_size=3, strides=1),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=2, strides=1, padding='same')])

    self.layer1 = self.build_resblock(64, layer_dims[0])
    self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
    self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
    self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

    self.avgpool = layers.GlobalAveragePooling2D()
    self.fc = layers.Dense(num_classes)
    self.compile(optimizer=optimizers.Adam(lr=0.0001),
                 loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

  def call(self, inputs, training=None):
    x = self.stem(inputs)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = self.fc(x)
    return x


def resnet18():
  return ResNet([2, 2, 2, 2])

def resnet34():
  return ResNet([3, 4, 6, 3])


def create_db(strategy, x, y):
  def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.0
    y = tf.squeeze(y, axis=1)
    y = tf.one_hot(y, depth=10)
    return x, y

  db = tf.data.Dataset.from_tensor_slices((x, y))
  batch_size = 128 * strategy.num_replicas_in_sync
  db = db.shuffle(10000).batch(batch_size)
  db = db.map(preprocess)
  return db


def train(strategy, train_db, eval_db):
  with strategy.scope():
    model = resnet34()
    context.enable_run_metadata()
    history = model.fit(train_db, epochs=10, validation_data=eval_db,
                        validation_freq=5)
    run_metadata = context.export_run_metadata()
    context.disable_run_metadata()
    print(model.summary())
    print(history.history)
#    tf.saved_model.save(model, 'saved_model')
    print("StepStats: ", run_metadata.step_stats)


def main():
  (x, y), (x_test, y_test) = datasets.cifar10.load_data()
  print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)

  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: %d' % strategy.num_replicas_in_sync)

  train_db = create_db(strategy, x, y)
  eval_db = create_db(strategy, x_test, y_test)

  train(strategy, train_db, eval_db)


if __name__ == '__main__':
  main()

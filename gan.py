import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import context
from tensorflow.keras import datasets,optimizers,layers,losses
import glob
import numpy as np
from dataset import make_anime_dataset
from scipy.misc import toimage


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        filter = 64
        self.conv1 = layers.Conv2DTranspose(filter * 8, 4, 1, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(filter * 4, 4, 2, 'same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(filter * 2, 4, 2, 'same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2DTranspose(filter * 1, 4, 2, 'same', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2DTranspose(3, 4, 2, 'same', use_bias=False)

    def call(self, inputs, training=None):
        x = inputs # [b, 100]
        x = tf.reshape(x, (x.shape[0], 1, 1, x.shape[1])) # [b, 1, 1, 100]
        x = tf.nn.relu(x)
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training)) # [b, 4, 4, 512]
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training)) # [b, 8, 8, 256]
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training)) # [b, 16, 16, 128]
        x = tf.nn.relu(self.bn4(self.conv4(x), training=training)) # [b, 32, 32, 64]
        x = self.conv5(x)
        x = tf.tanh(x)
        return x # [b, 64, 64, 3]


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        filter = 64
        self.conv1 = layers.Conv2D(filter, 4, 2, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filter * 2, 4, 2, 'valid', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(filter * 4, 4, 2, 'valid', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(filter * 8, 3, 1, 'valid', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2D(filter * 16, 3, 1, 'valid', use_bias=False)
        self.bn5 = layers.BatchNormalization()
        self.pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        # [b, 64, 64, 3]
        x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training)) # [b, 31, 31, 64]
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training)) # [b, 14, 14, 128]
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training)) # [b, 6, 6, 256]
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training)) # [b, 4, 4, 512]
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training)) # [b, 2, 2, 1024]
        x = self.pool(x) # [b, 1, 1024]
        x = self.fc(self.flatten(x)) # [b, 1]
        return x


def d_loss_fn(generator, discriminator, bs_z, bs_x, is_training):
    fake_image = generator(bs_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(bs_x, is_training)
    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    loss = d_loss_real + d_loss_fake
    return loss


def celoss_ones(logits):
    y = tf.ones_like(logits)
    loss = losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    y = tf.zeros_like(logits)
    loss = losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def g_loss_fn(generator, discriminator, bs_z, is_training):
    fake_image = generator(bs_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)
    return loss


def create_dataset(bs):
    img_path = glob.glob(r'data/faces/*.jpg')
    print('images: ', len(img_path))
    dataset, img_shape, _ = make_anime_dataset(img_path, bs, resize=64)
    print(dataset, img_shape)
    dataset = dataset.repeat(100)
    db_iter = iter(dataset)
    return db_iter


def post_result(val_out, val_block_size, img_path, color_mode):
    def preprocess(img):
        imge = ((img + 1.0) * 127.5).astype(np.uint8)

    preprocessed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        if single_row.size == 0:
            single_row = preprocessed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocessed[b, :, :, :]), axis=1)

        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    toimage(final_image).save(img_path)


def main():
    z_dim = 100
    epochs = 3000000
    bs = 64
    lr = 0.0002
    is_training = True
    generator = Generator()
    discriminator = Discriminator()
    g_optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.5)
    d_optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.5)

    db_iter = create_dataset(bs)

    for epoch in range(epochs):
        for _ in range(5):
            bs_z = tf.random.normal([bs, z_dim])
            bs_x = next(db_iter)
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, bs_z, bs_x, is_training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        bs_z = tf.random.normal([bs, z_dim])
        bs_x = next(db_iter)
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, bs_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print('Epoch ', epoch, ', d-loss: ', d_loss.numpy(), ' g-loss: ', g_loss.numpy())
            z = tf.random.normal([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('gan-images', 'gan-%d.png' % epoch)
            post_result(fake_image.numpy(), 10, img_path, color_mode='P')
            
        if epoch % 10000 == 1:
            generator.save_weights('generator.ckpt')
            discriminator.save_weights('discriminator.ckpt')


if __name__ == '__main__':
    main()

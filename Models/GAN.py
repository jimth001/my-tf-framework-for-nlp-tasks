# -*- coding: utf-8 -*-
"""
Simple GAN
Test TensorFlow 2.0
"""
from __future__ import absolute_import, print_function, division
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import imageio
import glob
import time

# tf.config.gpu.set_per_process_memory_growth(enabled=True)

# hpy
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 5000
z_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, z_dim])

# load data
(train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()

plt.imshow(train_images[0])
plt.show()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# generator
def make_generator():
    generator = keras.Sequential([
        keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Reshape((7, 7, 256)),
        keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'),
    ])

    return generator


# test
g = make_generator()
z = tf.random.normal([1, 100])
fake_image = g(z, training=False)
plt.imshow(fake_image[0, :, :, 0], cmap='gray')


def make_discriminator():
    discriminator = keras.Sequential([
        keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(2, activation='softmax'),

    ])

    return discriminator


d = make_discriminator()
pred = d(fake_image)
print('pred score is: ', pred)

# loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_optimizer = keras.optimizers.Adam(1e-4)
d_optimizer = keras.optimizers.Adam(1e-4)


# loss function
def generator_loss(fake_iamge):
    fake_iamge = fake_iamge[:, 1]
    return tf.keras.losses.mean_squared_error(tf.ones_like(fake_iamge), fake_iamge)


def discriminator_loss(fake_iamge, real_iamge):
    fake_iamge = fake_iamge[:, 1]
    real_iamge = real_iamge[:, 1]
    # real_target=tf.ones_like(real_iamge)
    # fake_target=tf.zeros_like(fake_iamge)
    real_loss = tf.keras.losses.mean_squared_error(tf.ones_like(real_iamge), real_iamge)
    fake_loss = tf.keras.losses.mean_squared_error(tf.zeros_like(fake_iamge), fake_iamge)
    # tf.math.log()
    # real_loss = cross_entropy(tf.ones_like(real_iamge), real_iamge)
    # fake_loss = cross_entropy(tf.zeros_like(fake_iamge), fake_iamge)
    return real_loss + fake_loss


# checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(g_optimizer=g_optimizer,
                                 d_optimizer=d_optimizer,
                                 g=g,
                                 d=d)


# traing
@tf.function
def train_one_step(images):
    z = tf.random.normal([BATCH_SIZE, z_dim])

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_images = g(z, training=True)

        real_pred = d(images, training=True)
        fake_pred = d(fake_images, training=True)

        g_loss = generator_loss(fake_pred)
        d_loss = discriminator_loss(real_iamge=real_pred, fake_iamge=fake_pred)

    g_gradients = g_tape.gradient(g_loss, g.trainable_variables)
    d_gradients = d_tape.gradient(d_loss, d.trainable_variables)

    g_optimizer.apply_gradients(zip(g_gradients, g.trainable_variables))
    d_optimizer.apply_gradients(zip(d_gradients, d.trainable_variables))


@tf.function
def train_d_one_step(images):
    z = tf.random.normal([BATCH_SIZE, z_dim])
    with tf.GradientTape() as d_tape:
        fake_images = g(z, training=False)
        real_pred = d(images, training=True)
        fake_pred = d(fake_images, training=True)
        d_loss = discriminator_loss(real_iamge=real_pred, fake_iamge=fake_pred)
        d_gradients = d_tape.gradient(d_loss, d.trainable_variables)
        d_optimizer.apply_gradients(zip(d_gradients, d.trainable_variables))
        return d_loss


@tf.function
def train_g_one_step():
    z = tf.random.normal([BATCH_SIZE, z_dim])
    with tf.GradientTape() as g_tape:
        fake_images = g(z, training=True)
        fake_pred = d(fake_images, training=False)
        g_loss = generator_loss(fake_pred)
        g_gradients = g_tape.gradient(g_loss, g.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, g.trainable_variables))
        return g_loss


def train(dataset, epochs):
    train_mode_is_d = True
    losses = {'d': 100, 'g': 100}
    print_per_batch = 10
    batch_counter = 1
    for epoch in range(epochs):
        start = time.time()
        if train_mode_is_d:
            for image_batch in dataset:
                d_loss = train_d_one_step(image_batch)
                losses['d'] = d_loss.numpy()
                if batch_counter % print_per_batch == 0:
                    print('batch %d, d_loss:' % batch_counter, d_loss)
                batch_counter += 1
            if losses['d'] < 0.1:
                train_mode_is_d = False
        else:
            g_loss = train_g_one_step()
            losses['g'] = g_loss.numpy()
            print('batch %d, g_loss:' % batch_counter, g_loss)
            t = 0
            while losses['g'] > 0.1 or t < 100:
                g_loss = train_g_one_step()
                losses['g'] = g_loss.numpy()
                if t % 10 == 0:
                    print('batch %d, g_loss:' % batch_counter, g_loss)
                batch_counter += 1
                t += 1
                # if g_loss.numpy() < 0.1:
            train_mode_is_d = True
        # display.clear_output(wait = True)
        """if train_mode_is_d and losses['d'].numpy() < 0.3:
            train_mode_is_d = False
        elif train_mode_is_d:
            pass
        elif not train_mode_is_d and losses['g'].numpy() <0.3:
            train_mode_is_d = True
        else:
            pass"""
        generate_and_save_images(g, epoch + 1, seed)

        if (epoch + 1) % 100000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print('losses: ', losses)

    # diplay_clear_output(wait = True)
    generate_and_save_images(g, epochs, seed)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    plt.close(fig)


if __name__ == '__main__':
    train(train_dataset, EPOCHS)

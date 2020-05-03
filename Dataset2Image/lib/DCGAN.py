# @title Default title text
import tensorflow as tf

tf.__version__



import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from hyperopt import STATUS_OK
from hyperopt import tpe, hp, Trials, fmin
from IPython import display
from matplotlib import pyplot

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 64
optimizable_variable = {"batch": hp.choice("batch", [256]),
                        'dropout_rate': hp.uniform("dropout_rate", 0.3, 0.3),
                        'lr_initial_g': hp.uniform("lr_initial_g", 1e-4, 1e-4),
                        "lr_initial_d": hp.uniform("lr_initial_d", 1e-4, 1e-4),
                        "apply_label_smoothing": hp.choice("apply_label_smoothing", [0, 0]),
                        "label_noise": hp.choice("label_noise", [0, 0])
                        }
EPOCHS = 200
noise_dim = 100
num_examples_to_generate = 16
dropout_rate = 0.3

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.BatchNormalization())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1))

    return model


def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * int(y.shape[0]))
    # choose labels to flip
    flip_ix = np.random.choice([i for i in range(int(y.shape[0]))], size=n_select)

    op_list = []
    # invert the labels in place
    # y_np[flip_ix] = 1 - y_np[flip_ix]
    for i in range(int(y.shape[0])):
        if i in flip_ix:
            op_list.append(tf.subtract(1, y[i]))
        else:
            op_list.append(y[i])

    outputs = tf.stack(op_list)
    return outputs


def smooth_positive_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.5)


def smooth_negative_labels(y):
    return y + np.random.random(y.shape) * 0.3


def discriminator_loss(real_output, fake_output):
    real_output_noise = noisy_labels(tf.ones_like(real_output), 5)
    fake_output_noise = noisy_labels(tf.zeros_like(fake_output), 5)
    real_output_smooth = smooth_positive_labels(real_output_noise)
    fake_output_smooth = smooth_negative_labels(fake_output_noise)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    #real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    #fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    #total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    fake_output_smooth = smooth_negative_labels(tf.ones_like(fake_output))
    return cross_entropy(tf.ones_like(fake_output_smooth), fake_output)

    #return cross_entropy(tf.ones_like(fake_output), fake_output)


#@tf.function
def train_step(images, train_d, train_g):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    if train_d:
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    if train_g:
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss, disc_loss


def plot_history(d_hist, g_hist, step=0, is_global=False):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.plot(d_hist, label='d')
    pyplot.plot(g_hist, label='gen')
    pyplot.legend()
    pyplot.show()


def train(dataset, epochs):
    gen_ls = []
    disc_ls = []
    train_g = True
    train_d = True
    for epoch in range(epochs):
        start = time.time()
        gen = []
        disc = []
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, train_d, train_g)
            disc.append(disc_loss)
            gen.append(gen_loss)

        gen_ls.append(np.mean(gen))
        disc_ls.append(np.mean(disc))

        # Produce images for the GIF as we go
        display.clear_output(wait=True)

        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)
        plot_history(disc_ls, gen_ls)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print("GEN:" + str(train_g) + " DISC" + str(train_d))
        print("GEN:" + str(gen_ls[-1]) + " DISC" + str(disc_ls[-1]))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epochs,
                             seed)


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
    plt.show()


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

seed = tf.random.normal([num_examples_to_generate, noise_dim])

train(train_dataset, EPOCHS)

"""Restore the latest checkpoint."""

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""## Create a GIF"""


# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


display_image(EPOCHS)

"""Use `imageio` to create an animated gif using the images saved during training."""

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2 * (i ** 0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import IPython

if IPython.version_info > (6, 2, 0, ''):
    display.Image(filename=anim_file)

"""If you're working in Colab you can download the animation with the code below:"""

try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(anim_file)

"""## Next steps

This tutorial has shown the complete code necessary to write and train a GAN. As a next step, you might like to experiment with a different dataset, for example the Large-scale Celeb Faces Attributes (CelebA) dataset [available on Kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset). To learn more about GANs we recommend the [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160).
"""
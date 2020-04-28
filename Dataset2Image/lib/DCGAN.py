import pickle

import tensorflow as tf
from tensorflow.python.keras.initializers import TruncatedNormal

tf.__version__

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

leaky_relu_slope = 0.2
weight_init_std = 0.02
weight_init_mean = 0.0
dropout_rate = 0.5
lr_initial_d = 0.0002
lr_initial_g = 0.0002
lr_decay_steps = 1000
noise_dim = 100

weight_initializer = TruncatedNormal(stddev=weight_init_std, mean=weight_init_mean,
                                     seed=42)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 128, input_shape=(100,), kernel_initializer=weight_initializer))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 128)))
    model.add(layers.Conv2DTranspose(128, (5, 5),
                                     strides=(1, 1), padding="same",
                                     kernel_initializer=weight_initializer, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2DTranspose(64, (5, 5),
                                     strides=(2, 2), padding="same",
                                     kernel_initializer=weight_initializer, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2DTranspose(32, (5, 5),
                                     strides=(2, 2), padding="same",
                                     kernel_initializer=weight_initializer, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Dense(1, activation='tanh', kernel_initializer=weight_initializer))

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=False,
                            input_shape=[28, 28, 1],kernel_initializer=weight_initializer))
    model.add(layers.LeakyReLU(alpha=leaky_relu_slope))

    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_relu_slope))

    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_relu_slope))

    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_relu_slope))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


"""### Generator loss
The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, we will compare the discriminators decisions on the generated images to an array of 1s.
"""


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# @tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    print("gen loss: " + str(np.array(gen_loss)) + " | disc loss: " + str(np.array(disc_loss)))
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return np.array(gen_loss),np.array(disc_loss)

decay_step = 50


def train(dataset, epochs):
    all_gl = np.array([]);
    all_dl = np.array([])

    exp_replay = []
    for epoch in range(epochs):

        G_loss = [];
        D_loss = []

        start = time.time()
        new_lr_d = lr_initial_d
        new_lr_g = lr_initial_g
        global_step = 0

        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)
            global_step = global_step + 1
            G_loss.append(g_loss);
            D_loss.append(d_loss)
            all_gl = np.append(all_gl, np.array([G_loss]))
            all_dl = np.append(all_dl, np.array([D_loss]))

        # generate an extra image for each epoch and store it in memory for experience replay

        '''
        generated_image = dog_generator(tf.random.normal([1, noise_dim]), training=False)
        exp_replay.append(generated_image)
        if len(exp_replay) == replay_step:
            print('Executing experience replay..')
            replay_images = np.array([p[0] for p in exp_replay])
            dog_discriminator(replay_images, training=True)
            exp_replay = []    
        '''

        # display.clear_output(wait=True)

        # Cosine learning rate decay
        if (epoch + 1) % decay_step == 0:
            new_lr_d = tf.train.cosine_decay(new_lr_d, min(global_step, lr_decay_steps), lr_decay_steps)
            new_lr_g = tf.train.cosine_decay(new_lr_g, min(global_step, lr_decay_steps), lr_decay_steps)
            generator_optimizer = tf.train.AdamOptimizer(learning_rate=new_lr_d, beta_1=0.5)
            discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=new_lr_g, beta_1=0.5)

        print('Epoch: {} computed for {} sec'.format(epoch + 1, time.time() - start))
        print('Gen_loss mean: ', np.mean(G_loss), ' std: ', np.std(G_loss))
        print('Disc_loss mean: ', np.mean(D_loss), ' std: ', np.std(D_loss))

    # Generate after the final epoch
    # display.clear_output(wait=True)
    # final_seed = tf.random.normal([64, noise_dim])
    generate_and_save_images(generator, epochs, seed)
    checkpoint.save(file_prefix=checkpoint_prefix)
    print('Final epoch.')


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


# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


# Load DATASET
# f_myfile = open('../dataset/CICDS2017/train_10x10_MI.pickle', 'rb')
# x_train = pickle.load(f_myfile)
# f_myfile.close()
# f_myfile = open('../dataset/CICDS2017/ytrain.pickle', 'rb')
# y_train = pickle.load(f_myfile)
# f_myfile.close()
# attacks = np.where(y_train == 0)  # ONLY ATTACKS
# x_train = np.reshape(x_train, [-1, 10, 10, 1]).astype('float32')
# train_images = x_train[attacks[0]]
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
generator = make_generator_model()
generator.summary()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
discriminator = make_discriminator_model()
discriminator.summary()
decision = discriminator(generated_image)
print(decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_initial_g, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_initial_d, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

"""## Define the training loop"""

EPOCHS = 3
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

noise = tf.random.normal([BATCH_SIZE, noise_dim])

generated_images = generator(noise, training=False)

real_output = discriminator(train_images[:, 0:256], training=False)
fake_output = discriminator(generated_images, training=False)

gen_loss = generator_loss(fake_output)
disc_loss = discriminator_loss(real_output, fake_output)
# print("gen loss: "+str(np.array(gen_loss))+" | disc loss: "+ str(np.array(disc_loss)))


display_image(EPOCHS)
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

import csv
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.python.keras.initializers import TruncatedNormal
from tqdm import tqdm, tqdm_notebook

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

weight_init_std = 0.02
weight_init_mean = 0.0

leaky_relu_slope = 0.2  #
# dropout_rate = 0.5  #
# lr_initial_d = 0.0002  #
# lr_initial_g = 0.0002  #
lr_decay_steps = 1000  #
noise_dim = 100  # or 120
# BATCH_SIZE =128 #
num_examples_to_generate = 16
decay_step = 50
BUFFER_SIZE = 60000
EPOCHS = 50
weight_initializer = TruncatedNormal(stddev=weight_init_std, mean=weight_init_mean,
                                     seed=42)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
seed = tf.random.normal([num_examples_to_generate, noise_dim])

optimizable_variable = {"batch": hp.choice("batch", [512]),
                        'dropout_rate': hp.uniform("dropout_rate", 0, 1),
                        'lr_initial_g': hp.uniform("lr_initial_g", 1e-4, 1e-1),
                        "lr_initial_d": hp.uniform("lr_initial_d", 1e-4, 1e-1)

                        }


def make_generator_model(dropout_rate):
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
                            input_shape=[28, 28, 1], kernel_initializer=weight_initializer))
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


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# @tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer, BATCH_SIZE):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    # print("gen loss: " + str(np.array(gen_loss)) + " | disc loss: " + str(np.array(disc_loss)))
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return np.array(gen_loss), np.array(disc_loss)


res = []


def train(dataset, epochs, param, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint):
    all_gl = np.array([]);
    all_dl = np.array([])
    print(param)
    for epoch in range(epochs):
        print("epoch"+str(epoch))
        G_loss = []
        D_loss = []

        start = time.time()
        new_lr_d = param["lr_initial_d"]
        new_lr_g = param["lr_initial_g"]
        global_step = 0

        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch, generator, discriminator, generator_optimizer,
                                        discriminator_optimizer, param["batch"])

            global_step = global_step + 1
            G_loss.append(g_loss);
            D_loss.append(d_loss)
            all_gl = np.append(all_gl, np.array([G_loss]))
            all_dl = np.append(all_dl, np.array([D_loss]))

        # Cosine learning rate decay
        if (epoch + 1) % decay_step == 0:
            new_lr_d = tf.cosine_decay(new_lr_d, min(global_step, lr_decay_steps), lr_decay_steps)
            new_lr_g = tf.train.cosine_decay(new_lr_g, min(global_step, lr_decay_steps), lr_decay_steps)
            generator_optimizer = tf.train.AdamOptimizer(learning_rate=new_lr_d, beta_1=0.5)
            discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=new_lr_g, beta_1=0.5)

        print('Epoch: {} computed for {} sec'.format(epoch + 1, time.time() - start))
        print('Gen_loss mean: ', np.mean(G_loss), ' std: ', np.std(G_loss))
        print('Disc_loss mean: ', np.mean(D_loss), ' std: ', np.std(D_loss))
    global res
    res.append(param)
    res[-1].update({"g_loss": G_loss[-1], "d_loss": D_loss[-1]})
    try:
        with open("res_gan_hyper.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=res[0].keys())
            writer.writeheader()
            writer.writerows(res)
    except IOError:
        print("I/O error")

    # Generate after the final epoch
    # display.clear_output(wait=True)
    # final_seed = tf.random.normal([64, noise_dim])
    # generate_and_save_images(generator, epochs, seed)
    # checkpoint.save(file_prefix=checkpoint_prefix)
    # print('Final epoch.')
    return G_loss[-1]


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


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()


# Batch and shuffle the data
# generator.summary()
# noise = tf.random.normal([1, 100])


# generated_image = generator(noise, training=False)
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# discriminator.summary()
# decision = discriminator(generated_image)
# print(decision)

# This method returns a helper function to compute cross entropy loss


def opt(param):
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(param["batch"])
    generator = make_generator_model(param["dropout_rate"])
    discriminator = make_discriminator_model()
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=param["lr_initial_g"], beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=param["lr_initial_d"], beta_1=0.5)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    g_loss = train(train_dataset, EPOCHS, param, generator, discriminator, generator_optimizer,
                   discriminator_optimizer, checkpoint)

    return {'loss': g_loss, 'status': STATUS_OK}


trials = Trials()
fmin(opt, optimizable_variable, trials=trials, algo=tpe.suggest, max_evals=20)
# train(train_dataset, EPOCHS)

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


#
# noise = tf.random.normal([BATCH_SIZE, noise_dim])
#
# generated_images = generator(noise, training=False)
#
# real_output = discriminator(train_images[:, 0:256], training=False)
# fake_output = discriminator(generated_images, training=False)
#
# gen_loss = generator_loss(fake_output)
# disc_loss = discriminator_loss(real_output, fake_output)
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

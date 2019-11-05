import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt

def get_data(n):
    (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images.reshape(-1, 28, 28, 1)
    train_images = train_images/127.5-1
    m = 4*n//5
    return train_images[:m], train_images[m:n]

def conv_block(x, filters, kernel_size, strides, padding, dropout=False, up=False, activation=tf.nn.leaky_relu):
    if up:
        x = layers.Conv2DTranspose(filters, kernel_size, strides, padding)(x)
    else:
        x = layers.Conv2D(filters, kernel_size, strides, padding)(x)
    if dropout != False:
        x = layers.Droput(dropout)(x)
    x = layers.Activation(activation)(x)
    return x

def get_generator(latent_dim):
    inp = tf.keras.Input((latent_dim,))
    x = layers.Dense(7*7*64)(inp)
    x = layers.Reshape((7, 7, 64))(x)
    x = conv_block(x, 64, 3, 1, 'same')
    x = conv_block(x, 32, 3, 2, 'same', up=True)
    x = conv_block(x, 32, 3, 1, 'same')
    x = conv_block(x, 16, 3, 2, 'same', up=True)
    x = conv_block(x, 16, 3, 1, 'same')
    out = conv_block(x, 1, 3, 1, 'same', activation='tanh')
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

def get_discriminator():
    inp = tf.keras.Input((28, 28, 1))
    x = conv_block(inp, 4, 3, 1, 'same')
    x = conv_block(x, 4, 3, 1, 'same')
    x = conv_block(x, 8, 3, 2, 'same')
    x = conv_block(x, 8, 3, 1, 'same')
    x = conv_block(x, 16, 3, 2, 'same')
    x = conv_block(x, 16, 3, 1, 'same')
    x = layers.Flatten()(x)
    x = layers.Dense(30)(x)
    out = layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model

n = 5000
latent_dim = 30
epochs = 100
batch_size = 32
gen_learning_rate = 1e-4
disc_learning_rate = 1e-4
train_images, val_images = get_data(n)
real_data = tf.data.Dataset.from_tensor_slices(train_images).shuffle(n).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices(val_images).batch(batch_size)
generator = get_generator(latent_dim)
discriminator = get_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(gen_learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(disc_learning_rate)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

gen_path = "d/d_gen_fash.h5"
disc_path = "d/d_disc_fash.h5"

def load_model(model, path):
    if os.path.isfile(path):
        model.load_weights(path)

# load_model(generator, gen_path)
# load_model(discriminator, disc_path)

def train():
    for epoch in range(epochs):
        total_gen_loss = 0
        total_disc_train_loss = 0
        total_disc_val_loss = 0
        total_disc_gen_acc = 0
        total_disc_real_acc = 0
        total_disc_val_acc = 0
        count = 0
        for real_images in real_data:
            z = np.random.randn(batch_size, latent_dim)
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(z)
                real_y = discriminator(real_images)
                gen_y = discriminator(generated_images)
                gen_loss = bce(np.ones_like(gen_y), gen_y)
                disc_loss = bce(np.ones_like(real_y), real_y) + bce(np.zeros_like(gen_y), gen_y)
            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
            total_gen_loss+= gen_loss
            total_disc_train_loss+= disc_loss
            total_disc_gen_acc+= np.count_nonzero(gen_y < 0)
            total_disc_real_acc+= np.count_nonzero(real_y > 0)
            count+= 1
        for real_images in val_data:
            real_y = discriminator(real_images)
            disc_val_loss = bce(np.ones_like(real_y), real_y)
            total_disc_val_loss+= disc_val_loss
            total_disc_val_acc+= np.count_nonzero(real_y > 0)
        
        total_gen_loss/=count*batch_size
        total_disc_train_loss/=(count*batch_size+train_images.shape[0])
        total_disc_val_loss/=val_images.shape[0]
        total_disc_gen_acc/=count*batch_size
        total_disc_real_acc/=train_images.shape[0]
        total_disc_val_acc/=val_images.shape[0]
        print("Epoch: {} \t generator loss: {}\n discriminator losses: train {}, val {}".format(epoch, total_gen_loss, total_disc_train_loss, total_disc_val_loss))
        print("Discriminator accuracy: fake: {}, real: {}, val: {}".format(total_disc_gen_acc, total_disc_real_acc, total_disc_val_acc))
        generator.save_weights(gen_path)
        discriminator.save_weights(disc_path)
        display_examples(2, 2, epoch)

z = np.random.randn(4, latent_dim)
def display_examples(n, m, epoch=epochs):
    gen_images = generator(z).numpy().reshape(-1, 28, 28)
    for i in range(n*m):
        plt.subplot(n, m, i+1)
        plt.imshow(gen_images[i]*0.5+0.5)
    plt.savefig("d/" + str(epoch)+".jpg")
    plt.clf()

# generator.summary()
# discriminator.summary()
# display_examples(2, 2)
train()
# display_examples(2, 2)
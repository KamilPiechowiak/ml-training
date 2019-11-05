import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data():
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images=train_images[:10000]
    mi = np.mean(train_images)
    sigma = np.std(train_images)
    train_images = (train_images.reshape(-1, 28, 28, 1)-mi)/sigma
    n = train_images.shape[0]
    m = n*4//5
    return train_images[:m], train_images[m:], mi, sigma

def conv_block(x, filters, kernel_size, strides, padding, dropout=False, up=False):
    if up:
        x = layers.Conv2DTranspose(filters, kernel_size, strides, padding)(x)
    else:
        x = layers.Conv2D(filters, kernel_size, strides, padding)(x)
    if dropout != False:
        x = layers.Dropout(dropout)(x)
    x = layers.ReLU()(x)
    return x

def get_encoder(latent_dim):
    inp = tf.keras.Input(shape=(28,28,1))
    x = conv_block(inp, 4, 3, 1, 'same')
    x = conv_block(x, 8, 3, 2, 'same')
    x = conv_block(x, 8, 3, 1, 'same')
    x = conv_block(x, 16, 3, 2, 'same')
    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim)(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model

def get_decoder(latent_dim):
    inp = tf.keras.Input((latent_dim,))
    x = layers.Dense(16*49)(inp)
    x = layers.Reshape((7, 7, 16))(x)
    x = conv_block(x, 8, 3, 2, 'same', up=True)
    x = conv_block(x, 8, 3, 1, 'same', up=True)
    x = conv_block(x, 4, 3, 2, 'same', up=True)
    x = conv_block(x, 1, 3, 1, 'same', up=True)
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model

epochs = 10
batch_size = 32
train_images, val_images, mi, sigma = load_data()
train_data = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices(val_images).batch(128)

latent_dim = 30
encoder = get_encoder(latent_dim)
decoder = get_decoder(latent_dim)
encoder_optimizer = tf.keras.optimizers.Adam(1e-3)
decoder_optimizer = tf.keras.optimizers.Adam(1e-3)
mse = tf.keras.losses.MeanSquaredError()

encoder_path = "c_fash_enc.h5"
decoder_path = "c_fash_dec.h5"
def load_model(model, path):
    if os.path.isfile(path):
        print("IN")
        model.load_weights(path)

load_model(encoder, encoder_path)
load_model(decoder, decoder_path)

def train():
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for images in train_data:
            with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape:
                features = encoder(images)
                results = decoder(features)
                loss = mse(images, results)
            encoder_gradient = encoder_tape.gradient(loss, encoder.trainable_variables)
            decoder_gradient = decoder_tape.gradient(loss, decoder.trainable_variables)
            encoder_optimizer.apply_gradients(zip(encoder_gradient, encoder.trainable_variables))
            decoder_optimizer.apply_gradients(zip(decoder_gradient, decoder.trainable_variables))
            total_loss+=loss
            count+=1
        total_loss/=count
        val_loss = 0
        count = 0
        for images in val_data:
            loss = mse(images, decoder(encoder(images)))
            val_loss+=loss
            count+=1
        val_loss/=count
        print("Epoch {}, training_loss: {}, validation_loss: {}".format(epoch, total_loss, val_loss))

def display_single(n, m, i, data):
    inp = data.reshape(28,28)*sigma+mi
    out = decoder(encoder(data.reshape(1,28,28,1))).numpy().reshape(28,28)*sigma+mi
    plt.subplot(n, m, 2*i+1)
    plt.imshow(inp)
    plt.subplot(n, m, 2*i+2)
    plt.imshow(out)

def display(offset=0):
    n, m = 6, 6
    k = n*m//4
    for i in range(k):
        display_single(n, m, i, train_images[i+offset])
    for i in range(k):
        display_single(n, m, i+k, val_images[i+offset])
    plt.show()

def test_latent_variables(data):
    features = encoder(data.reshape(1,28,28,1))
    for i in range(10):
        for j in range(10):
            plt.subplot(10, 10, 10*i+j+1)
            v = np.zeros(features.shape)
            v[0,4] = (i-5)*4
            v[0,11] = (j-5)*4
            res = features+v
            plt.imshow(decoder(res).numpy().reshape(28,28))
    plt.show()

test_latent_variables(val_images[3])
# display()
# train()
# display()
# encoder.save_weights(encoder_path)
# decoder.save_weights(decoder_path)
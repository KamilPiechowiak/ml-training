#classification with a compiled model
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def normalize(data, mi, sigma):
    return (data-mi)/sigma

def load_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    mi = np.mean(train_images)
    sigma = np.std(train_images)
    train_images = normalize(train_images, mi, sigma).reshape(-1, 28, 28, 1)
    test_images = normalize(test_images, mi, sigma).reshape(-1, 28, 28, 1)
    return (train_images[:10000], train_labels[:10000]), (test_images[:1000], test_labels[:1000])

def conv_block(x, filters, kernel_size, strides, padding):
    x = layers.Conv2D(filters, kernel_size, strides, padding)(x)
    x = layers.ReLU()(x)
    return x

def get_network():
    inp = tf.keras.Input(shape=(28, 28, 1))
    x = conv_block(inp, 4, 3, 1, 'same')
    x = conv_block(x, 8, 3, 2, 'same')
    x = conv_block(x, 8, 3, 1, 'same')
    x = conv_block(x, 16, 3, 2, 'same')
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    out = layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)

    return model

epochs = 10
batch_size = 32
(train_images, train_labels), (test_images, test_labels) = load_data()
model = get_network()
model.compile(  optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.2)


fig, axs = plt.subplots(2, 1)
def draw(i, name):
    print(np.arange(epochs))
    print(history.history[name])
    axs[i].plot(np.arange(epochs), history.history[name], label=name)
draw(0, 'loss')
draw(0, 'val_loss')
axs[0].legend()
draw(1, 'sparse_categorical_accuracy')
draw(1, 'val_sparse_categorical_accuracy')
axs[1].legend()
plt.show()


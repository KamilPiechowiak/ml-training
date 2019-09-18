#classification with an own training scheme
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def normalize(data, mi, sigma):
    return (data-mi)/sigma

def shuffle(x, y):
    n = x.shape[0]
    order = np.random.shuffle(np.arange(n))
    x[:] = x[order]
    y[:] = y[order]
    return x, y

def load_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images, train_labels = train_images[:10000], train_labels[:10000]
    mi = np.mean(train_images)
    sigma = np.std(train_images)
    train_images, train_labels = shuffle(normalize(train_images, mi, sigma).reshape(-1, 28, 28, 1), train_labels)
    test_images = normalize(test_images, mi, sigma).reshape(-1, 28, 28, 1)
    m = train_images.shape[0]*4//5
    return (train_images[:m], train_labels[:m]), (train_images[m:], train_labels[m:]), (test_images, test_labels)

def conv_block(x, filters, kernel_size, strides, padding, regularizer, dropout=False):
    x = layers.Conv2D(filters, kernel_size, strides, padding, kernel_regularizer=regularizer)(x)
    if dropout != False:
        x = layers.Dropout(dropout)(x)
    x = layers.ReLU()(x)
    return x

def get_network():
    regularizer = tf.keras.regularizers.l2(1e-4)
    inp = tf.keras.Input(shape=(28, 28, 1))
    x = conv_block(inp, 4, 3, 1, 'same', regularizer)
    x = conv_block(x, 8, 3, 2, 'same', regularizer)
    x = conv_block(x, 8, 3, 1, 'same', regularizer)
    x = conv_block(x, 16, 3, 2, 'same', regularizer)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu', kernel_regularizer=regularizer)(x)
    out = layers.Dense(10, activation='softmax', kernel_regularizer=regularizer)(x)
    model = tf.keras.Model(inputs=inp, outputs=out)

    return model

(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_data()
print(train_images.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(train_images.shape[0]).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(128)
model = get_network()
optimizer = tf.keras.optimizers.Adam()
scce = tf.keras.losses.SparseCategoricalCrossentropy()

train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

epochs = 10
train, val = [], []

for epoch in range(epochs):
    for images, y_true in train_dataset:
        with tf.GradientTape() as tape:
            y_pred = model(images)
            loss = scce(y_true, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_acc_metric(y_true, y_pred)
    acc = train_acc_metric.result()
    train_acc_metric.reset_states()
    for images, y_true in val_dataset:
        y_pred = model(images)
        val_acc_metric(y_true, y_pred)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print(acc, val_acc)
    train.append(acc)
    val.append(val_acc)
plt.plot(np.arange(epochs), train, label='train acc')
plt.plot(np.arange(epochs), val, label='val acc')
plt.legend()
plt.show()
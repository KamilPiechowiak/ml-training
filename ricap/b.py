import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
import os

beta = 1
x_size=32
y_size=32

names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_data(filename):
    with open(filename, 'rb') as f:
        res = pickle.load(f, encoding='bytes')
    data = np.array(res[b'data']).reshape(-1, 3, x_size, y_size)
    data = np.moveaxis(data, 1, -1)
    labels = res[b'labels']
    return (data, labels)

def to_vector(id=None):
    l = np.zeros(10)
    if id is not None:
        l[id] = 1
    return l

def cut_from_image(data, i, x, y):
    p, q = np.random.randint(0, x_size-x+1), np.random.randint(0, y_size-y+1)
    return (data[i][0][p:p+x, q:q+y], x*y/x_size/y_size)

def data_generator(n, batch_size):
    data = []
    for i in range(1, 5):
        data+=list(zip(*load_data('cifar/data_batch_' + str(i))))
        if len(data) >= n:
            break
    data = data[:n]
    
    while True:
        imgs = []
        labels = []
        ids = np.random.randint(0, len(data), (batch_size, 4))
        xs = (x_size*np.random.beta(beta, beta, (batch_size,))).astype(np.int)
        ys = (y_size*np.random.beta(beta, beta, (batch_size,))).astype(np.int)
        for i in range(batch_size):
            l = to_vector()
            arr = [[]]*4
            arr[0] = cut_from_image(data, ids[i][0], xs[i], ys[i])
            arr[1] = cut_from_image(data, ids[i][1], x_size-xs[i], ys[i])
            arr[2] = cut_from_image(data, ids[i][2], xs[i], y_size-ys[i])
            arr[3] = cut_from_image(data, ids[i][3], x_size-xs[i], y_size-ys[i])
            img = np.concatenate((np.concatenate((arr[0][0], arr[1][0])), np.concatenate((arr[2][0], arr[3][0]))), axis=1)
            for j in range(4):
                l+=to_vector(data[ids[i][j]][1])*arr[j][1]
            imgs.append(img)
            labels.append(l)
        yield (np.array(imgs), np.array(labels))

def visualise_data(img, label):
    plt.imshow(img)
    for i in range(10):
        if lab[i] > 0:
            print(lab[i], names[i])
    plt.show()

def get_val_data(n):
    data = load_data('cifar/data_batch_5')
    for i in range(n):
       data[1][i] = to_vector(data[1][i])
    return (np.array(data[0][:n]), np.array(data[1][:n]))

def conv_block(x, filters, kernel_size, strides, padding='same', regularizer=None, dropout=False):
    x = layers.Conv2D(filters, kernel_size, strides, padding, kernel_regularizer=regularizer)(x)
    if dropout != False:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = layers.ReLU()(x)
    return x

def get_model():
    regularizer = tf.keras.regularizers.l2(1e-4)
    inp = tf.keras.Input(shape=(32, 32, 3))
    x = conv_block(inp, 4, 3, 1, regularizer=regularizer)
    x = conv_block(x, 8, 3, 2, regularizer=regularizer)
    x = conv_block(x, 8, 3, 1, regularizer=regularizer)
    x = conv_block(x, 16, 3, 2, regularizer=regularizer)
    x = conv_block(x, 16, 3, 1, regularizer=regularizer)
    x = conv_block(x, 32, 3, 2, regularizer=regularizer)
    x = conv_block(x, 32, 3, 1, regularizer=regularizer)
    x = conv_block(x, 64, 3, 2, regularizer=regularizer)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu', kernel_regularizer=regularizer)(x)
    x = layers.Dense(10, activation='softmax', kernel_regularizer=regularizer)(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model

# g = data_generator(100, 2)
# a = next(g)
# img = a[0][0]
# lab = a[1][0]
# visualise_data(img, lab)

model_path = 'ricap.h5'
if False and os.path.isfile(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = get_model()

train_n = 40000
val_n = 10000
epochs = 10
batch_size = 128

model.compile(  optimizer = tf.keras.optimizers.Adam(1e-3),
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = model.fit_generator(data_generator(train_n, batch_size),
        validation_data=get_val_data(val_n), epochs=epochs,
        steps_per_epoch=train_n//batch_size)

model.save(model_path)


fig, axs = plt.subplots(2, 1)
def draw(i, name):
    axs[i].plot(np.arange(epochs), history.history[name], label=name)
draw(0, 'loss')
draw(0, 'val_loss')
axs[0].legend()
draw(1, 'categorical_accuracy')
draw(1, 'val_categorical_accuracy')
axs[1].legend()
plt.savefig('ricap.pdf')
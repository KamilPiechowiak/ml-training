import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
import os
from tensorflow.keras.datasets import cifar10
from scipy.ndimage import rotate

beta = 0.2
x_size = 32
y_size = 32

"""
Accuracy
0.64 - without augumentation
0.68 - with simple augumentation
0.69 - with ricap (small improvement) #TODO test with more epochs
"""
names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

NUM_CLASSES = 10

train_n = 50000
val_n = 10000
epochs = 10
batch_size = 256

################################################################################

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    Layer,
    BatchNormalization,
    Activation,
    Add,
    MaxPool2D,
    GlobalAvgPool2D,
    Flatten,
    Dense,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation
from tensorflow.compat.v1.keras.utils import get_custom_objects
import os


class Mish(Activation):
    """
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    """
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = "Mish"


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


get_custom_objects().update({"Mish": Mish(mish)})


def bottleneck_residual_block(X, kernel_size, filters, reduce=False, s=2):
    F1, F2 = filters

    X_shortcut = X

    if reduce:
        X_shortcut = Conv2D(filters=F2,
                            kernel_size=(1, 1),
                            strides=(s, s),
                            padding="same")(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)

        X = Conv2D(filters=F1,
                   kernel_size=(3, 3),
                   strides=(s, s),
                   padding="same")(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("Mish")(X)

    else:
        X = Conv2D(filters=F1,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding="same")(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation("Mish")(X)

    X = Conv2D(filters=F2,
               kernel_size=kernel_size,
               strides=(1, 1),
               padding="same")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("Mish")(X)

    X = Add()([X, X_shortcut])
    X = Activation("Mish")(X)

    return X


def ResNet9(x_size, y_size, features):
    X_input = Input((x_size, y_size, features))

    X = Conv2D(16, (5, 5), strides=(2, 2), name="conv1",
               padding="same")(X_input)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("Mish")(X)
    X = MaxPool2D((3, 3), strides=(2, 2), padding="same")(X)

    X = bottleneck_residual_block(X, 3, [16, 16])

    X = bottleneck_residual_block(X, 3, [32, 32], reduce=True, s=2)

    X = bottleneck_residual_block(X, 3, [64, 64], reduce=True, s=2)

    # X = bottleneck_residual_block(X, 3, [256, 256], reduce=True, s=2)

    X = GlobalAvgPool2D()(X)

    X = Flatten()(X)
    X = Dense(10, activation="softmax", name="fc")(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet9")

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=0.001),
        metrics=["categorical_accuracy"],
    )

    return model


################################################################################

def data_generator_without_augumentation(n, batch_size):
    global NUM_CLASSES
    (x_train, y_train), (_, _) = cifar10.load_data()
    x_train = x_train[:n]
    y_train = y_train[:n]
    x_train = x_train/127.5-1
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)

    while True:
        size = x_train.shape[0]
        for i in range(0, size, batch_size):
            if i + batch_size > size:
                break
            yield x_train[i:i + batch_size], y_train[i:i + batch_size]

def data_generator_with_simple_augumentation(n, batch_size):

    def augument(img):
        if np.random.uniform() > 0.5:
            img = img[:,::-1]
        rot = np.random.uniform(-30, 30)
        img = rotate(img, rot, reshape=False, mode='nearest')
        return img

    global NUM_CLASSES
    (x_train, y_train), (_, _) = cifar10.load_data()
    x_train = x_train[:n]
    y_train = y_train[:n]
    x_train = x_train/127.5-1
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)

    size = x_train.shape[0]
    while True:
        ids = np.arange(0, size, 1)
        np.random.shuffle(ids)
        x_train = x_train[ids]
        y_train = y_train[ids]
        for i in range(0, size, batch_size):
            if i + batch_size > size:
                break
            x_aug = []
            for j in range(i,i+batch_size):
                x_aug.append(augument(x_train[j]))
            yield np.stack(x_aug), y_train[i:i + batch_size]

def data_generator_with_ricap(n, batch_size):

    def cut_from_image(data, i, x, y):
        p, q = (
            np.random.randint(0, x_size - x + 1),
            np.random.randint(0, y_size - y + 1),
        )
        return (data[i][p:p + x, q:q + y], x * y / x_size / y_size)

    global NUM_CLASSES
    (x_train, y_train), (_, _) = cifar10.load_data()
    x_train = x_train[:n]
    y_train = y_train[:n]
    x_train = x_train/127.5-1

    while True:
        imgs = []
        labels = []
        ids = np.random.randint(0, x_train.shape[0], (batch_size, 4))
        xs = (x_size * np.random.beta(beta, beta,
                                      (batch_size, ))).astype(np.int)
        ys = (y_size * np.random.beta(beta, beta,
                                      (batch_size, ))).astype(np.int)
        for i in range(batch_size):
            l = np.zeros(NUM_CLASSES)
            arr = [[]] * 4
            arr[0] = cut_from_image(x_train, ids[i][0], xs[i], ys[i])
            arr[1] = cut_from_image(x_train, ids[i][1], x_size - xs[i], ys[i])
            arr[2] = cut_from_image(x_train, ids[i][2], xs[i], y_size - ys[i])
            arr[3] = cut_from_image(x_train, ids[i][3], x_size - xs[i],
                                    y_size - ys[i])
            img = np.concatenate(
                (
                    np.concatenate((arr[0][0], arr[1][0])),
                    np.concatenate((arr[2][0], arr[3][0])),
                ),
                axis=1,
            )
            for j in range(4):
                l += tf.keras.utils.to_categorical(y_train[ids[i][j]], NUM_CLASSES).flatten() * arr[j][1]
            imgs.append(img)
            labels.append(l)
        yield (np.array(imgs), np.array(labels))


def get_val_data(n):
    global NUM_CLASSES
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test[:n]
    y_test = y_test[:n]
    x_test = x_test/127.5-1
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    return x_test, y_test


def visualise_data(img, label):
    plt.imshow(img)
    for i in range(NUM_CLASSES):
        if label[i] > 0:
            print(label[i], names[i])
    plt.show()

def test_generator():
    gen = data_generator_with_simple_augumentation(100, 16)
    arr = next(gen)
    _, ax = plt.subplots(4, 4)
    print(arr[0].shape)
    for i in range(16):
        s = (i//4, i%4)
        ax[s].imshow(arr[0][i])
        ax[s].set_title(arr[1][i])
    plt.show()

data_generator = data_generator_with_ricap
model_path = "ricap.h5"
if False and os.path.isfile(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = ResNet9(*(32, 32, 3))

history = model.fit_generator(
    data_generator(train_n, batch_size),
    validation_data=get_val_data(val_n),
    epochs=epochs,
    steps_per_epoch=train_n // batch_size,
)

model.save(model_path)

fig, axs = plt.subplots(2, 1)


def draw(i, name):
    axs[i].plot(np.arange(epochs), history.history[name], label=name)


draw(0, "loss")
draw(0, "val_loss")
axs[0].legend()
draw(1, "categorical_accuracy")
draw(1, "val_categorical_accuracy")
axs[1].legend()
plt.savefig("ricap.pdf")

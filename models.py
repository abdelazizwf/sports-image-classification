import tensorflow as tf
from layers import *


def AlexNet():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.Conv2D(48, 7, padding='valid', strides=4, activation='relu'),
        tf.keras.layers.Lambda(tf.nn.local_response_normalization),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='valid'),
        tf.keras.layers.Conv2D(116, 5, padding='same', activation='relu'),
        tf.keras.layers.Lambda(tf.nn.local_response_normalization),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='valid'),
        tf.keras.layers.Conv2D(184, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(184, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(116, 3, padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(124, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(66, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    return model


def ResNet():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.Conv2D(32, 7, strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
    ])

    prev_filters = 32
    for filters in [32] * 3 + [64] * 4 + [128] * 6 + [256] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters

    model.add(tf.keras.layers.GlobalAvgPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(6, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Nadam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    return model
    

def GoogLeNet():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
        tf.keras.layers.Lambda(tf.nn.local_response_normalization),
        tf.keras.layers.Conv2D(64, 1, strides=1, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(192, 3, strides=1, padding="same", activation="relu"),
        tf.keras.layers.Lambda(tf.nn.local_response_normalization),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
        InceptionModule([96, 16, 64, 128, 32, 32]),
        InceptionModule([128, 32, 128, 192, 96, 64]),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
        InceptionModule([96, 16, 192, 208, 48, 64]),
        InceptionModule([112, 24, 160, 224, 46, 64]),
        InceptionModule([128, 24, 128, 256, 64, 64]),
        InceptionModule([144, 32, 112, 288, 64, 64]),
        InceptionModule([160, 32, 256, 320, 128, 128]),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
        InceptionModule([160, 32, 256, 320, 128, 128]),
        InceptionModule([192, 48, 384, 384, 128, 128]),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(6, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Nadam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    return model


def Xception():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.Conv2D(16, 3, strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, 3, strides=1, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        XceptionEntryRU(64, start_with_relu=False),
        XceptionEntryRU(128),
        XceptionEntryRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionMiddleRU(256),
        XceptionEntryRU(256),
        tf.keras.layers.Conv2D(512, 3, strides=1, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(1024, 3, strides=1, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Nadam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    return model


def SEResNet():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.Conv2D(32, 7, strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
    ])

    prev_filters = 32
    for filters in [32] * 3 + [64] * 4 + [128] * 6 + [256] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(SEResidualUnit(filters, strides=strides))
        prev_filters = filters

    model.add(tf.keras.layers.GlobalAvgPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(6, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    return model

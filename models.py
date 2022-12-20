import tensorflow as tf


def get_alexnet():
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
        metrics=[tf.keras.metrics.CategoricalCrossentropy()],
    )

    return model


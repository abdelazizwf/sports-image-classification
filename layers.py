import tensorflow as tf


class ResidualUnit(tf.keras.layers.Layer):
    
    def __init__(self, filters, strides, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        
        self.main_layers = [
            tf.keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
        ]

        self.skip_layers = []

        if strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(),
            ]

    def call(self, inputs):
        Y = inputs
        for layer in self.main_layers:
            Y = layer(Y)

        skip_Y = inputs
        for layer in self.skip_layers:
            skip_Y = layer(skip_Y)

        return self.activation(Y + skip_Y)


class InceptionModule(tf.keras.layers.Layer):

    def __init__(self, filters_list, **kwargs):
        super().__init__(**kwargs)

        options = {
            "strides": 1,
            "padding": "same",
        }
        
        self.conv1ml = tf.keras.layers.Conv2D(filters_list[0], 1, **options, activation="relu")
        self.conv1mr = tf.keras.layers.Conv2D(filters_list[1], 1, **options, activation="relu")
        self.pool1r = tf.keras.layers.MaxPool2D(pool_size=3, **options)

        self.conv2l = tf.keras.layers.Conv2D(filters_list[2], 1, **options, activation="relu")
        self.conv2ml = tf.keras.layers.Conv2D(filters_list[3], 3, **options, activation="relu")
        self.conv2mr = tf.keras.layers.Conv2D(filters_list[4], 5, **options, activation="relu")
        self.conv2r = tf.keras.layers.Conv2D(filters_list[5], 1, **options, activation="relu")

    def call(self, inputs):
        return tf.concat([
            self.conv2l(inputs),
            self.conv2ml(self.conv1ml(inputs)),
            self.conv2mr(self.conv1mr(inputs)),
            self.conv2r(self.pool1r(inputs)),
        ], axis=3)


class NaiveInceptionModule(tf.keras.layers.Layer):

    def __init__(self, filters_list, **kwargs):
        super().__init__(**kwargs)

        options = {
            "strides": 1,
            "padding": "same",
            "activation": "relu",
            "use_bias": False
        }
        
        self.conv1 = tf.keras.layers.Conv2D(filters_list[0], 1, **options)
        self.conv3 = tf.keras.layers.Conv2D(filters_list[1], 3, **options)
        self.conv5 = tf.keras.layers.Conv2D(filters_list[2], 5, **options)
        self.pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")

    def call(self, inputs):
        return tf.concat([
            self.conv1(inputs),
            self.conv3(inputs),
            self.conv5(inputs),
            self.pool(inputs),
        ], axis=3)


class XceptionEntryRU(tf.keras.layers.Layer):

    def __init__(self, filters, start_with_relu=True, **kwargs):
        super().__init__(**kwargs)

        options = {
            "strides": 1,
            "padding": "same",
            "use_bias": False,
        }

        self.layers = []

        if start_with_relu == True:
            self.layers.append(tf.keras.layers.ReLU())

        self.layers += [
            tf.keras.layers.SeparableConv2D(filters, 3, **options),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SeparableConv2D(filters, 3, **options),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
        ]

        self.skip_layers = [
            tf.keras.layers.Conv2D(filters, 1, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
        ]

    def call(self, inputs):
        Y = inputs
        for layer in self.layers:
            Y = layer(Y)

        skip_Y = inputs
        for layer in self.skip_layers:
            skip_Y = layer(skip_Y)

        return Y + skip_Y


class XceptionMiddleRU(tf.keras.layers.Layer):

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)

        options = {
            "strides": 1,
            "padding": "same",
            "use_bias": False,
        }

        self.layers = [
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SeparableConv2D(filters, 3, **options),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SeparableConv2D(filters, 3, **options),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SeparableConv2D(filters, 3, **options),
        ]

    def call(self, inputs):
        Y = inputs
        for layer in self.layers:
            Y = layer(Y)

        return Y + inputs


class SEResidualUnit(tf.keras.layers.Layer):
    
    def __init__(self, filters, strides, ratio=16, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        
        self.layers = [
            tf.keras.layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
        ]

        self.skip_layers = []

        if strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(),
            ]

        self.se_layers = [
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Reshape((1, 1, filters)),
            tf.keras.layers.Dense(filters // ratio, activation="relu"),
            tf.keras.layers.Dense(filters, activation="sigmoid"),
        ]

    def call(self, inputs):
        Y = inputs
        for layer in self.layers:
            Y = layer(Y)

        y = Y
        for layer in self.se_layers:
            y = layer(y)

        Y = Y * y 

        skip_Y = inputs
        for layer in self.skip_layers:
            skip_Y = layer(skip_Y)

        return self.activation(Y + skip_Y)


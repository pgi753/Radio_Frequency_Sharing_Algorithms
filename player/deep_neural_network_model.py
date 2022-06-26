from typing import Dict, List, Optional
import tensorflow as tf
from tensorflow_addons.layers import NoisyDense
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np


class DnnModel(Model):
    def __init__(self, conv_layers_list: List[Dict], num_action: int, dueling: bool, noisy: bool, distribution: bool,
                 num_support: int = 1):
        super(DnnModel, self).__init__()
        self._num_action = num_action
        self._num_support = num_support
        self._dueling = dueling
        self._noisy = noisy
        self._distribution = distribution
        self._conv_layers = []
        for conv_layer in conv_layers_list:
            filters = conv_layer['filters']
            kernel_size = conv_layer['kernel_size']
            strides = conv_layer['strides']
            layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                           padding='same', activation=None,
                           kernel_initializer='glorot_normal', bias_initializer='zeros')
            self._conv_layers.append(layer)
            self._conv_layers.append(BatchNormalization())
            self._conv_layers.append(Activation('relu'))
            if conv_layer['max_pool_size'] is not None:
                max_pool_size = conv_layer['max_pool_size']
                layer = MaxPooling2D(pool_size=max_pool_size)
                self._conv_layers.append(layer)
        self._flatten = Flatten()
        if self._noisy is True:
            self.v = NoisyDense(self._num_support, activation=tf.keras.activations.relu, use_bias=False)
            self.a = NoisyDense(self._num_action * self._num_support, activation=tf.keras.activations.relu, use_bias=False)
        else:
            self.v = Dense(self._num_support, activation=tf.keras.activations.relu)
            self.a = Dense(self._num_action * self._num_support, activation=tf.keras.activations.relu)

        if self._distribution is True:
            if self._noisy is True:
                self._fully_conn_layer = NoisyDense(units=512, activation=tf.keras.activations.relu, use_bias=False)
            else:
                self._fully_conn_layer = Dense(units=512, activation=tf.keras.activations.relu,
                                               kernel_initializer='glorot_normal', bias_initializer='zeros')
            self._fully_conn_layer2 = Dense(units=self._num_action * self._num_support, activation=None,
                                            kernel_initializer='glorot_normal', bias_initializer='zeros')
        else:
            if self._noisy is True:
                self._fully_conn_layer = NoisyDense(units=self._num_action, activation=tf.keras.activations.relu,
                                                    use_bias=False)
            else:
                self._fully_conn_layer = Dense(units=self._num_action, activation=tf.keras.activations.relu,
                                               kernel_initializer='glorot_normal', bias_initializer='zeros')

        self._soft_max = Softmax(axis=2)

    @tf.function
    def call(self, inputs):
        x = inputs
        for layer in self._conv_layers:
            x = layer(x)
        x = self._flatten(x)
        x = self._fully_conn_layer(x)
        if self._dueling is True:
            v = self.v(x)
            a = self.a(x)
            v = tf.reshape(v, [-1, 1, self._num_support])
            a = tf.reshape(a, [-1, self._num_action, self._num_support])
            x = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))

        elif self._distribution is True:
            x = self._fully_conn_layer2(x)
            x = tf.reshape(x, [-1, self._num_action, self._num_support])

        if self._distribution is True:
            x = self._soft_max(x)
        outputs = x
        return outputs

    def advantage(self, state):
        x = state
        for layer in self._conv_layers:
            x = layer(x)
        x = self._flatten(x)
        a = self.a(x)
        if self._distribution is True:  # is this right?
            a = tf.reshape(a, [-1, self._num_action, self._num_support])
        return a


# Test Tensorflow model
if __name__ == '__main__':
    layers = [
        {'filters': 4,
         'kernel_size': (2, 2),
         'strides': (2, 2),
         'max_pool_size': None},
        {'filters': 8,
         'kernel_size': (4, 4),
         'strides': (1, 1),
         'max_pool_size': (2, 2)}
    ]
    model = DnnModel(layers, 10)
    learning_rate = 0.001
    model.compile(optimizer=Adam(lr=learning_rate), loss="mse")
    inp = np.random.random((3, 1000, 100, 2))
    print(model(inp))

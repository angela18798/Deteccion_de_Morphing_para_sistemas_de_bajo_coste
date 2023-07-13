import tensorflow as tf
from typing import Tuple
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model

class ClassifierModel(tf.keras.Model):
    def __init__(self, input_shape: Tuple=(512, )):
        super(ClassifierModel, self).__init__()

        self.dense_layer = Dense(1, activation="sigmoid")

        self.build((None, ) + input_shape)

    def call(self, inputs):
        x = self.dense_layer(inputs)

        return x
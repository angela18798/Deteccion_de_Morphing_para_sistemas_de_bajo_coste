import tensorflow as tf
from typing import Tuple
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model
from tensorflow.keras.layers import Input

class BaseModel(tf.keras.Model):
    def __init__(self, network: tf.keras.Model, input_shape: Tuple=(224, 224, 3)):
        super(BaseModel, self).__init__()
        
        self.network = network
        self._main_body= network(weights="imagenet", include_top=False, input_shape=input_shape)

        self.globalAvgPool_layer = GlobalAveragePooling2D()
        self.dense_layer = Dense(512, activation="relu")
        self.drop  = tf.keras.layers.Dropout(0.3)

        self._freeze_layers()

        self.build((None, ) + input_shape)

    def call(self, inputs):
        x = self._main_body(inputs)
        x = self.globalAvgPool_layer(x)
        # x = self.flatten(x)
        x = self.dense_layer(x)
        x = self.drop(x)

        return x
    
    def _freeze_layers(self, layer_name: str="conv5_block2_out"):
        for layer in self._main_body.layers:
            if layer.name == layer_name:
                break
            layer.trainable = False

    def unfreeze(self, layer_name: str="conv5_block2_out"):
        for layer in self._main_body.layers:
            if layer.name == layer_name:
                break
            layer.trainable = True
        self._flatten_layer.trainable = True
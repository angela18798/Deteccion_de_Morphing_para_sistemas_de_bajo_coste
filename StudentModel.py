import tensorflow as tf
from typing import Tuple

class StudentModel(tf.keras.Model):
    def __init__(self, base_model: tf.keras.Model, top_model: tf.keras.Model, network: tf.keras.Model, input_shape: Tuple=(224,224,3)):
        super(StudentModel, self).__init__()
        self.base_model = base_model(network=network, input_shape=input_shape)

        if top_model is not None:
            self.top_model = top_model()
        else: 
            self.top_model = None

        self.build((None, ) + input_shape)

    def call(self, inputs):
        base_model = self.base_model(inputs)
        
        if self.top_model is not None:
            output = self.top_model(base_model)
        else: 
            output = base_model

        return output
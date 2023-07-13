import tensorflow as tf
from typing import Tuple

class TeacherModel(tf.keras.Model):
    def __init__(self, base_model: tf.keras.Model, top_model: tf.keras.Model, network: tf.keras.Model, input_shape: Tuple=(224,224,3)):
        super(TeacherModel, self).__init__()
        self.base_model = base_model(network=network, input_shape=input_shape)
        self.top_model = top_model()

        self.build((None, ) + input_shape)

    def call(self, inputs):
        base_model = self.base_model(inputs)
        output = self.top_model(base_model)

        return output
    
    def get_classifier(self):
        return self.top_model
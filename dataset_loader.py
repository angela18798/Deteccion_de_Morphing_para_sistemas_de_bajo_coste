import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet import preprocess_input as prep_inp_resnet
from tensorflow.keras.applications.mobilenet import preprocess_input as prep_inp_mobilenet
from tensorflow.keras.applications.efficientnet import preprocess_input as prep_inp_efficientnet
from tensorflow.keras.applications.nasnet import preprocess_input as prep_inp_nasnet
from tensorflow.keras.applications.regnet import preprocess_input as prep_inp_regnet
import cv2
from typing import List

# Dataset Loader

def dataset_generator(paths: tf.string, labels: tf.int32):
    def gen():

        for path, label in zip(paths, labels):
            yield path, [label]
    
    return gen


def dataset_generator_student(paths: tf.string, embeddings: tf.float32):
    def gen():

        for path, emb in zip(paths, embeddings):
            yield path, emb
    
    return gen

@tf.function
def load_image(path: tf.string, label: tf.int32=None):
    def _load_image(path):
        img = Image.open(path).convert("RGB")
        return img.resize((224, 224))

    
    image = tf.numpy_function(_load_image, [path], tf.uint8)

    if label is not None:
        return image, label
    else:
        return image


@tf.function
def image_to_grayscale(image: tf.float32, label: tf.int32=None):
    if label is not None:
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image)), label
    else:
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))


@tf.function
def preprocess(image: tf.float32, label: tf.int32=None):
    image = tf.cast(image, "float32")
    image = prep_inp_regnet(image)

    if label is not None:
        return image, label
    else:
        return image


def get_dataset(image_paths: str = None, labels: int = None, batch_size: int = None, grayscale: bool = False, shuffle: bool = True):
    # labels = tf.keras.utils.to_categorical(labels, num_classes=2, dtype="int32")

    dataset = tf.data.Dataset.from_generator(
        dataset_generator(image_paths, labels),
        (tf.string, tf.int32), 
        output_shapes=((None), (1,))
        )
    
    dataset = dataset.map(load_image)

    if grayscale:
        dataset = dataset.map(image_to_grayscale)

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size = 1000)

    if batch_size is not None:
        dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset

def get_dataset_val(image_paths: str = None, labels: int = None, batch_size: int = None, grayscale: bool = False, shuffle: bool = True):
    # labels = tf.keras.utils.to_categorical(labels, num_classes=2, dtype="int32")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, tf.constant(labels)))
    
    dataset = dataset.map(load_image)

    if grayscale:
        dataset = dataset.map(image_to_grayscale)

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size = 1000)

    if batch_size is not None:
        dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset

def get_dataset_test(image_paths: str = None, labels: int = None, grayscale: bool = False):
    # labels = tf.keras.utils.to_categorical(labels, num_classes=2, dtype="int32")
    
    imgs = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))

        imgs.append(img)
    imgs = prep_inp_regnet(np.array(imgs))
    return imgs, labels


def get_dataset_student(image_paths: str = None, embeddings: List = None, batch_size: int = None, grayscale: bool = False, shuffle: bool = True):
    # labels = tf.keras.utils.to_categorical(labels, num_classes=2, dtype="int32")

    dataset = tf.data.Dataset.from_generator(
        dataset_generator_student(image_paths, embeddings),
        (tf.string, tf.float32), 
        output_shapes=((None), (512,))
        )
    
    dataset = dataset.map(load_image)

    if grayscale:
        dataset = dataset.map(image_to_grayscale)

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size = 1000)

    if batch_size is not None:
        dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset

def get_dataset_val_student(image_paths: str = None, embeddings: List = None, batch_size: int = None, grayscale: bool = False, shuffle: bool = True):
    # labels = tf.keras.utils.to_categorical(labels, num_classes=2, dtype="int32")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, embeddings))
    
    dataset = dataset.map(load_image)

    if grayscale:
        dataset = dataset.map(image_to_grayscale)

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size = 1000)

    if batch_size is not None:
        dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset
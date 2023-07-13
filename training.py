from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.layers import Input
import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model
from keras.optimizers import SGD, Adadelta , Adam, Nadam, Adagrad
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os
import dataset_loader
import pandas as pd
import numpy as np
from BaseModel import BaseModel
from ClassifierModel import ClassifierModel
from TeacherModel import TeacherModel

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as e:
        print("Error setting memory growth")
        print(e)

def main():

    # load data
    train_path = "C:/Users/Angela/Desktop/morph/training.csv"
    val_path = "C:/Users/Angela/Desktop/morph/validation.csv"
    eval_path = "C:/Users/Angela/Desktop/morph/test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    eval_df = pd.read_csv(eval_path)

    train_dataset = dataset_loader.get_dataset(
        image_paths=train_df["path"].to_list(),
        labels=train_df["label"].to_list(),
        batch_size=256, 
        shuffle=True
    )

    validation_dataset = dataset_loader.get_dataset_val(
        image_paths=val_df["path"].to_list(),
        labels=val_df["label"].to_list(),
        batch_size=256
    )

    eval_dataset, labels_test = dataset_loader.get_dataset_test(
        image_paths=eval_df["path"].to_list(),
        labels=eval_df["label"].to_list(),
    )

    # Model
    # base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    # x = Dense(512, activation="relu")(x)
    # output = Dense(1, activation="sigmoid")(x)

    # model = Model(inputs=base_model.input, outputs=output)

    # for layer in model.layers:
    #     if layer.name == "conv5_block2_out":
    #         break
    #     layer.trainable = False

    model = TeacherModel(base_model=BaseModel, top_model=ClassifierModel, network=ResNet152, input_shape=(224, 224, 3))

    model.summary()

    model.compile(loss = "binary_crossentropy", optimizer = Adagrad(learning_rate=0.0001), metrics = ["accuracy"])

    # # Data balance
    # weight_class_0 = 3
    # weight_class_1 = (1/np.sum(train_df["label"]==1)) * len(train_df["label"])/2.0
    # class_weights = {0: weight_class_0, 1: weight_class_1}

    # Training
    model.fit(
        train_dataset,
        # steps_per_epoch = len(train_df["path"])//256,
        epochs = 4,
        validation_data = validation_dataset,
        verbose = 1,
        # class_weight = class_weights,
        # validation_steps = len(val_df["path"])//256
        )
    
    # Poner checkpoints para early stopping y guardado del mejor modelo
    #TODO

    model.save("CNN_source_final.tf",save_format="tf")
    model.save_weights("CNN_source_weights_final.tf",save_format="tf")

    predictions = model.predict(eval_dataset, batch_size = 64).round()
    print(classification_report(labels_test, predictions))
    print(confusion_matrix(labels_test, predictions))

if __name__ == "__main__":
    main()
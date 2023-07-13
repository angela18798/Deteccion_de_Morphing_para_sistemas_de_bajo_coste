from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.regnet import RegNetX016, RegNetX002, RegNetX004
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.layers import Input
import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model
from keras.optimizers import SGD, Adadelta , Adam, Nadam, Adagrad
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os
import h5py
import dataset_loader
import pandas as pd
import numpy as np
from BaseModel import BaseModel
from ClassifierModel import ClassifierModel
from TeacherModel import TeacherModel
from StudentModel import StudentModel
import fire

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
    except RuntimeError as e:
        print("Error setting memory growth")
        print(e)

pruebas = [
    {
        "iter": 0,
        "epochs": 5
    },
    {
        "iter": 1,
        "epochs": 5
    },
    {
        "iter": 0,
        "epochs": 10
    },
    {
        "iter": 1,
        "epochs": 10
    }
]

def main():

    # load data
    train_path = "C:/Users/Angela/Desktop/morph/training.csv"
    val_path = "C:/Users/Angela/Desktop/morph/validation.csv"
    eval_path = "C:/Users/Angela/Desktop/morph/test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    eval_df = pd.read_csv(eval_path)

    train_embs = h5py.File("C:/Users/Angela/Desktop/morph/pred_teacher.h5", mode='r')["dataset"]
    val_embs = h5py.File("C:/Users/Angela/Desktop/morph/pred_teacher_val.h5", mode='r')["dataset"]

    train_dataset = dataset_loader.get_dataset_student(
        image_paths=train_df["path"].to_list(),
        embeddings=train_embs[:],
        batch_size=256, 
        shuffle=True
    )

    validation_dataset = dataset_loader.get_dataset_val_student(
        image_paths=val_df["path"].to_list(),
        embeddings=val_embs[:],
        batch_size=256
    )

    eval_dataset, labels_test = dataset_loader.get_dataset_test(
        image_paths=eval_df["path"].to_list(),
        labels=eval_df["label"].to_list(),
    )

    for prueba in pruebas:
        iter = prueba["iter"]
        epochs = prueba["epochs"]

        model = StudentModel(base_model=BaseModel, top_model=None, network=RegNetX004, input_shape=(224, 224, 3))

        model.summary()

        model.compile(loss = "mean_squared_error", optimizer = Adam(learning_rate=0.001), metrics = ["accuracy"])

        # # Data balance
        # weight_class_0 = 3
        # weight_class_1 = (1/np.sum(train_df["label"]==1)) * len(train_df["label"])/2.0
        # class_weights = {0: weight_class_0, 1: weight_class_1}

        # Training
        model.fit(
            train_dataset,
            # steps_per_epoch = len(train_df["path"])//256,
            epochs = epochs,
            validation_data = validation_dataset,
            verbose = 1,
            # class_weight = class_weights,
            # validation_steps = len(val_df["path"])//256
            )
        
        # Poner checkpoints para early stopping y guardado del mejor modelo
        #TODO

        # model.save("CNN_Student_final.tf",save_format="tf")
        # model.save_weights("CNN_Student_weights_final.tf",save_format="tf")
        model.base_model.save_weights(f"regnetX004/CNN_Student_weights_final_base_model_{iter}_{epochs}epochs.h5")

        student_model = StudentModel(base_model=BaseModel, top_model=ClassifierModel, network=RegNetX004, input_shape=(224, 224, 3))
        student_model.base_model.load_weights(f"regnetX004/CNN_Student_weights_final_base_model_{iter}_{epochs}epochs.h5")
        student_model.top_model.load_weights("top_model_weights.h5")

        predictions = student_model.predict(eval_dataset, batch_size = 64).round()
        print(classification_report(labels_test, predictions))
        print(confusion_matrix(labels_test, predictions))

        matriz_confusion = confusion_matrix(labels_test, predictions)
        reporte_clasificacion = classification_report(labels_test, predictions)

        with open(f"regnetX004/classification_report_{iter}_{epochs}epochs.txt", "w") as file:
            file.write(reporte_clasificacion)

        clases = ['Reales', 'Morphing']

        # Plotear la matriz de confusión
        sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
        plt.savefig(f'regnetX004/{iter}_{epochs}epochs.png')
        plt.clf()

if __name__ == "__main__":
    fire.Fire(main)
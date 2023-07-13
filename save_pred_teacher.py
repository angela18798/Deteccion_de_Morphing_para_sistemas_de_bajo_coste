import tensorflow as tf 
import pandas as pd
import dataset_loader
import json
import h5py

model = tf.keras.models.load_model('best_CNN_source_final.tf')
base_model = model.base_model

df = pd.read_csv("C:/Users/Angela/Desktop/morph/validation.csv")
data, labels = dataset_loader.get_dataset_test(
    image_paths=df["path"].to_list(),
    labels=df["label"].to_list(),
)

with h5py.File('pred_teacher_val.h5', 'w') as file:
    pred_file = file.create_dataset("dataset", (len(data), 512))

    for batch in range(0, len(data), 1024):
        end_batch = min(batch + 1024, len(data))

        batch_data = data[batch:end_batch]
        pred_batch = base_model.predict(batch_data, batch_size=64)
        pred_file[batch:end_batch] = pred_batch

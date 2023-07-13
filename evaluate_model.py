import tensorflow as tf 
import numpy as np
import pandas as pd
import dataset_loader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model = tf.keras.models.load_model('best_CNN_source_final.tf')

eval_df = pd.read_csv("C:/Users/Angela/Desktop/morph/test.csv")
eval_dataset, labels_test = dataset_loader.get_dataset_test(
        image_paths=eval_df["path"].to_list(),
        labels=eval_df["label"].to_list(),
    )

predictions = model.predict(eval_dataset, batch_size = 64).round()
print(classification_report(labels_test, predictions))
matriz_confusion = confusion_matrix(labels_test, predictions)

clases = ['Reales', 'Morphing']

# Plotear la matriz de confusión
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)

plt.show()

print("Finish")
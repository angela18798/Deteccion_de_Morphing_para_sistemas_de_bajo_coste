import glob
import os 
import csv
import numpy as np

base_path = "C:/Users/Angela/Desktop/facelab_london"
folders_morph = ["morph_amsl", "morph_facemorpher", "morph_opencv", "morph_stylegan", "morph_webmorph"]
folders_raw = ["raw"]

def split_images():
    training_paths = []
    training_labels = []
    validation_paths = []
    validation_labels = []
    test_paths = []
    test_labels = []

    for folder in folders_morph:
        attack_paths = glob.glob(os.path.join(base_path, folder) + "/*.jpg")
        np.random.shuffle(attack_paths)
        attack_labels = [1 for _ in range(0, len(attack_paths))]

        num_training = int(len(attack_paths) * 0.7)
        num_validation = (len(attack_paths) - num_training) // 2

        training_paths +=  attack_paths[:num_training]
        training_labels += attack_labels[:num_training]

        validation_paths += attack_paths[num_training:num_training+num_validation]
        validation_labels += attack_labels[num_training:num_training+num_validation]

        test_paths += attack_paths[num_training+num_validation:]
        test_labels += attack_labels[num_training+num_validation:]
    
    raw_paths = glob.glob(os.path.join(base_path, folders_raw[0]) + "/*/*.jpg")
    np.random.shuffle(raw_paths)
    raw_labels = [0 for _ in range(0, len(raw_paths))]

    num_training = int(len(raw_paths) * 0.7)
    num_validation = (len(raw_paths) - num_training) // 2

    training_paths +=  raw_paths[:num_training]
    training_labels += raw_labels[:num_training]

    validation_paths += raw_paths[num_training:num_training+num_validation]
    validation_labels += raw_labels[num_training:num_training+num_validation]

    test_paths += raw_paths[num_training+num_validation:]
    test_labels += raw_labels[num_training+num_validation:]

    training_paths = np.array(training_paths)
    training_labels = np.array(training_labels)
    validation_paths = np.array(validation_paths)
    validation_labels = np.array(validation_labels)
    test_paths = np.array(test_paths)
    test_labels = np.array(test_labels)

    p_training = np.random.permutation(len(training_paths))
    p_validation = np.random.permutation(len(validation_paths))
    p_test = np.random.permutation(len(test_paths))

    training_paths = training_paths[p_training]
    training_labels = training_labels[p_training]
    validation_paths = validation_paths[p_validation]
    validation_labels = validation_labels[p_validation]
    test_paths = test_paths[p_test]
    test_labels = test_labels[p_test]

    with open("C:/Users/Angela/Desktop/morph/training.csv", "w", newline='') as training_csv:
        writer = csv.writer(training_csv)
        writer.writerow(["path", "label"])

        for path, label in zip(training_paths, training_labels):
            writer.writerow([path.replace("\\", "/"), label])

    with open("C:/Users/Angela/Desktop/morph/validation.csv", "w", newline='') as validation_csv:
        writer = csv.writer(validation_csv)
        writer.writerow(["path", "label"])

        for path, label in zip(validation_paths, validation_labels):
            writer.writerow([path.replace("\\", "/"), label])
    
    with open("C:/Users/Angela/Desktop/morph/test.csv", "w", newline='') as test_csv:
        writer = csv.writer(test_csv)
        writer.writerow(["path", "label"])

        for path, label in zip(test_paths, test_labels):
            writer.writerow([path.replace("\\", "/"), label])

split_images()
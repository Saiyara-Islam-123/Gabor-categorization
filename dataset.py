import os
import cv2
import torch
from sklearn.model_selection import train_test_split

def get_dataset():
    path = "GABORS/gabors_1/experimentFiles/gabors/testing"
    folders = os.listdir(path)
    gabors = []
    import numpy as np
    labels = []

    for folder in folders:
        pair = os.listdir(path + "/" + folder)
        for im in pair:
            im_gray = cv2.imread(path + "/" + folder + "/" + im, cv2.IMREAD_GRAYSCALE)
            im_gray = im_gray/255
            gabors.append(im_gray)
            if "cat_1" in im:
                labels.append(1)

            else:
                labels.append(0)

    gabors = np.array(gabors)
    labels = np.array(labels)

    return (gabors), (labels)

def test_train_split():
    X, y = get_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return torch.tensor(X_train).float(), torch.tensor(y_train).long(), torch.tensor(X_test).float(), torch.tensor(y_test).long()
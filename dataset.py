import os
import cv2
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def get_dataset():

    gabors = []

    labels = []

    df = pd.read_excel("experimentFiles/categorisation.xlsx")

    for i in range(len(df["category"])):
        label = df["category"].iloc[i]

        if label == "l":
            labels.append(1)
        else:
            labels.append(0)

        local = df["Image_file"].iloc[i]
        path = "GABORS/gabors_1/"+local[1:]
        im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        im_gray = im_gray / 255
        gabors.append(im_gray)

    gabors = np.array(gabors)
    labels = np.array(labels)

    return (gabors), (labels)

def test_train_split():
    X, y = get_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return torch.tensor(X_train).float(), torch.tensor(y_train).long(), torch.tensor(X_test).float(), torch.tensor(y_test).long()



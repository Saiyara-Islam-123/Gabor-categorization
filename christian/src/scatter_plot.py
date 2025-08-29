import torch
from Net import *
from sklearn.manifold import MDS
from dataset import load_gabor_data
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import math
from math import sin, cos

import pandas as pd

label_colors_freq = {
    1: "limegreen",
    0: "green"

}
label_colors_size = {
    1 : "lightsteelblue",
    0: "slategrey"

}

def get_indices_corr_cats(labels):
    ones = labels == 1
    zeros = labels == 0

    indices_ones = ones.nonzero()[0]
    indices_zeros = zeros.nonzero()[0]
    return indices_ones, indices_zeros

def rotation(X_mds, labels):

    indices_ones, indices_zeros = get_indices_corr_cats(labels)

    X_mds_ones = X_mds[indices_ones] #cluster
    X_mds_zeros = X_mds[indices_zeros] #cluster
    mean_of_ones = np.mean(X_mds_ones, axis=0)
    mean_of_zeros = np.mean(X_mds_zeros, axis=0)
    print(mean_of_ones, mean_of_zeros)

    vector_between = abs(mean_of_ones - mean_of_zeros)
    gradient = (mean_of_ones[1]-mean_of_zeros[1])/(mean_of_ones[0]-mean_of_zeros[0])
    angle = math.atan(vector_between[1]/vector_between[0])

    if gradient <= 0: #anticlockwise
        mat = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])

    else: #clockwise
        mat = np.array([[cos(angle), sin(angle)], [-sin(angle), cos(angle)]])

    X_mds_ones_prime = (mat @ X_mds_ones.T).T
    X_mds_zeros_prime = (mat @ X_mds_zeros.T).T
    return X_mds_ones_prime, X_mds_zeros_prime


def of_two(matrix):
    mds = MDS(n_components=2, random_state=42)
    return mds.fit_transform(matrix)


def scatter_plot(train_type, weights, lr, batch, epoch, loc):

    if train_type == "sup":
        unsup_net = Net()
        unsup_weights_path = "../net_weights/unsup/unsup_weights_ lr= 0.001 0 100.pth"
        unsup_net.load_state_dict(torch.load(unsup_weights_path))

        m = SupervisedNet(unsup_net)
        m.load_state_dict(torch.load(weights))

    elif train_type == "unsup":
        m = Net()
        m.load_state_dict(torch.load(weights))

    m_outputs = []

    dataset, _, _ = load_gabor_data(excel_file="categorisation 4000.xlsx", batch_size=500)


    for images, labels in dataset:

        if train_type == "sup":
            _ = m(images)
            encoder_outputs = m.encoder_output
            m_outputs.append(encoder_outputs.detach().numpy())
            labels_arr=(labels.detach().numpy())

        elif train_type == "unsup":
            _ = m(images)
            encoder_outputs = m.encoded
            m_outputs.append(encoder_outputs.detach().numpy())
            labels_arr=(labels.detach().numpy())
        break



    X_mds = of_two(m_outputs[0])

    ones, zeros = rotation(X_mds, labels_arr)

    ones_x_axis = ones[:, 0]
    zeros_x_axis = zeros[:, 0]

    ones_y_axis = ones[:, 1]
    zeros_y_axis = zeros[:, 1]
    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.scatter(ones_x_axis, ones_y_axis, color="limegreen", s=10)
    plt.scatter(zeros_x_axis, zeros_y_axis, color="green", s=10)


    plt.scatter([], [], color= "limegreen", label ="Cat1" )
    plt.scatter([], [], color= "green", label='Cat0')


    plt.title( train_type +f"ervised training scatterplot, Epoch: {epoch} Batch: {batch}" )
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.legend()
    plt.savefig(f"{loc}/{train_type} lr = {lr}, {epoch} {batch}.png")
    plt.show()

def plot_raw_data():

    excel_file = "categorisation 4000.xlsx"
    dataset, _, _ = load_gabor_data(excel_file, batch_size=500)

    im = []

    for images, labels in dataset:
        im.append(images)
        labels_arr = (labels.detach().numpy())
        break


    X_mds = of_two(im[0].reshape(500, 128*128*3))

    indices_ones, indices_zeros = get_indices_corr_cats(labels_arr)

    ones = X_mds[indices_ones]
    zeros = X_mds[indices_zeros]

    ones_x_axis = ones[:, 0]
    zeros_x_axis = zeros[:, 0]

    ones_y_axis = ones[:, 1]
    zeros_y_axis = zeros[:, 1]
    plt.xlim(-60, 60)
    plt.ylim(-60, 60)

    plt.scatter(ones_x_axis, ones_y_axis, color="limegreen", s=10)
    plt.scatter(zeros_x_axis, zeros_y_axis, color="green", s=10)



    plt.scatter([], [], color= "limegreen", label ="Cat1" )
    plt.scatter([], [], color= "green", label='Cat0')


    plt.title( "No training scatterplot" )
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig("../whole_plots/scatter_plots/no_train/no_training.png")

    plt.show()

if __name__ == '__main__':
    #plot_raw_data()

    for i in range(100):
        scatter_plot(train_type="unsup", weights=f"../net_weights/unsup/unsup_weights_ lr= 0.001 0 {i+1}.pth", lr=0.001, batch=i, epoch=0, loc="../whole_plots/scatter_plots/unsup_rotated")

    for i in range(100):
        scatter_plot(train_type="sup", weights=f"../net_weights/sup/sup_net_weights_lr=0.001 0 {i}.pth", lr=0.001, batch=i, epoch=0, loc="../whole_plots/scatter_plots/sup_rotated")
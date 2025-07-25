import torch
from Net import *
from sklearn.manifold import TSNE
from dataset import load_gabor_data
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.spatial import procrustes
import pandas as pd

label_colors_freq = {
    1: "limegreen",
    0: "green"

}
label_colors_size = {
    1 : "lightsteelblue",
    0: "slategrey"

}



def of_two(matrix):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(matrix)


def scatter_plot(train_type, weights, lr, batch, epoch, loc, excel_file, base_dir="Default"):

    if train_type == "sup":
        unsup_net = Net()
        unsup_weights_path = "../net_weights/unsup/unsup_net_weights_ lr= 0.0001 0 49.pth"
        unsup_net.load_state_dict(torch.load(unsup_weights_path))

        m = SupervisedNet(unsup_net)
        m.load_state_dict(torch.load(weights))

    elif train_type == "unsup":
        m = Net()
        m.load_state_dict(torch.load(weights))

    m_outputs = []
    if base_dir != "Default":
        base_dir = "C:\\Users\\Admin\\Documents\\GitHub\\Gabor-categorization\\christian\\src\\Control\\gabors_2\\"

        dataset, _, _ = load_gabor_data(excel_file, batch_size=400, base_dir=base_dir)

    else:
        dataset, _, _ = load_gabor_data(excel_file, batch_size=500)

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




    X_tnse = of_two(m_outputs[0])

    #_, mx, _ = procrustes(standard_dataset, X_tnse)
    mx = X_tnse
    pc1 = mx[:, 0]
    pc2 = mx[:, 1]

    for i in range(mx.shape[0]):
        x_axis = pc1[i]
        y_axis = pc2[i]

        if base_dir == "Default":

            color = label_colors_freq[labels_arr[i]]

        else:
            color = label_colors_size[labels_arr[i]]

        plt.scatter(x_axis, y_axis, color=color, s=10)

    if base_dir == "Default":
        plt.scatter([], [], color= "limegreen", label ="Cat1" )
        plt.scatter([], [], color= "green", label='Cat0')
    else:
        plt.scatter([], [], color="lightsteelblue", label="Cat1")
        plt.scatter([], [], color="slategrey", label='Cat0')



    plt.title( train_type +f"ervised training scatterplot, Epoch: {epoch} Batch: {batch}" )
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.legend()
    plt.savefig(f"{loc}/{train_type} lr = {lr}, {epoch} {batch}.png")
    plt.show()

def plot_raw_data(default):
    if default:
        excel_file = "categorisation 4000.xlsx"
        dataset, _, _ = load_gabor_data(excel_file, batch_size=500)

    else:
        base_dir = "C:\\Users\\Admin\\Documents\\GitHub\\Gabor-categorization\\christian\\src\\Control\\gabors_2\\"
        excel_file = "Control/gabors_2/experimentFiles/categorisation.xlsx"
        dataset, _, _ = load_gabor_data(excel_file, batch_size=400, base_dir=base_dir)

    im = []

    for images, labels in dataset:
        im.append(images)
        labels_arr = (labels.detach().numpy())
        break

    print(im[0].shape)
    if default:
        X_tnse = of_two(im[0].reshape(500, 128*128*3))

    else:
        X_tnse = of_two(im[0].reshape(160, 128*128*3))

    pc1 = X_tnse[:, 0]
    pc2 = X_tnse[:, 1]


    for i in range(len(labels_arr)):
        x_axis = pc1[i]
        y_axis = pc2[i]

        if default:
            color = label_colors_freq[labels_arr[i]]
        else:
            color = label_colors_size[labels_arr[i]]

        plt.scatter(x_axis, y_axis, color=color, s=10)


    if default:
        plt.scatter([], [], color= "limegreen", label ="Cat1" )
        plt.scatter([], [], color= "green", label='Cat0')

    else:
        plt.scatter([], [], color="lightsteelblue", label="Cat1")
        plt.scatter([], [], color="slategrey", label='Cat0')

    plt.title( "No training scatterplot" )
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.legend()
    if default:
        plt.savefig("../whole_plots/scatter_plots_by_freq/no_train/no_training.png")

    else:
        plt.savefig("../whole_plots/scatter_plots_size/no_train/no_training.png")
    plt.show()

if __name__ == "__main__":
    plot_raw_data(default=False)


'''
if __name__ == '__main__':
    #get_standard_dataset()

    #plot_raw_data()
   
    for i in range(50):
        scatter_plot(train_type="unsup", weights=f"../net_weights/unsup/unsup_net_weights_ lr= 0.0001 0 {i}.pth", lr=0.0001, batch=i, epoch=0, loc="unsup, every batch")

     

    for i in range(5):
        for j in range(5):
            w = f"../net_weights/sup_control/sup_net_weights_ lr=0.0001 {i} {j}.pth"

            e = i
            c = j
            print(e, c)
            scatter_plot(train_type="sup", weights=w, lr=0.0001, batch=c, epoch=e, loc="sup_control_slow")
'''
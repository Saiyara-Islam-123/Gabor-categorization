import torch
from Net import *
from sklearn.manifold import TSNE
from dataset import load_gabor_data
import numpy as np
import matplotlib.pyplot as plt
import os
import re


label_colors = {
    1: "limegreen",
    0: "green"

}


def of_two(matrix):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(matrix)


def scatter_plot(train_type, weights, lr, batch, epoch):

    if train_type == "sup":
        unsup_net = Net()
        unsup_weights_path = "../net_weights/unsup_4000/unsup_net_weights_ lr= 0.0001 0 49.pth"
        unsup_net.load_state_dict(torch.load(unsup_weights_path))

        m = SupervisedNet(unsup_net)
        m.load_state_dict(torch.load(weights))

    elif train_type == "unsup":
        m = Net()
        m.load_state_dict(torch.load(weights))

    m_outputs = []

    excel_file = "categorisation 4000.xlsx"
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
    print(m_outputs[0].shape)
    pc1 = X_tnse[:, 0]
    pc2 = X_tnse[:, 1]

    for i in range(len(labels_arr)):
        x_axis = pc1[i]
        y_axis = pc2[i]

        color = label_colors[labels_arr[i]]

        plt.scatter(x_axis, y_axis, color=color, s=10)

    plt.scatter([], [], color= "limegreen", label ="Cat1" )
    plt.scatter([], [], color= "green", label='Cat2')


    plt.title( train_type +f"ervised training scatterplot, Epoch: {epoch} Batch: {batch}" )
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.legend()
    plt.savefig(f"whole_plots/scatter_plots/sup, every batch, slow lr, 2 epochs/{train_type} lr = {lr}, {batch} {epoch}.png")
    plt.show()

def plot_raw_data():
    excel_file = "categorisation 4000.xlsx"
    dataset, _, _ = load_gabor_data(excel_file, batch_size=500)

    im = []

    for images, labels in dataset:
        im.append(images)
        labels_arr = (labels.detach().numpy())
        break

    print(im[0].shape)
    X_tnse = of_two(im[0].reshape(500, 128*128*3))
    pc1 = X_tnse[:, 0]
    pc2 = X_tnse[:, 1]

    for i in range(len(labels_arr)):
        x_axis = pc1[i]
        y_axis = pc2[i]

        color = label_colors[labels_arr[i]]

        plt.scatter(x_axis, y_axis, color=color, s=10)

    plt.scatter([], [], color= "limegreen", label ="Cat1" )
    plt.scatter([], [], color= "green", label='Cat2')


    plt.title( "No training scatterplot" )
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.legend()
    plt.savefig("whole_plots/scatter_plots/no_train/no_training.png")
    plt.show()



if __name__ == '__main__':
    #plot_raw_data()
    #weights_unsup = os.listdir("../net_weights/unsup_4000")
    #c = 0
    #for w in weights_unsup:
        #scatter_plot(train_type="unsup", weights="../net_weights/unsup_4000/" + w, lr=0.0001, batch=c, epoch=0)
        #c += 1


    weights_sup = os.listdir("../net_weights/sup_4000/slow lr, 2 epochs")
    c = 0
    for w in weights_sup:
        scatter_plot(train_type="sup", weights="../net_weights/sup_4000/slow lr, 2 epochs/"+w, lr=0.0001, batch=c, epoch=0)
        c += 1
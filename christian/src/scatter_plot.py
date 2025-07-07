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

label_colors = {
    1: "limegreen",
    0: "green"

}
'''
def change_mat_shape(standard_mat, num_rows_target):
    print(standard_mat.shape)
    if num_rows_target > standard_mat.shape[0]:

        dif = num_rows_target - standard_mat.shape[0]
        new_row = np.array([[0,0]])
        for i in range(dif):
            standard_mat = np.append(standard_mat, new_row, axis=0)
    elif num_rows_target < standard_mat.shape[0]:

        dif = standard_mat.shape[0] - num_rows_target
        standard_mat = standard_mat[:standard_mat.shape[0]-dif, :]
    print(standard_mat.shape, num_rows_target)
    return standard_mat
'''

def of_two(matrix):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(matrix)


def get_standard_dataset():
    weights = "../net_weights/sup_4000/slow lr, 2 epochs/sup_net_weights_ lr=0.0001 1 49.pth"
    unsup_net = Net()
    unsup_weights_path = "../net_weights/unsup_4000/unsup_net_weights_ lr= 0.0001 0 49.pth"
    unsup_net.load_state_dict(torch.load(unsup_weights_path))

    m = SupervisedNet(unsup_net)
    m.load_state_dict(torch.load(weights))

    m_outputs = []

    excel_file = "categorisation 4000.xlsx"
    dataset, _, _ = load_gabor_data(excel_file, batch_size=500)

    for images, labels in dataset:
        _ = m(images)
        encoder_outputs = m.encoder_output
        m_outputs.append(encoder_outputs.detach().numpy())
        ls=(labels.detach().numpy().tolist())
        break

    X_tnse = of_two(m_outputs[0])
    df = pd.DataFrame(X_tnse)

    df.to_csv("Standard Dataset", index=False)



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

    #standard_dataset = pd.read_csv("Standard Dataset").to_numpy()


    X_tnse = of_two(m_outputs[0])

    #_, mx, _ = procrustes(standard_dataset, X_tnse)
    mx = X_tnse
    pc1 = mx[:, 0]
    pc2 = mx[:, 1]

    for i in range(mx.shape[0]):
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
    plt.savefig(f"whole_plots/scatter_plots/sup_old_gabor_dataset_code/{train_type} lr = {lr}, {epoch} {batch}.png")
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
    #get_standard_dataset()

    #plot_raw_data()
    #weights_unsup = os.listdir("../net_weights/unsup_4000")
    #c = 0
    #for w in weights_unsup:
        #scatter_plot(train_type="unsup", weights="../net_weights/unsup_4000/" + w, lr=0.0001, batch=c, epoch=0)
        #c += 1


    weights_sup = os.listdir("../net_weights/sup_control")

    for i in range(len(weights_sup)):
        w = weights_sup[i]
        w_splitted = w.split("lr=0.007")
        w_splitted = w_splitted[1]
        w_splitted = (w_splitted.split(" "))
        e = w_splitted[1]
        c = w_splitted[2].strip(".pth")
        print(e, c)
        scatter_plot(train_type="sup", weights="../net_weights/sup_control/"+w, lr=0.007, batch=c, epoch=e)


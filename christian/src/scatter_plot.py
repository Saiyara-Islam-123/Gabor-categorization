import torch
from Net import *
from sklearn.manifold import TSNE
from dataset import load_whole_dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from plotting import *

label_colors = {
    1: "red",
    0: "purple"

}

def of_two(matrix):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(matrix)


def scatter_plot(train_type, weights, time_step):

    if train_type == "sup":
        unsup_net = Net()
        unsup_weights_path = "../net_weights/unsup/unsup_net_weights_14 4.pth"
        unsup_net.load_state_dict(torch.load(unsup_weights_path))

        m = SupervisedNet(unsup_net)
        m.load_state_dict(torch.load(weights))

    elif train_type == "unsup":
        m = Net()
        m.load_state_dict(torch.load(weights))

    m_outputs = []
    excel_file = "categorisation.xlsx"
    dataset = load_whole_dataset(excel_file)

    for images, labels in dataset:

        if train_type == "sup":
            _ = m(images)
            encoder_outputs = m.encoder_output
            m_outputs.append(encoder_outputs.detach().numpy())
            labels_arr = (labels.detach().numpy())

        elif train_type == "unsup":
            _ = m(images)
            encoder_outputs = m.encoded
            m_outputs.append(encoder_outputs.detach().numpy())
            labels_arr = (labels.detach().numpy())

    X_tnse = of_two(m_outputs[0])
    pc1 = X_tnse[:, 0]
    pc2 = X_tnse[:, 1]

    for i in range(labels_arr.shape[0]):
        x_axis = pc1[i]
        y_axis = pc2[i]

        color = label_colors[labels_arr[i]]

        plt.scatter(x_axis, y_axis, color=color, s=6)


    splitted = weights.split(" ")
    plt.title( train_type +" gabor scatter plot, Epoch: " + re.sub(r'[a-z _ / .]', "", splitted[0]) + ", Batch: " + splitted[1].strip(".pth"))
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.savefig(w.strip(".pth"))
    plt.show()



if __name__ == '__main__':
    weights_unsup = [
                    "sup_net_weights_0 0.pth",
                    "sup_net_weights_0 1.pth",
                    "sup_net_weights_0 2.pth",
                    "sup_net_weights_0 3.pth",
                    "sup_net_weights_0 4.pth",

                    "sup_net_weights_1 0.pth",
                    "sup_net_weights_1 1.pth",
                    "sup_net_weights_1 2.pth",
                    "sup_net_weights_1 3.pth",
                    "sup_net_weights_1 4.pth",

                    "sup_net_weights_2 0.pth",
                    "sup_net_weights_2 1.pth",
                    "sup_net_weights_2 2.pth",
                    "sup_net_weights_2 3.pth",
                    "sup_net_weights_2 4.pth",

                     "sup_net_weights_3 0.pth",
                     "sup_net_weights_3 1.pth",
                     "sup_net_weights_3 2.pth",
                     "sup_net_weights_3 3.pth",
                     "sup_net_weights_3 4.pth",

                        "sup_net_weights_4 0.pth",
                        "sup_net_weights_4 1.pth",
                        "sup_net_weights_4 2.pth",
                        "sup_net_weights_4 3.pth",
                        "sup_net_weights_4 4.pth",




                     ]

    for w in weights_unsup:
        scatter_plot("sup", "../net_weights/sup/" + w)
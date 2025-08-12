
import shutil

import matplotlib.pyplot as plt

from plotter import *
from sup_trainer import *
from train_unsupervised import *
from dist import *
import os
import pandas as pd
from Net import *
import torch

def run_slow_sup_lr(run_number):
    os.mkdir(f"Prev runs/run_{run_number}")
    print("No train dists")
    no_train_dist(f"run_{run_number}")
    os.mkdir(f"../net_weights/Prev runs/run_{run_number}")
    print("Unsup")
    os.mkdir(f"../net_weights/Prev runs/run_{run_number}/unsup_4000")
    unsup_trainer(weights_dir=f"run_{run_number}/unsup_4000", csv_dir = f"run_{run_number}")
    print("Sup")
    os.mkdir(f"../net_weights/Prev runs/run_{run_number}/slow_lr_freq")
    sup_trainer = Slow_lr_Freq_Trainer(unsup_weight_path=f"run_{run_number}/unsup_4000", sup_weights_dir=f"run_{run_number}/slow_lr_freq", csv_dir=f"run_{run_number}")
    sup_trainer.train()
    #plotter = Slow_lr_Freq_Plotter()
    #plot(plotter)

def take_avg_dist(csv_file):
    dfs_as_np = []
    for i in range(1, 10):
        df = pd.read_csv(f"Prev runs/run_{i}/{csv_file}")
        dfs_as_np.append(df.to_numpy())

    dfs_as_np = np.array(dfs_as_np)
    np_arr_mean = np.mean(dfs_as_np, axis=0)
    df_mean = pd.DataFrame()
    df_mean["within 0"] = np_arr_mean[:, 0]
    df_mean["within 1"] = np_arr_mean[:, 1]
    df_mean["between"] = np_arr_mean[:, 2]

    if "sup" in csv_file and "unsup" not in csv_file:
        df_mean["acc"] = np_arr_mean[:, 3]

    df_mean.to_csv(f"mean_dist/{csv_file}", index=False)

def mean_scatter_plot(weights, weights_type, epoch, batch, lr):

    if "unsup" in weights_type:
        train_type = "unsup"
    else:
        train_type = "sup"

    dataset, _, _ = load_gabor_data(excel_file="categorisation 4000.xlsx", batch_size=500)
    tnses= []
    for i in range(1, 11):
        weights_path = f"../net_weights/Prev runs/run_{i}/{weights_type}/{weights}"
        if "unsup" in weights_type:
            m = Net()
        else:
            unsup_network = Net()
            unsup_network.load_state_dict(torch.load(f"../net_weights/Prev runs/run_{i}/unsup_4000/unsup_weights_ lr= 0.005 0 99.pth"))
            m = SupervisedNet(unsup_network)

        m.load_state_dict(torch.load(weights_path))
        for images, labels in dataset:
            if "unsup" in weights_type:
                _ = m(images)
                encoder_outputs = m.encoded
                m_outputs=(encoder_outputs.detach().numpy())
                labels_arr = (labels.detach().numpy())
            else:
                _ = m(images)
                encoder_outputs = m.encoder_output
                m_outputs=(encoder_outputs.detach().numpy())
                labels_arr = (labels.detach().numpy())
            break

        tnses.append(of_two(m_outputs))

    tnses = np.array(tnses)
    X_tnse = tnses.mean(axis=0)

    mx = X_tnse
    pc1 = mx[:, 0]
    pc2 = mx[:, 1]

    for i in range(mx.shape[0]):
        x_axis = pc1[i]
        y_axis = pc2[i]

        color = label_colors_freq[labels_arr[i]]

        plt.scatter(x_axis, y_axis, color=color, s=10)

    plt.scatter([], [], color="limegreen", label="Cat1")
    plt.scatter([], [], color="green", label='Cat0')

    plt.title(f"{train_type}ervised training scatterplot, Epoch: {epoch} Batch: {batch}")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig(f"../whole_plots/mean_scatter_plots/sup_slow/{train_type} lr = {lr}, {epoch} {batch}.png")
    plt.show()

def rename():
    for i in range(69, 100):
        prev_path = f"../net_weights/Prev runs/run_1/unsup_4000/unsup_net_weights_ lr= 0.005 0 {i}.pth"
        new_path = f"../net_weights/Prev runs/run_1/unsup_4000/unsup_weights_ lr= 0.005 0 {i}.pth"
        shutil.copyfile(prev_path, new_path)


if __name__ == '__main__':
    '''
    take_avg_dist(csv_file="Distance no train.csv")
    take_avg_dist(csv_file="LR=0.001, Distance every batch sup, epochs.csv")
    take_avg_dist(csv_file="LR=0.005, Distance every batch unsup.csv")


    for i in range(2,10):
        os.mkdir(f"../whole_plots/scatter_plots_multi_run/unsup/run_{i}")
        for j in range(0, 100):
            scatter_plot(train_type="unsup", weights=f"../net_weights/Prev runs/run_{i}/unsup_4000/unsup_weights_ lr= 0.005 0 {j}.pth", lr=0.005, batch=j, epoch=0, loc=f"../whole_plots/scatter_plots_multi_run/unsup/run_{i}", run=i)

    '''
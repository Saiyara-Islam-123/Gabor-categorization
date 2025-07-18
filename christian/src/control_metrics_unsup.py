from dist import *
import itertools
from Net import Net, SupervisedNet
import pandas as pd
from scatter_plot import *


def unsup_size_dist():
    excel_file = "Control/gabors_2/experimentFiles/categorisation.xlsx"
    base_dir = "C:\\Users\\Admin\\Documents\\GitHub\\Gabor-categorization\\christian\\src\\Control\\gabors_2\\"

    train_loader, _, _ = load_gabor_data(excel_file, batch_size=32, base_dir=base_dir)

    avg_distances = {}
    avg_distances[(0, 0)] = []
    avg_distances[(0, 1)] = []
    avg_distances[(1, 1)] = []

    for i in range(50):
        w = f"../net_weights/unsup/unsup_net_weights_ lr= 0.0001 0 {i}.pth"
        unsup_net = Net()
        unsup_net.load_state_dict(torch.load(w))

        images, labels = next(itertools.cycle(train_loader))
        _ = unsup_net(images)
        encoder_outputs = unsup_net.encoded
        zero, zero_one, one = sampled_all_distance(encoder_outputs, labels)
        avg_distances[(0, 0)].append(zero)
        avg_distances[(0, 1)].append(zero_one)
        avg_distances[(1, 1)].append(one)

    df = pd.DataFrame()
    df["within 0"] = avg_distances[(0, 0)]
    df["within 1"] = avg_distances[(1, 1)]
    df["between"] = avg_distances[(0, 1)]

    df.to_csv("LR=0.0001, Unsup size dist", index=False)

def unsup_size_scatter_plots():
    for i in range(50):
        w = f"../net_weights/unsup/unsup_net_weights_ lr= 0.0001 0 {i}.pth"
        scatter_plot(train_type="unsup", weights=w, lr=0.0001, batch=i, epoch=0, loc="../whole_plots/scatter_plots_size/unsup", excel_file="Control/gabors_2/experimentFiles/categorisation.xlsx", base_dir="Not Default")

if __name__ == "__main__":
    unsup_size_scatter_plots()
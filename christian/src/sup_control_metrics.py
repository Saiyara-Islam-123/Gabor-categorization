from dist import *
import itertools
from Net import Net, SupervisedNet
import pandas as pd
from scatter_plot import *

def sup_size_dist(sup_weights_dir, lr, trained_on):
    excel_file = "Control/gabors_2/experimentFiles/categorisation.xlsx"
    base_dir = "C:\\Users\\Admin\\Documents\\GitHub\\Gabor-categorization\\christian\\src\\Control\\gabors_2\\"

    train_loader, _, _ = load_gabor_data(excel_file, batch_size=32, base_dir=base_dir)

    avg_distances = {}
    avg_distances[(0, 0)] = []
    avg_distances[(0, 1)] = []
    avg_distances[(1, 1)] = []

    w = f"../net_weights/unsup/unsup_net_weights_ lr= 0.0001 0 49.pth"
    unsup_net = Net()
    unsup_net.load_state_dict(torch.load(w))

    for i in range(50):
        epoch = int(i / 5)
        batch = i % 5
        print(epoch, batch)
        sup_weights = f"{sup_weights_dir}/sup_net_weights_lr={lr} {epoch} {batch}.pth"
        sup_net = SupervisedNet(unsup_net)
        sup_net.load_state_dict(torch.load(sup_weights))
        images, labels = next(itertools.cycle(train_loader))
        _ = sup_net(images)
        encoder_outputs = sup_net.encoder_output
        zero, zero_one, one = sampled_all_distance(encoder_outputs, labels)
        avg_distances[(0, 0)].append(zero)
        avg_distances[(0, 1)].append(zero_one)
        avg_distances[(1, 1)].append(one)

    df = pd.DataFrame()
    df["within 0"] = avg_distances[(0, 0)]
    df["within 1"] = avg_distances[(1, 1)]
    df["between"] = avg_distances[(0, 1)]

    df.to_csv(f"LR={lr}, {trained_on}, Sup size dist", index=False)

if __name__ == "__main__":
    #sup_size_dist(sup_weights_dir= "../net_weights/fast_lr_size", lr=0.005, trained_on="size")


    df_with_accs = pd.read_csv("LR=0.005 Fast_lr_Size Distance every batch sup.csv")
    df_without_accs = pd.read_csv("LR=0.005, size, Sup size dist")
    df_without_accs["acc size"] = df_with_accs["acc size"]
    df_without_accs["acc freq"] = df_with_accs["acc freq"]
    df_without_accs.to_csv("LR=0.005, size, Sup size dist")
    

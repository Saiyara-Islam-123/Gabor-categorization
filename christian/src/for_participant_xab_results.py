import pandas as pd
from PIL import Image
from torchvision import transforms
from Net import *
import numpy as np
#compute distances only for pairs given in the excel file
import matplotlib.pyplot as plt

def clean_file():
    df = pd.read_csv("complete_xab_results.csv")

    new_type_column = []
    for i in range(len(df["A"])):
        cat_A = df["A"].iloc[i]
        cat_B = df["B"].iloc[i]

        if ("cat_0" in cat_A and "cat_1" in cat_B) or ("cat_1" in cat_A and "cat_0" in cat_B):
            new_type_column.append("between")
        elif "cat_0" in cat_A and "cat_0" in cat_B:
            new_type_column.append("within_0")
        elif "cat_1" in cat_A and "cat_1" in cat_B:
            new_type_column.append("within_1")

    df["type"] = new_type_column

    df.to_csv("complete_xab_results.csv", index=False)


def get_distances(weights_path, is_sup):
    if is_sup:
        unsup_model = Net()
        unsup_model.load_state_dict(torch.load("../net_weights/unsup_400/unsup_net_weights_ lr= 0.0001 4 4.pth"))

        sup_model = SupervisedNet(unsup_model)
        sup_model.load_state_dict(torch.load(weights_path))
    else:
        unsup_model = Net()
        unsup_model.load_state_dict(torch.load(weights_path))

    d = {"between" :[],
     "within_0" :[],
     "within_1" :[]
     }

    df = pd.read_csv("complete_xab_results.csv")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),

    ])

    for i in range(len(df["A"])):
        im_A_path = "../../GABORS_400/gabors_1/" + df["A"].iloc[i].strip("./")
        im_B_path = "../../GABORS_400/gabors_1/" + df["B"].iloc[i].strip("./")
        im_A = transform(Image.open(im_A_path).convert('RGB')).reshape(1, 3, 128, 128)
        im_B = transform(Image.open(im_B_path).convert('RGB')).reshape(1, 3, 128, 128)
        stacked = torch.concat([im_A, im_B], dim=0)

        if is_sup:
            _ = sup_model(stacked)
            stacked_mat = sup_model.encoder_output

        else:
            _ = unsup_model(stacked)
            stacked_mat = unsup_model.encoded

        type = df["type"].iloc[i]

        mat_A, mat_B = torch.unbind(stacked_mat, dim=0)


        mat_A_flattened = mat_A.view(mat_A.size(0), -1)
        mat_B_flattened = mat_B.view(mat_B.size(0), -1)

        mat_A_flattened_normalized = torch.nn.functional.normalize(mat_A_flattened, p=2, dim=1)
        mat_B_flattened_normalized = torch.nn.functional.normalize(mat_B_flattened, p=2, dim=1)

        dist = torch.norm(mat_A_flattened_normalized - mat_B_flattened_normalized).item()
        d[type].append(dist)
        print(type, dist)


    return np.mean(np.array(d["between"])), np.mean(np.array(d["within_0"])), np.mean(np.array(d["within_1"]))

def dists(sup_dir):
    d = {"between": [],
         "within_0": [],
         "within_1": []
         }
    for i in range(5):
        for j in range(5):
            weights_path = f"../net_weights/unsup_400/unsup_net_weights_ lr= 0.0001 {i} {j}.pth"
            b, w0, w1 = get_distances(weights_path=weights_path, is_sup=False)
            d["between"].append(b)
            d["within_0"].append(w0)
            d["within_1"].append(w1)

    for i in range(5):
        for j in range(5):
            weights_path = f"../net_weights/sup_400/sup_net_weights_ lr=0.0001 {i} {j}.pth"
            b, w0, w1 = get_distances(weights_path=weights_path, is_sup=True)

            d["between"].append(b)
            d["within_0"].append(w0)
            d["within_1"].append(w1)

    df = pd.DataFrame(d)

    df.to_csv(f"xab_{sup_dir}_dists.csv", index=False)

def plot():
    df = pd.read_csv("xab_sup_400_dists.csv")
    x = []
    for i in range(50):
        x.append(i)


    plt.plot(x, df['within_1'], color="limegreen", label="within Cat1")
    plt.plot(x, df['within_0'], color="green", label="within Cat0")
    plt.plot(x, df['between'], color="blue", label="between")

    plt.axvline(x=24, color='r', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel('Distance')
    plt.title("Gabor categorization distances across training batches")
    plt.show()

if __name__ == "__main__":
    get_distances(weights_path="../net_weights/sup_400/sup_net_weights_ lr=0.0001 4 4.pth", is_sup=True)
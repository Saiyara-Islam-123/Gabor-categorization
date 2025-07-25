import pandas as pd
from PIL import Image
from torchvision import transforms
from Net import *
import numpy as np
#compute distances only for pairs given in the excel file

def clean_file():
    df = pd.read_csv("complete_xab_results.csv")
    new_type_column = []
    for i in range(len(df["A"])):
        cat_A = df["A"].iloc[i].split("_0_")[1]
        cat_B = df["B"].iloc[i].split("_0_")[1]
        print(cat_A, cat_B)
        if cat_A != cat_B:
            new_type_column.append("between")
        else:
            if cat_A == "0":
                new_type_column.append("within_0")
            else:
                new_type_column.append("within_1")

    df.to_csv("complete_xab_results.csv", index=False)

def get_distances(weights_path, is_sup):
    if is_sup:
        unsup_model = Net()
        unsup_model.load_state_dict(torch.load("../net_weights/slow_lr_freq/sup_net_weights_lr=0.0001 0 99.pth"))

        sup_model = SupervisedNet(unsup_model)
        sup_model.load_state_dict(torch.load(weights_path))
    else:
        unsup_model = Net()
        unsup_model.load_state_dict(torch.load(weights_path))

    between_dists = []
    within_dists = []

    df = pd.read_csv("complete_xab_results.csv")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor()  # Converts image to tensor and scales to [0, 1]
    ])

    for i in range(df["A"]):
        im_A_path = "../" + df["A"].iloc[i].strip("./")
        im_B_path = "../" + df["B"].iloc[i].strip("./")
        im_A = transform(Image.open(im_A_path))
        im_B = transform(Image.open(im_B_path))

        _ = model(im_A)
        if is_sup:
            mat_A = model.encoder_output

        else:
            mat_A = model.encoded


        _ = model(im_B)
        if is_sup:
            mat_B = model.encoder_output

        else:
            mat_B = model.encoded


        type = df["xab_type_freq"].iloc[i]

        mat_A_flattened = mat_A.view(mat_A.size(0), -1)
        mat_B_flattened = mat_B.view(mat_B.size(0), -1)

        mat_A_flattened_normalized = torch.nn.functional.normalize(mat_A_flattened, p=2, dim=1)
        mat_B_flattened_normalized = torch.nn.functional.normalize(mat_B_flattened, p=2, dim=1)



        if type == "between":
            between_dists.append(torch.norm(mat_A_flattened_normalized - mat_A_flattened_normalized).item())
        else:
            within_dists.append(torch.norm(mat_B_flattened_normalized - mat_B_flattened_normalized).item())


    return np.mean(np.array(between_dists)), np.mean(np.array(within_dists))
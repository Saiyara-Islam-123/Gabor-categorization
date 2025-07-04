from dataset import load_gabor_data
import torch
from dist import *
import pandas as pd
import os
from Net import Net, SupervisedNet

def control_acc(dir):


    excel_file = "Control/gabors_2/experimentFiles/categorisation.xlsx"
    base_dir = "C:\\Users\\Admin\\Documents\\GitHub\\Gabor-categorization\\christian\\src\\Control\\gabors_2\\"
    trainloader, _, _ = load_gabor_data(excel_file, batch_size=180, base_dir=base_dir, split=0.9)
    unsup_net = Net()

    weight_path = "../net_weights/unsup_4000/unsup_net_weights_ lr= 0.0001 0 49.pth"
    unsup_net.load_state_dict(torch.load(weight_path))

    accuracy_values = []

    for i in range(2):
        for j in range(50):
            w = f"sup_net_weights_ lr=0.0001 {i} {j}.pth"
            lr = w.split(" ")[1].strip("lr=")

            sup_net = SupervisedNet(unsup_net)
            sup_net.load_state_dict(torch.load(dir +"/"+ w))

            total = 0
            correct = 0

            for images_true, labels_true in trainloader:
                outputs_2 = sup_net(images_true)

                _, predicted_true = torch.max(outputs_2.data, 1)
                total += labels_true.size(0)
                correct += (predicted_true == labels_true).sum().item()

                accuracy = 100 * correct / total
                accuracy_values.append(accuracy)
                print("Control acc: ", accuracy, w)
                break

    df = pd.DataFrame()
    df["Control data acc"] = accuracy_values
    df.to_csv(f"LR={lr}, Distance every batch sup, 2 epochs, size.csv", index=False)

if __name__ == "__main__":
    #control_acc(dir="../net_weights/sup_4000/slow lr, 2 epochs")
    df= pd.read_csv("LR=0.0001, Distance every batch sup, 2 epochs.csv")
    df= df.rename(columns={'acc': 'acc freq'})
    df_size = pd.read_csv("LR=0.0001, Distance every batch sup, 2 epochs, size.csv")
    df["acc size"] = df_size["Control data acc"]
    df.to_csv("LR=0.0001, Distance every batch sup, 2 epochs.csv with size", index=False)


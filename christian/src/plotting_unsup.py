import matplotlib.pyplot as plt
import pandas as pd
from Net import *
from christian.src.dataset import load_gabor_data


def acc_unsup():
    size_accs = []
    freq_accs = []

    for i in range(50):

        unsup_net = Net()
        unsup_net.load_state_dict(torch.load(f"../net_weights/unsup_4000/unsup_net_weights_ lr= 0.0001 0 {i}.pth"))
        sup_net = SupervisedNet(unsup_net)
        freq_train_loader, _, _ = load_gabor_data("categorisation 4000.xlsx", batch_size=50)

        for images1, labels1 in freq_train_loader:
            outputs1 = sup_net(images1)

            _, predicted1 = torch.max(outputs1.data, 1)
            total1 = labels1.size(0)
            correct1 = (predicted1 == labels1).sum().item()

            freq_accs.append(100 * correct1 / total1)

            break


        excel_file = "Control/gabors_2/experimentFiles/categorisation.xlsx"
        base_dir = "C:\\Users\\Admin\\Documents\\GitHub\\Gabor-categorization\\christian\\src\\Control\\gabors_2\\"
        size_train_loader, _, _ = load_gabor_data(excel_file, batch_size=50, base_dir=base_dir)

        for images2, labels2 in size_train_loader:
            outputs2 = sup_net(images2)

            _, predicted2 = torch.max(outputs2.data, 1)
            total2 = labels2.size(0)
            correct2 = (predicted2 == labels2).sum().item()

            size_accs.append(100 * correct2 / total2)

            break

        print(i, freq_accs[i], size_accs[i])
    df = pd.DataFrame()
    df["freq acc"] = freq_accs
    df["size acc"] = size_accs

    df.to_csv("Unsup accs")

def plot_unsup_accs():
    df = pd.read_csv("Unsup accs")
    plt.xlabel("Unsup batches")
    plt.ylabel("Accuracy")
    plt.title("Accuracy during unsup training")
    plt.plot( df["freq acc"], color="coral", label="frequency accuracy", marker="o")
    plt.plot( df["size acc"], color="palevioletred", label="size accuracy", linestyle="--")
    plt.legend()
    plt.savefig("unsup_accs.png")
    plt.show()



if __name__ == "__main__":
    plot_unsup_accs()


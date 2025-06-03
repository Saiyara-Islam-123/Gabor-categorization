import matplotlib.pyplot as plt
import pandas as pd



def plot():
    df_unsup = pd.read_csv("Distance every epoch unsup.csv")
    df_unsup = df_unsup.tail(15)


    df_sup = pd.read_csv("Distance every epoch sup.csv")
    df_sup = df_sup.head(15)


    accs = df_sup["acc"]
    df_sup.drop(columns=["acc"], inplace=True)

    df_whole = pd.concat([df_unsup, df_sup])
    x = []
    for i in range(30):
        x.append(i)
    x2 = []
    for i in range(15, 30):
        x2.append(i)

    fig, ax1 = plt.subplots()
    ax1.plot(x, df_whole['within 1'], color="limegreen", label="within 1")
    ax1.plot(x, df_whole['within 0'], color="green", label="within 0")
    ax1.plot(x, df_whole['between'], color="blue", label="between")
    ax1.axvline(x=14, color='r', linestyle='--')
    ax1.set_ylabel('Distance')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(x2, accs, color="orange", label="accuracy")
    ax2.set_ylabel('Accuracy')


    plt.xlabel("Epoch")
    plt.title("Gabor categorization acc and distances across epochs")

    plt.show()

plot()
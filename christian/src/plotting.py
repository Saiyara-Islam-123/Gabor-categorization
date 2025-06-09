import matplotlib.pyplot as plt
import pandas as pd



def plot(time_step):
    df_unsup = pd.read_csv("LR=0.0001, Distance every batch unsup.csv")
    df_unsup = df_unsup.tail(30)


    df_sup = pd.read_csv("LR=0.0001, Distance every batch sup.csv")
    df_sup = df_sup.head(30)


    accs = df_sup["acc"]
    df_sup.drop(columns=["acc"], inplace=True)

    df_whole = pd.concat([df_unsup, df_sup])
    x = []
    for i in range(60):
        x.append(i)
    x2 = []
    for i in range(30, 60):
        x2.append(i)

    fig, ax1 = plt.subplots()
    ax1.plot(x, df_whole['within 1'], color="limegreen", label="within 1")
    ax1.plot(x, df_whole['within 0'], color="green", label="within 0")
    ax1.plot(x, df_whole['between'], color="blue", label="between")
    ax1.axvline(x=29, color='r', linestyle='--')
    ax1.set_ylabel('Distance')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(x2, accs, color="orange", label="accuracy")
    ax1.axvline(x=time_step+30, color='black', linestyle='dashed')
    ax2.set_ylabel('Accuracy')


    plt.xlabel("Epoch")
    plt.title("Gabor categorization acc and distances across epochs")
    batch = time_step % 5
    epoch = time_step // 5
    plt.savefig("sup " + str(epoch) + " " + str(batch)  + " .png")
    plt.show()

for i in range(25):
    plot(i)
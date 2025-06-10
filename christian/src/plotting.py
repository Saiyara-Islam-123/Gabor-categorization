import matplotlib.pyplot as plt
import pandas as pd

def plot_epoch(time_step):
    df_unsup = pd.read_csv("LR=0.0001, Distance every batch unsup.csv")
    df_unsup = df_unsup.tail(40)
    rows_unsup = []


    df_sup = pd.read_csv("LR=0.0001, Distance every batch sup.csv")
    df_sup = df_sup.head(40)
    rows_sup = []

    for r in range(len(df_unsup["between"])):
        if r % 5 == 0:

            rows_unsup.append(df_unsup.iloc[r])
            rows_sup.append(df_sup.iloc[r])


    df_unsup_epoch = pd.DataFrame(rows_unsup)
    df_sup_epoch = pd.DataFrame(rows_sup)

    accs = df_sup_epoch["acc"]
    df_sup_epoch.drop(columns=["acc"], inplace=True)

    df_whole = pd.concat([df_unsup_epoch, df_sup_epoch])

    x = []
    for k in range(16):
        x.append(k)
    x2 = []
    for k in range(8, 16):
        x2.append(k)


    fig, ax1 = plt.subplots()
    ax1.plot(x, df_whole['within 1'], color="limegreen", label="within Cat1")
    ax1.plot(x, df_whole['within 0'], color="green", label="within Cat2")
    ax1.plot(x, df_whole['between'], color="blue", label="between")
    ax1.axvline(x=7, color='r', linestyle='--')
    ax1.set_ylabel('Distance')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(x2, accs, color="coral", label="accuracy")
    ax1.axvline(x=time_step + 8, color='black', linestyle='dashed')
    ax2.set_ylabel('Accuracy')

    plt.xlabel("Epoch")
    plt.title("Gabor categorization accuracy and distances across training Epochs")
    batch = 0
    epoch = time_step
    plt.savefig("sup " + str(epoch) + " " + str(batch) + " .png")
    plt.show()



def plot_batch(time_step):
    df_unsup = pd.read_csv("LR=0.0001, Distance every batch unsup.csv")
    df_unsup = df_unsup.tail(40)


    df_sup = pd.read_csv("LR=0.0001, Distance every batch sup.csv")
    df_sup = df_sup.head(40)


    accs = df_sup["acc"]
    df_sup.drop(columns=["acc"], inplace=True)

    df_whole = pd.concat([df_unsup, df_sup])
    x = []
    for k in range(80):
        x.append(k)
    x2 = []
    for k in range(40, 80):
        x2.append(k)

    fig, ax1 = plt.subplots()
    ax1.plot(x, df_whole['within 1'], color="limegreen", label="within Cat1")
    ax1.plot(x, df_whole['within 0'], color="green", label="within Cat2")
    ax1.plot(x, df_whole['between'], color="blue", label="between")
    ax1.axvline(x=39, color='r', linestyle='--')
    ax1.set_ylabel('Distance')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(x2, accs, color="coral", label="accuracy")
    ax1.axvline(x=time_step+40, color='black', linestyle='dashed')
    ax2.set_ylabel('Accuracy')


    plt.xlabel("Epoch")
    plt.title("Gabor categorization accuracy and distances across training batches")
    batch = time_step % 5
    epoch = time_step // 5
    plt.savefig("sup " + str(epoch) + " " + str(batch)  + " .png")
    plt.show()



if __name__ == '__main__':

    for i in range(8):
        plot_epoch(i)
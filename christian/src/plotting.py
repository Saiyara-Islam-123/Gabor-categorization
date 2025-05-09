import matplotlib.pyplot as plt
import pandas as pd

def plot():
    df_unsup = pd.read_csv("Distance per batch unsup.csv")
    df_unsup = df_unsup.tail(15)

    df_sup = pd.read_csv("Distance per batch sup.csv")
    df_sup = df_sup.head(15)


    df_whole = pd.concat([df_unsup, df_sup])
    x = []
    for i in range(30):
        x.append(i)


    plt.plot(x, df_whole['within 1'], color="limegreen", label="within 1")
    plt.plot(x, df_whole['within 0'], color="green", label="within 0")
    plt.plot(x, df_whole['between'], color="blue", label="between")
    plt.axvline(x=14, color='r', linestyle='--')
    plt.xlabel("Batch")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()

plot()
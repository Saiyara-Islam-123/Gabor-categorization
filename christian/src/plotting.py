import matplotlib.pyplot as plt
import pandas as pd

def plot_skip_batch(time_step, csv_unsup, csv_sup, lr, loc):
    df_unsup = pd.read_csv(csv_unsup)
    df_unsup = df_unsup.tail(50)
    rows_unsup = []


    df_sup = pd.read_csv(csv_sup)
    df_sup = df_sup.head(50)
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
    for k in range(20):
        x.append(k)
    x2 = []
    for k in range(10, 20):
        x2.append(k)


    fig, ax1 = plt.subplots()
    ax1.plot(x, df_whole['within 1'], color="limegreen", label="within Cat1")
    ax1.plot(x, df_whole['within 0'], color="green", label="within Cat2")
    ax1.plot(x, df_whole['between'], color="blue", label="between")
    ax1.axvline(x=9, color='r', linestyle='--')
    ax1.set_ylabel('Distance')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(x2, accs, color="coral", label="accuracy")
    ax1.axvline(x=time_step + 10, color='black', linestyle='dashed')
    ax2.set_ylabel('Accuracy')

    plt.xlabel("Epoch")
    plt.title("Gabor categorization accuracy and distances across training Epochs")
    batch = 0
    epoch = time_step
    plt.savefig(loc+f"/Lr={lr}" + str(epoch) + " " + str(batch)  + " .png")
    plt.show()

def plot_batch(time_step, csv_unsup, csv_sup, num_unsup_rows, num_sup_rows, lr, loc, is_sup):
    #df_no_train = pd.read_csv("Distance no train.csv")

    df_unsup = pd.read_csv(csv_unsup)
    df_unsup = df_unsup.tail(num_unsup_rows)


    df_sup = pd.read_csv(csv_sup)
    df_sup = df_sup.head(num_sup_rows)


    accs = df_sup["acc"]
    df_sup.drop(columns=["acc"], inplace=True)

    df_whole = pd.concat([df_unsup , df_sup])
    x = []
    for k in range(num_unsup_rows+num_sup_rows):
        x.append(k)
    x2 = []
    for k in range(num_unsup_rows, num_unsup_rows+num_sup_rows):
        x2.append(k)

    fig, ax1 = plt.subplots()
    ax1.plot(x, df_whole['within 1'], color="limegreen", label="within Cat1")
    ax1.plot(x, df_whole['within 0'], color="green", label="within Cat2")
    ax1.plot(x, df_whole['between'], color="blue", label="between")
    ax1.axvline(x=num_unsup_rows-1, color='r', linestyle='--')
    ax1.set_ylabel('Distance')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(x2, accs, color="coral", label="accuracy")
    ax1.axvline(x=time_step, color='black', linestyle='dashed')
    ax2.set_ylabel('Accuracy')


    plt.xlabel("Epoch")
    plt.title("Gabor categorization accuracy and distances across training batches")
    '''
    if is_sup == "sup":
        plt.savefig(loc+f"/Lr={lr} " + str(time_step-51)+ ".png")
    elif is_sup == "unsup":
        plt.savefig(loc + f"/Lr={lr} " + str(time_step - 1) + ".png")
    else:
        plt.savefig(loc + f"/Lr={lr} " + str(time_step) + ".png")
    '''
    plt.show()

def plot_batch_control(time_step, csv_unsup, csv_sup, num_unsup_rows, num_sup_rows, lr, loc, is_sup, is_control=False, by_size = False): #is control -> trained on size, by_size -> dist labelled by size
    if not by_size:

        df_no_train = pd.read_csv("Distance no train.csv")

    else:
        df_no_train = pd.read_csv("Size Distance no train.csv")

    df_unsup = pd.read_csv(csv_unsup)
    df_unsup = df_unsup.tail(num_unsup_rows)


    df_sup = pd.read_csv(csv_sup)
    df_sup = df_sup.head(num_sup_rows)

    if is_control:
        accs_side = df_sup["acc freq"]
        accs_main = df_sup["acc size"]
    else:
        accs_main = df_sup["acc freq"]
        accs_side = df_sup["acc size"]


    df_whole = pd.concat([df_no_train, df_unsup , df_sup])
    x = []
    for k in range(num_unsup_rows+num_sup_rows+1):
        x.append(k)
    x2 = []
    for k in range(num_unsup_rows+1, num_unsup_rows+num_sup_rows+1):
        x2.append(k)

    fig, ax1 = plt.subplots()
    if by_size == False:
        ax1.plot(x, df_whole['within 1'], color="limegreen", label="within Cat1")
        ax1.plot(x, df_whole['within 0'], color="green", label="within Cat0")
        ax1.plot(x, df_whole['between'], color="blue", label="between")
    else:
        ax1.plot(x, df_whole['within 1'], color="lightsteelblue", label="within Cat1")
        ax1.plot(x, df_whole['within 0'], color="slategrey", label="within Cat0")
        ax1.plot(x, df_whole['between'], color="navy", label="between")


    ax1.axvline(x=num_unsup_rows, color='r', linestyle='--')
    ax1.axvline(x=1, color='r', linestyle='--')
    ax1.set_ylabel('Distance')
    ax1.legend()

    ax2 = ax1.twinx()

    if is_control:

        ax2.plot(x2, accs_side, color="coral", label="frequency accuracy")
        ax2.plot(x2, accs_main, color="coral", label="size accuracy",linestyle='dashed')

    else:
        ax2.plot(x2, accs_main, color="coral", label="frequency accuracy")
        ax2.plot(x2, accs_side, color="coral", label="size accuracy", linestyle='dashed')

    ax1.axvline(x=time_step, color='black', linestyle='dashed')
    ax2.set_ylabel('Point Accuracy')
    ax2.legend()

    plt.xlabel("Epoch")
    plt.title("Gabor categorization accuracy and distances across training batches")



    plt.savefig(loc+f"/{is_sup} {str(time_step)} Lr={lr}.png")



    plt.show()



if __name__ == '__main__':
    plot_batch(time_step=0, csv_unsup="LR=0.0001, XAB Distance every batch unsup.csv", csv_sup="LR=0.0001, XAB Distance every batch sup, epochs.csv", num_unsup_rows=35, num_sup_rows=35, lr=0.0001, loc="", is_sup=False)

    #no train

    '''

    plot_batch_control(time_step=0, csv_unsup="LR=0.0001, Distance every batch unsup.csv",
               csv_sup="LR=0.0001, Distance every batch sup, 2 epochs.csv with size", num_unsup_rows=50, num_sup_rows=100, lr=0.0001,
               loc="whole_plots/blue-green/slow_lr_point_acc", is_sup="no_train_2_epochs")

    for i in range(1, 51): #unsup
        #plot_skip_batch(time_step=i, csv_unsup="LR=0.0001, Distance every batch unsup.csv", csv_sup="LR=0.0001, Distance every batch sup.csv", lr=0.0001, loc="whole_plots/skip_batch")
        plot_batch_control(time_step=i, csv_unsup="LR=0.0001, Distance every batch unsup.csv",
                           csv_sup="LR=0.0001, Distance every batch sup, 2 epochs.csv with size", num_unsup_rows=50, num_sup_rows=100, lr=0.0001,
                           loc="whole_plots/blue-green/slow_lr_point_acc", is_sup="unsup")



    for i in range(51, 50+100+1): #sup
        #plot_skip_batch(time_step=i, csv_unsup="LR=0.0001, Distance every batch unsup.csv", csv_sup="LR=0.0001, Distance every batch sup.csv", lr=0.0001, loc="whole_plots/skip_batch")
        plot_batch_control(time_step=i, csv_unsup="LR=0.0001, Distance every batch unsup.csv",
                           csv_sup="LR=0.0001, Distance every batch sup, 2 epochs.csv with size", num_unsup_rows=50, num_sup_rows=100, lr=0.0001,
                           loc="whole_plots/blue-green/slow_lr_point_acc", is_sup="sup")
    '''
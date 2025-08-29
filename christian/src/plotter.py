from christian.src.sup_trainer import Fast_lr_Freq_Trainer
from plotting import *
from scatter_plot import *
from combine_plots import *

class ParentPlotter:
    def __init__(self):
        self.unsup_csv = "LR=0.001, Distance every batch unsup.csv"
        self.scatter_plot_excel_file = "categorisation 4000.xlsx"
        self.num_unsup_rows = 100
        self.num_sup_rows = 100

    def plot_blue_green(self):
        for index in range(0, self.num_unsup_rows+self.num_sup_rows+1):
            print(index)
            plot_batch(time_step=index, csv_unsup=self.unsup_csv,
                                   csv_sup=self.sup_csv,
                                   num_unsup_rows=self.num_unsup_rows, num_sup_rows=self.num_sup_rows, lr=self.lr,
                                   loc=self.blue_green_dir)

    def plot_scatter_plots(self):
        for e in range(1):
            for b in range(100):
                weights = f"{self.weights_dir}/sup_net_weights_lr={self.lr} {e} {b}.pth"
                scatter_plot(train_type="sup", weights=weights, lr=self.lr, batch=b, epoch=e, loc=self.scatter_plots_dir)

    def combine_plots(self):
        for index in range(0, 201):
            if index == 0:
                a = f"../whole_plots/blue-green/fast_lr_freq/Lr=0.01 0.png"
                b = "../whole_plots/scatter_plots/no_train/no_training.png"
                merge(a, b, title="no_train ", loc=self.combined_dir)

            elif index < 101:
                a  = f"{self.blue_green_dir}/Lr={self.lr} {index}.png"
                b = f"../whole_plots/scatter_plots/unsup_rotated/unsup lr = 0.001, 0 {index-1}.png"
                merge(a, b, title=f"unsup {index} ", loc=self.combined_dir)

            else:
                a = f"{self.blue_green_dir}/Lr={self.lr} {index}.png"
                b = f"{self.scatter_plots_dir}/sup lr = {self.lr}, 0 {index - 101}.png"
                merge(a, b, title=f"z sup {index} ", loc=self.combined_dir)

class Slow_lr_Freq_Plotter(ParentPlotter):
    def __init__(self):
        super().__init__()
        self.sup_csv ="LR=0.001, Distance every batch sup, epochs.csv"
        self.blue_green_dir = "../whole_plots/blue-green/slow_lr_freq"
        self.lr = 0.001
        self.weights_dir = "../net_weights/sup"
        self.scatter_plots_dir= "../whole_plots/scatter_plots/sup_rotated"
        self.combined_dir = "../whole_plots/combined/slow_lr"


class Fast_lr_Freq_Plotter(ParentPlotter):
    def __init__(self):
        super().__init__()
        self.sup_csv ="LR=0.01, Distance every batch sup, epochs.csv"
        self.blue_green_dir = "../whole_plots/blue-green/fast_lr_freq"
        self.lr = 0.01
        self.weights_dir = "../net_weights/fast_lr_freq"
        self.scatter_plots_dir= "../whole_plots/scatter_plots/sup_lr_fast"
        self.combined_dir = "../whole_plots/combined/fast_lr"



def plot(plotter):
    plotter.plot_blue_green()
    plotter.plot_scatter_plots()
    plotter.combine_plots()


if __name__ == "__main__":
    plotter =  Slow_lr_Freq_Plotter()
    plot(plotter)

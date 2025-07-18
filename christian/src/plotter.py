from plotting import *
from scatter_plot import *
from combine_plots import *

class ParentPlotter:
    def __init__(self):
        self.unsup_csv = "LR=0.0001, Distance every batch unsup.csv"
        self.scatter_plot_excel_file = "categorisation 4000.xlsx"


    def plot_blue_green_labeled_by_freq(self):
        for index in range(0, 151):
            if index == 0:
                plot_batch_control(time_step=index, csv_unsup=self.unsup_csv,
                                   csv_sup=self.sup_csv,
                                   num_unsup_rows=50, num_sup_rows=100, lr=self.lr,
                                   loc=self.blue_green_dir, is_sup="no_train", is_control=self.is_control)
            elif index < 51:
                plot_batch_control(time_step=index, csv_unsup=self.unsup_csv,
                                   csv_sup=self.sup_csv,
                                   num_unsup_rows=50, num_sup_rows=100, lr=self.lr,
                                   loc=self.blue_green_dir, is_sup="unsup", is_control=self.is_control)

            else:
                plot_batch_control(time_step=index, csv_unsup=self.unsup_csv,
                                   csv_sup=self.sup_csv,
                                   num_unsup_rows=50, num_sup_rows=100, lr=self.lr,
                                   loc=self.blue_green_dir, is_sup="z sup", is_control=self.is_control)

    def plot_scatter_plots_labeled_by_freq(self):
        for e in range(1):
            for b in range(100):
                weights = f"../net_weights/{self.weights_dir}/sup_net_weights_lr={self.lr} {e} {b}.pth"
                scatter_plot(train_type="sup", weights=weights, lr=0.0001, batch=b, epoch=e, loc=self.scatter_plots_dir, excel_file=self.scatter_plot_excel_file)

    def combine_plots(self):
        for index in range(0, 151):
            if index == 0:
                a = f"{self.blue_green_dir}/no_train 0 Lr=0.0001.png"
                b = "../whole_plots/scatter_plots_freq/no_train/no_training.png"
                merge(a, b, title="no_train ", loc=self.combined_dir)

            elif index < 51:
                a  = f"{self.blue_green_dir}/unsup {index} Lr={self.lr}.png"
                b = f"../whole_plots/scatter_plots_freq/unsup, every batch/unsup lr = 0.0001, 0 {index-1}.png"
                merge(a, b, title=f"unsup {index} ", loc=self.combined_dir)

            else:
                a = f"{self.blue_green_dir}/z sup {index} Lr={self.lr}.png"
                b = f"{self.scatter_plots_dir}/sup lr = {self.lr}, 0 {index-51}.png"
                merge(a, b, title=f"z sup {index} ", loc=self.combined_dir)

class Slow_lr_Freq_Plotter(ParentPlotter):
    def __init__(self):
        super().__init__()
        self.sup_csv ="LR=0.0001 Slow_lr_Freq Distance every batch sup.csv"
        self.blue_green_dir = "../whole_plots/blue-green/slow_lr_freq"
        self.lr = 0.0001
        self.weights_dir = "../net_weights/slow_lr_freq"
        self.scatter_plots_dir= "../whole_plots/scatter_plots_freq/slow_lr_freq"
        self.is_control = False
        self.combined_dir = "../whole_plots/combined/slow_lr_with_control"

class Fast_lr_Freq_Plotter(ParentPlotter):
    def __init__(self):
        super().__init__()
        self.sup_csv ="LR=0.005 Fast_lr_Freq Distance every batch sup.csv"
        self.blue_green_dir = "../whole_plots/blue-green/fast_lr_freq"
        self.lr = 0.005
        self.weights_dir = "../net_weights/fast_lr_freq"
        self.scatter_plots_dir= "../whole_plots/scatter_plots_freq/fast_lr_freq"
        self.is_control = False

class Slow_lr_Size_Plotter(ParentPlotter):
    def __init__(self):
        super().__init__()
        self.sup_csv ="LR=0.0001 Slow_lr_Size Distance every batch sup.csv"
        self.blue_green_dir = "../whole_plots/blue-green/slow_lr_size"
        self.lr = 0.0001
        self.weights_dir = "../net_weights/slow_lr_size"
        self.scatter_plots_dir= "../whole_plots/scatter_plots_freq/slow_lr_size"
        self.is_control = True

class Fast_lr_Size_Plotter(ParentPlotter):
    def __init__(self):
        super().__init__()
        self.sup_csv ="LR=0.005 Fast_lr_Size Distance every batch sup.csv"
        self.blue_green_dir = "../whole_plots/blue-green/fast_lr_size"
        self.lr = 0.005
        self.weights_dir = "../net_weights/fast_lr_size"
        self.scatter_plots_dir= "../whole_plots/scatter_plots_freq/fast_lr_size"
        self.is_control = True

def plot(plotter):
    plotter.plot_blue_green_labeled_by_freq()
    plotter.plot_scatter_plots_labeled_by_freq()
    plotter.combine_plots()

if __name__ == "__main__":
    plotters = [Fast_lr_Freq_Plotter(), Slow_lr_Size_Plotter(), Fast_lr_Size_Plotter()]
    for plotter in plotters:
        plot(plotter)

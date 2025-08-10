from plotter import *
from sup_trainer import *
from train_unsupervised import *
from dist import *
import os

def run_slow_sup_lr():
    no_train_dist()
    #unsup_trainer()
    #sup_trainer = Slow_lr_Freq_Trainer()
    #sup_trainer.train()
    #plotter = Slow_lr_Freq_Plotter()
    #plot(plotter)

if __name__ == '__main__':
    run_slow_sup_lr()
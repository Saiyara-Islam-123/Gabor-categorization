from abc import ABC

from train_supervised import *

class ParentTrainer:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = 1


    def train(self):
        unsup_net = Net()

        weight_path = "../net_weights/unsup/unsup_net_weights_ lr= 0.0001 0 49.pth"
        unsup_net.load_state_dict(torch.load(weight_path))
        sup_net = SupervisedNet(unsup_net)

        train_supervised_control(model=sup_net, main_trainloader=self.main_trainloader, device=self.device, lr=self.lr, epochs=self.epochs, side_train_loader=self.side_trainloader, title=self.title, weights_dir=self.weights_dir, is_control=self.is_control)


class Slow_lr_Freq_Trainer(ParentTrainer):
    def __init__(self):
        super().__init__()
        self.main_trainloader, _, _ = load_gabor_data("categorisation 4000.xlsx", batch_size=32)
        excel_file = "Control/gabors_2/experimentFiles/categorisation.xlsx"
        base_dir = "C:\\Users\\Admin\\Documents\\GitHub\\Gabor-categorization\\christian\\src\\Control\\gabors_2\\"

        self.side_trainloader, _, _ = load_gabor_data(excel_file, batch_size=32, base_dir=base_dir)
        self.lr = 0.0001
        self.title = "Slow_lr_Freq"
        self.weights_dir = "slow_lr_freq"
        self.is_control = False

class Fast_lr_Freq_Trainer(ParentTrainer):
    def __init__(self):
        super().__init__()
        self.main_trainloader, _, _ = load_gabor_data("categorisation 4000.xlsx", batch_size=32)
        excel_file = "Control/gabors_2/experimentFiles/categorisation.xlsx"
        base_dir = "C:\\Users\\Admin\\Documents\\GitHub\\Gabor-categorization\\christian\\src\\Control\\gabors_2\\"

        self.side_trainloader, _, _ = load_gabor_data(excel_file, batch_size=32, base_dir=base_dir)
        self.lr = 0.005
        self.title = "Fast_lr_Freq"
        self.weights_dir = "fast_lr_freq"
        self.is_control =False

class Slow_lr_Size_Trainer(ParentTrainer):
    def __init__(self):
        super().__init__()
        self.side_trainloader, _, _ = load_gabor_data("categorisation 4000.xlsx", batch_size=32)
        excel_file = "Control/gabors_2/experimentFiles/categorisation.xlsx"
        base_dir = "C:\\Users\\Admin\\Documents\\GitHub\\Gabor-categorization\\christian\\src\\Control\\gabors_2\\"

        self.main_trainloader, _, _ = load_gabor_data(excel_file, batch_size=32, base_dir=base_dir)
        self.lr = 0.0001
        self.title = "Slow_lr_Size"
        self.weights_dir = "slow_lr_size"
        self.is_control = True

class Fast_lr_Size_Trainer(ParentTrainer):
    def __init__(self):
        super().__init__()
        self.side_trainloader, _, _ = load_gabor_data("categorisation 4000.xlsx", batch_size=32)
        excel_file = "Control/gabors_2/experimentFiles/categorisation.xlsx"
        base_dir = "C:\\Users\\Admin\\Documents\\GitHub\\Gabor-categorization\\christian\\src\\Control\\gabors_2\\"

        self.main_trainloader, _, _ = load_gabor_data(excel_file, batch_size=32, base_dir=base_dir)
        self.lr = 0.005
        self.title = "Fast_lr_Size"
        self.weights_dir = "fast_lr_size"
        self.is_control = True

if __name__ == "__main__":
    slow_freq_trainer = Slow_lr_Freq_Trainer()
    slow_freq_trainer.train()
from Transformer import *
from train_supervised import *

class ParentTrainer:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = 10


    def train(self):
        unsup_net = Net()

        unsup_net.load_state_dict(torch.load("../net_weights/Prev runs/"+self.unsup_weight_path+"/unsup_weights_ lr= 0.005 0 99.pth"))
        sup_net = SupervisedNet(unsup_net)
        self.trainloader, _, _ = load_gabor_data("categorisation 4000.xlsx", batch_size=32)

        train_supervised(model=sup_net, trainloader=self.trainloader, device=self.device, lr=self.lr, epochs=1, dist_func = sampled_all_distance, weights_dir=self.weights_dir, csv_dir=self.csv_dir)

class Slow_lr_Freq_Trainer(ParentTrainer):
    def __init__(self, unsup_weight_path, sup_weights_dir, csv_dir):
        super().__init__()
        self.lr = 0.001
        self.weights_dir = sup_weights_dir
        self.unsup_weight_path = unsup_weight_path
        self.csv_dir = csv_dir


class Fast_lr_Freq_Trainer(ParentTrainer):
    def __init__(self, unsup_weight_path, sup_weights_dir, csv_dir):
        super().__init__()
        self.lr = 0.01
        self.weights_dir = sup_weights_dir
        self.unsup_weight_path = unsup_weight_path
        self.csv_dir = csv_dir

class XABTrainer(ParentTrainer):
    def __init__(self):
        super().__init__()
        self.main_trainloader, _, _ = load_gabor_data("categorisation_xab.xlsx", batch_size=32)
        self.lr = 0.0001
        self.title = "Slow_lr_Freq_400"
        self.weights_dir = "slow_lr_freq_400"

    def train(self):
        unsup_net = Net()

        weight_path = "../net_weights/unsup/unsup_net_weights_ lr= 0.005 0 99.pth"
        unsup_net.load_state_dict(torch.load(weight_path))
        sup_net = SupervisedNet(unsup_net)

        train_supervised(model=sup_net, trainloader=self.main_trainloader, device=self.device, lr=self.lr, epochs=5, dist_func=xab_pairs_dist, weights_dir=self.weights_dir)

class TransformerTrainer(ParentTrainer):
    def __init__(self):
        super().__init__()

    def train(self):
        unsup_net = Transformer()

        unsup_net.load_state_dict(torch.load("../net_weights/Transformer_unsup/unsup_weights_ lr= 0.001 2 32.pth"))
        sup_net = SupNetwork(unsup_net)
        self.trainloader, _, _ = load_gabor_data("categorisation 4000.xlsx", batch_size=100)

        train_supervised(model=sup_net, trainloader=self.trainloader, device=self.device, lr=0.01, epochs=3, dist_func = sampled_all_distance, weights_dir="../net_weights/Transformer_sup", csv_dir=".")




if __name__ == "__main__":
    freq_trainer = TransformerTrainer()
    freq_trainer.train()


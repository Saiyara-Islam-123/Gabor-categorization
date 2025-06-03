import torch
from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoded = None

        self.encoder = torch.nn.Sequential(

            nn.Flatten(),
            nn.Linear(128 * 128 * 3, 600),
            nn.ReLU(),
            nn.Linear(600, 500),
            nn.ReLU(),
            nn.Linear(500, 128),

        )

        self.decoder = torch.nn.Sequential(

            nn.Linear(128, 500),
            nn.ReLU(),
            nn.Linear(500, 600),
            nn.ReLU(),
            nn.Linear(600, 128*128*3),
            nn.Unflatten(1, (3, 128, 128)),

            nn.Sigmoid()

        )


    def forward(self, x):
        self.encoded = self.encoder(x)
        decoded = self.decoder(self.encoded)
        return decoded


################################################################################################

class LastLayer(nn.Module):
    def __init__(self, autoencoder):
        super(LastLayer, self).__init__()

        self.encoder_output = None
        self.encoder = autoencoder.encoder

        self.supervised_part = nn.Sequential(nn.Linear(128, 2),

                                             )


    def forward(self, x):
        x = self.encoder(x)

        self.encoder_output = x



        x = self.supervised_part(x)

        return x
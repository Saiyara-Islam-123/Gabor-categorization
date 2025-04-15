import torch
from torch import nn


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoded = None

        self.encoder = torch.nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),

            nn.Flatten(),
            nn.Linear(64 * 33 * 33, 128),
            nn.ReLU(True)


        )



        self.decoder = torch.nn.Sequential(

            nn.Linear(128, 64 * 33 * 33),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 33, 33)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0),
            # Correcting stride/padding
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=0),  # Exact match for 28x28
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

        x = x.view(x.size(0), -1)

        x = self.supervised_part(x)

        return x
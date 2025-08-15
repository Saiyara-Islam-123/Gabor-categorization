import math
import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):


    def __init__(self):
        super().__init__()
        self.encoded = None

        self.nn1_encoder = nn.Sequential(

                nn.Conv2d(3,    8, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64*64
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  #64*64 -> 32 * 32
                nn.ReLU(),


        )#batch, 16*32*32 output -> turn into batch, 16, 32*32

        self.nn1_decoder = nn.Sequential(
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, output_padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1, output_padding=0),
                nn.Sigmoid(),

        )

        self.att_encoder = nn.MultiheadAttention(16, 4, batch_first=True)
        self.att_decoder = nn.MultiheadAttention(16, 4, batch_first=True)

        self.nn2_encoder = nn.Sequential(

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8X8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 8X8 -> 4X4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 500),
            nn.ReLU(),

        )

        self.nn2_decoder = nn.Sequential(
            nn.Linear(500, 2048),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),

        )
        self.positional_encoder = PositionalEncoding(16)

    def forward(self, x):
        encoder1_outputs = []
        encoder2_outputs = []
        for layer in self.nn1_encoder:
            x = layer(x)
            encoder1_outputs.append(x)
        batch_size = x.size(0)

        x = x.reshape(32*32, batch_size, 16).clone()
        x = self.positional_encoder(x)
        x = x.reshape(batch_size, 32*32, 16).clone()
        print("Pre attention")

        x, _ = self.att_encoder(x, x, x)
        x = x.reshape(batch_size, 16, 32,32).clone()

        for layer in self.nn2_encoder:
            x = layer(x)
            encoder2_outputs.append(x)

        self.encoded = x

        encoder2_outputs = encoder2_outputs[::-1]
        ########################################################

        for i, layer in enumerate(self.nn2_decoder):
            if isinstance(layer, nn.ConvTranspose2d) and i < len(encoder2_outputs):
                x = x + 0.5 * encoder2_outputs[i]
            x = layer(x)

        x = x.reshape(batch_size, 32 * 32, 16).clone()

        x, _ = self.att_decoder(x, x, x)
        x = x.reshape(batch_size, 16, 32, 32).clone()
        print("Post attention")
        encoder1_outputs = encoder1_outputs[::-1]
        for i, layer in enumerate(self.nn1_decoder):
            if isinstance(layer, nn.ConvTranspose2d) and i < len(encoder1_outputs):
                x = x + 0.5 * encoder1_outputs[i]
            x = layer(x)
        return x

class SupNetwork(nn.Module):
    def __init__(self, transformer):
        super(SupNetwork, self).__init__()
        self.transformer = transformer
        self.encoder_output = None

        self.classifier = nn.Sequential(
            nn.Linear(500, 128),
            nn.ReLU(),
            nn.Linear(128, 2),

        )
        self.positional_encoder = PositionalEncoding(16)

    def forward(self, x):

        for layer in self.transformer.nn1_encoder:
            x = layer(x)
        batch_size = x.size(0)

        x = x.reshape(batch_size, 16, 32*32)
        x = x.permute(2, 0, 1)

        x = self.positional_encoder(x)

        #32*32, batch_size, 16
        x= x.permute(1, 0, 2)

        x, _ = self.transformer.att_encoder(x, x, x)

        #batch_size, 32*32, 16
        x = x.reshape(batch_size, 32,32, 16)
        x = x.permute(0, 3, 1,2)

        for layer in self.transformer.nn2_encoder:
            x = layer(x)
        self.encoder_output = x
        return x
import math
import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):

    def __init__(self, d_model=16*16, dropout= 0.1, max_len = 5000):
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

        x = x + self.pe[:x.size(0)] #adding positional encoding to matrix
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoded = None

        self.nn1_encoder = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),


        )#batch, 16*16*16 output -> turn into batch, 16, 16*16

        self.nn1_decoder = nn.Sequential(

            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )

        self.att = nn.ModuleList(nn.MultiheadAttention(16*16, 16, batch_first=True) for _ in range(5))

        self.nn2_encoder = nn.Sequential(

            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),

        )

        self.nn2_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Unflatten(1, (16, 4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, output_padding=0),

        )
        self.positional_encoder = PositionalEncoding()

    def forward(self, x):
        x = self.nn1_encoder(x)
        x = x.reshape(16 ,x.size(0) , 16*16).clone()
        x = self.positional_encoder(x)
        x = x.reshape(x.size(1), 16, 16*16).clone()

        for multihead in self.att:
            x, _ = multihead(x, x, x)
        x = x.reshape(x.size(0), 16, 16,16).clone()

        x = self.nn2_encoder(x)
        self.encoded = x
        ########################################################
        x = self.nn2_decoder(x)

        x = x.reshape(x.size(0), 16, 16 * 16).clone()

        for multihead in self.att:
            x, _ = multihead(x, x, x)
        x = x.reshape(x.size(0), 16, 16, 16).clone()
        x = self.nn1_decoder(x)

        return x

class SupNetwork(nn.Module):
    def __init__(self, transformer):
        super(SupNetwork, self).__init__()
        self.transformer = transformer
        self.encoder_output = None
        self.classifier = nn.Sequential(
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.transformer(x)
        self.encoder_output = x
        x = self.classifier(x)
        return x

import math
import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):

    def __init__(self, d_model=16, dropout= 0.1, max_len = 5000):
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

            nn.Conv2d(3, 16, kernel_size=68, stride=2, padding=1),  # 128x128 -> 32X32
            nn.ReLU(),

        )#batch, 16*32*32 output -> turn into batch, 16, 32*32

        self.nn1_decoder = nn.Sequential(

            nn.ConvTranspose2d(16, 3, kernel_size=68, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )

        self.att_encoder = nn.ModuleList(nn.MultiheadAttention(32*32, 32, batch_first=True) for _ in range(10))
        self.att_decoder = nn.ModuleList(nn.MultiheadAttention(32 * 32, 32, batch_first=True) for _ in range(10))

        self.nn2_encoder = nn.Sequential(

            nn.Conv2d(16, 16, kernel_size=28, stride=2, padding=1),  # 32x32 -> 4x4
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
            nn.ConvTranspose2d(16, 16, kernel_size=28, stride=2, padding=1, output_padding=0),
        )
        self.positional_encoder = PositionalEncoding()

    def forward(self, x):
        encoder_outputs = []
        for layer in self.nn1_encoder:
            x = layer(x)
            encoder_outputs.append(x)
        batch_size = x.size(0)
        x = x.reshape(32*32, batch_size, 16).clone()
        x = self.positional_encoder(x)
        x = x.reshape(batch_size, 16, 32*32).clone()
        for multihead in self.att_encoder:
            x, _ = multihead(x, x, x)
        x = x.reshape(x.size(0), 16, 32,32).clone()

        for layer in self.nn2_encoder:
            x = layer(x)
            encoder_outputs.append(x)

        self.encoded = x

        encoder_outputs = encoder_outputs[::-1]
        ########################################################
        for i, layer in enumerate(self.nn2_decoder):
            if isinstance(layer, nn.ConvTranspose2d) and i < len(encoder_outputs):
                x = x + 0.5 * encoder_outputs[i]
            x = layer(x)

        x = x.reshape(x.size(0), 16, 32 * 32).clone()
        for multihead in self.att_decoder:
            x, _ = multihead(x, x, x)
        x = x.reshape(x.size(0), 16, 32, 32).clone()

        x= self.nn1_decoder(x)
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

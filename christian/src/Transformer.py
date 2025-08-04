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
        x = x + self.pe[:x.size(0)] #adding positional encoding to matrix
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoded = None

        self.nn1_encoder = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(True),


        ) #batch, 16*16*16 output -> turn into batch, 16, 16*16

        self.att = nn.ModuleList(nn.MultiheadAttention(16*16, 16, batch_first=True) for _ in range(5))

        self.nn2_encoder = nn.Sequential(

            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(True),

        )

    def forward(self, x):
        x = self.nn1_encoder(x)
        x = x.reshape(x.size(0), 16, 16*16)
        positional_encoder = PositionalEncoding()
        #x = positional_encoder(x)

        for multihead in self.att:
            x, _ = multihead(x, x, x)
        x = x.reshape(x.size(0), 16, 16,16)
        x = self.nn2_encoder(x)
        self.encoded = x
        return x

if __name__ == '__main__':
    t = Transformer()
    input = torch.randn(1, 3, 128, 128)
    output = t(input)
    print(output.size())
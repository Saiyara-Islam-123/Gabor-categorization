import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim=2, latent_dim=32):
        super(Net, self).__init__()
        # Encoder: Fully Connected (MLP) layers for 2D ring points (x, y)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),

            nn.Linear(32, latent_dim),
            #nn.LayerNorm(latent_dim),
            #nn.GELU()
        )

        # Decoder: mirrors the encoder back to 2D coordinates
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            #nn.LayerNorm(32),
            #nn.GELU(),

            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),                  

            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, input_dim)  # final reconstruction of (x, y)
            # No sigmoid/tanh: ring coordinates can be slightly >1 due to jitter
        )

    def forward(self, x):
        """
        Forward pass through the MLP autoencoder:
        - x is expected to be shape [batch, 2] containing (x, y) ring coordinates.
        - Returns reconstruction with the same shape.
        """
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    # If you ever want to experiment with skip connections for MLPs, you can
    # adapt a residual-style path between symmetric layers. For now, the
    # straightforward encoder->decoder path is sufficient for rings.


class SupervisedNet(nn.Module):
    def __init__(self, autoencoder, latent_dim=32, num_classes=2):
        """
        Supervised classifier that reuses the trained autoencoder's encoder.
        Expects the encoder to output a 'latent_dim'-dimensional feature vector.
        """
        super(SupervisedNet, self).__init__()
        # Reuse the encoder from the autoencoder
        self.encoder = autoencoder.encoder

        # Simple linear head for 2-way classification
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, x):
        # x should be [batch, 2] ring coordinates during classification too
        feats = self.encoder(x)
        logits = self.classifier(feats)
        return logits


if __name__ == "__main__":
    # Instantiate the autoencoder (MLP)
    model = Net(input_dim=2, latent_dim=256)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Print the model architecture
    print(model)

    # Instantiate the supervised network using the encoder from the autoencoder
    supervised_model = SupervisedNet(model, latent_dim=256, num_classes=2)
    supervised_model.to(device)

    # Print the supervised model architecture (optional)
    # print(supervised_model)

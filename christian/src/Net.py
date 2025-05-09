import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Encoder: Convolutional layers
        self.encoded = None

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.BatchNorm2d(16),  # Batch normalization after the Conv2d layer
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),  # Output of the fully connected layer now has 256 units
            nn.BatchNorm1d(256),  # Batch normalization for fully connected layer
            nn.ReLU(True)
        )

        # Decoder: Ensuring the output matches 128x128
        self.decoder = nn.Sequential(
            nn.Linear(256, 64 * 16 * 16),  # Input layer updated for 256 units
            nn.BatchNorm1d(64 * 16 * 16),  # Batch normalization for the Linear layer
            nn.ReLU(True),
            nn.Unflatten(1, (64, 16, 16)),  # Unflatten to (64, 16, 16)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32 -> 64x64
            nn.BatchNorm2d(16),  # Batch normalization
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64 -> 128x128
            nn.Sigmoid()  # Sigmoid to ensure values are between 0 and 1
        )

    def forward(self, x):
        """
        Performs a forward pass on the neural network composed of an encoder and a
        decoder, which utilizes skip connections for information flow. The method
        iteratively processes the input tensor `x` through the encoder layers, stores
        intermediate results for skip connections, and then passes the processed
        tensor through the decoder layers, applying skip connections where applicable.

        :param x: The input tensor to the model.
        :type x: torch.Tensor
        :return: The output tensor after processing through the encoder-decoder
                 architecture with skip connections.
        :rtype: torch.Tensor
        """
        encoder_outputs = []

        # Encoder forward pass
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)
            self.encoded = x

        # Reverse the encoder outputs for skip connections
        encoder_outputs = encoder_outputs[::-1]

        # Decoder forward pass with skip connections
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.ConvTranspose2d) and i < len(encoder_outputs):
                x = x + 0.5 * encoder_outputs[i]
            x = layer(x)

        return x

    #def forward(self, x):
    #    x = self.encoder(x)
    #    x = self.decoder(x)
    #    return x



class SupervisedNet(nn.Module):
    def __init__(self, autoencoder):
        """
        SupervisedNet is a neural network model designed for classification tasks. It utilizes
        a pre-trained autoencoder's encoder as its feature extractor, followed by a classifier
        module to perform the prediction. The classifier is expected to output two-dimensional
        predictions corresponding to the target classes. This class assumes that the passed
        autoencoder has a properly trained encoder.

        Attributes:
            encoder : nn.Module
                The encoder part of the autoencoder, used as a feature extractor.
            classifier : nn.Module
                A sequential module containing layers for classification.

        :param autoencoder: A pre-trained autoencoder, from which the encoder will be
            extracted and used in the supervised network.
        :type autoencoder: nn.Module
        """
        super(SupervisedNet, self).__init__()
        # Load the encoder from the autoencoder
        self.encoder = autoencoder.encoder
        self.encoder_output = None
        # Classifier layer on top of the encoder
        self.classifier = nn.Sequential(
            nn.Linear(256, 2)  # Adjusted input size of 256 for the updated encoder
        )

    def forward(self, x):
        # Pass through the encoder
        x = self.encoder(x)
        # Pass the encoder's output through the classifier
        self.encoder_output = x
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Instantiate the autoencoder
    model = Net()

    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Print the model architecture
    print(model)

    # Instantiate the supervised network using the encoder from the autoencoder
    supervised_model = SupervisedNet(model)
    supervised_model.to(device)

    # Print the supervised model architecture
    #print(supervised_model)

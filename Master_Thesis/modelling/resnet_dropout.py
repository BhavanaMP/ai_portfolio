import torch.nn as nn
from segmentation_models_pytorch.encoders.resnet import ResNetEncoder


class ResNetEncoderDropout(ResNetEncoder):
    def __init__(self, dropout_prob=0.5, *args, **kwargs):
        """
        Adding Dropout in ResNet
        """
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        features = []   # List of features from each layer
        features.append(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.dropout(x)
        features.append(x)

        x = self.layer2(x)
        x = self.dropout(x)
        features.append(x)

        x = self.layer3(x)
        x = self.dropout(x)
        features.append(x)

        x = self.layer4(x)
        x = self.dropout(x)
        features.append(x)

        return features

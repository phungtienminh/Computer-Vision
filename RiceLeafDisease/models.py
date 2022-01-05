import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


class Classifier(nn.Module):
    def init_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)

    def __init__(self):
        super().__init__()

        self.vgg_block_1 = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.vgg_block_2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.vgg_block_3 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.vgg_block_1.apply(self.init_weights)
        self.vgg_block_2.apply(self.init_weights)
        self.vgg_block_3.apply(self.init_weights)

        self.maxpool = nn.MaxPool2d((2, 2), stride=2)
        self.dropout = nn.Dropout(0.5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 28 * 28, 64)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.a1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 4)
        nn.init.xavier_normal_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.vgg_block_1(x)
        x = self.maxpool(x)
        if self.training:
            x = self.dropout(x)

        x = self.vgg_block_2(x)
        x = self.maxpool(x)
        if self.training:
            x = self.dropout(x)

        x = self.vgg_block_3(x)
        x = self.maxpool(x)
        if self.training:
            x = self.dropout(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.a1(x)
        if self.training:
            x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.softmax(x)
        return x


def load_model():
    model = Classifier()
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))
        model.load_state_dict(torch.load('../model.pth'))
    else:
        model.load_state_dict(torch.load('../model.pth', map_location = torch.device('cpu')))

    model.eval()
    return model

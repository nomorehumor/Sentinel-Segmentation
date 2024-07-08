import torch.nn as nn
import torch
import torch.nn.functional as F


class SegmentationModel(nn.Module):
    def __init__(self, num_channels=4, dropout_rate=0.2):
        super(SegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 1, kernel_size=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x
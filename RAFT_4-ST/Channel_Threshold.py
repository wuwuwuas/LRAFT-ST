import torch
import torch.nn as nn
class channel_threshold(nn.Module):
    def __init__(self,C):
        super(channel_threshold, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(C, C)
        self.bn = nn.BatchNorm1d(C)
        self.fc2 = nn.Linear(C, C)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,features):
        features_abs = features.abs()
        abs_mean = self.avg(features_abs).view(features.shape[0],features.shape[1])
        scales = self.fc1(abs_mean)
        scales = self.bn(scales)
        scales = self.relu(scales)
        scales = self.fc2(scales)
        scales = self.sigmoid(scales)
        thres = abs_mean * scales
        zeros = torch.zeros_like(features)
        thres = thres.unsqueeze(2).unsqueeze(2) + zeros
        sub = features_abs - thres
        n_sub = torch.max(sub, zeros)
        features = torch.sign(features) * n_sub
        return features
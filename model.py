import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, input_features):
        super(ResidualBlock, self).__init__()

        self.fc1 = nn.Linear(input_features, input_features)
        self.fc2 = nn.Linear(input_features, input_features)
        self.fc3 = nn.Linear(input_features, input_features)

    def forward(self, x):
        residual = x
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out += residual
        out = F.relu(out)
        return out


class NNewMLPNet(nn.Module):

    def __init__(self):
        super(NNewMLPNet, self).__init__()
        self.fc1 = nn.Linear(784, 392, bias=True)
        self.fc2 = nn.Linear(392, 196, bias=True)
        self.fc3 = nn.Linear(196, 98, bias=True)
        self.fc4 = nn.Linear(98, 49, bias=True)
        self.fc5 = nn.Linear(49, 10, bias=True)

        self.dropout = nn.Dropout(p=0.3)

        self.batch_norm1 = nn.BatchNorm1d(392)
        self.batch_norm2 = nn.BatchNorm1d(196)
        self.batch_norm3 = nn.BatchNorm1d(98)
        self.batch_norm4 = nn.BatchNorm1d(49)

        self.residual_block1 = ResidualBlock(256)
        self.residual_block2 = ResidualBlock(128)
        self.residual_block3 = ResidualBlock(64)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.batch_norm3(x)
        x = self.dropout(x)

        x = F.relu(self.fc4(x))
        x = self.batch_norm4(x)
        x = self.dropout(x)

        x = F.log_softmax(self.fc5(x), dim=1)
        return x

# class CankMLPNet(nn.Module):
#     # 训练30个epochs后测试集准确率高达93.8%
#
#     def __init__(self):
#         super(CankMLPNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 128, 1, padding=1)
#         self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.bn1 = nn.BatchNorm2d(128)
#         self.relu1 = nn.ReLU()
#
#         self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
#         self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2, padding=1)
#         self.bn2 = nn.BatchNorm2d(256)
#         self.relu2 = nn.ReLU()
#
#         self.fc5 = nn.Linear(256 * 8 * 8, 512)
#         self.drop1 = nn.Dropout2d()
#         self.fc6 = nn.Linear(512, 128)
#         self.fc7 = nn.Linear(128, 10)
#
#         self.dropout2 = nn.Dropout(p=0.3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#
#         x = self.pool1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#
#         x = self.conv3(x)
#         x = self.conv4(x)
#
#         x = self.pool2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#
#         # print(" x shape ",x.size())
#         x = x.view(-1, 256 * 8 * 8)
#         x = F.relu(self.fc5(x))
#         x = self.drop1(x)
#         x = self.fc6(x)
#         x = self.dropout2(x)
#         x = self.fc7(x)
#         return x
#
#
# class NewMLPNet(nn.Module):
#     def __init__(self):
#         super(NewMLPNet, self).__init__()
#
#         self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu1 = nn.ReLU()
#         self.mp1 = nn.MaxPool2d(3, 2, 1)
#
#         self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.relu2 = nn.ReLU()
#         self.mp2 = nn.MaxPool2d(3, 2, 1)
#
#         self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.relu3 = nn.ReLU()
#         self.mp3 = nn.MaxPool2d(3, 2, 1)
#
#         self.dense1 = nn.Linear(1024, 512)
#         self.dense1_1 = nn.Linear(512, 256)
#         self.dense2 = nn.Linear(256, 128)
#         self.dense2_1 = nn.Linear(128, 64)
#         self.dense3 = nn.Linear(64, 10)
#
#         self.flatten = nn.Flatten(start_dim=1)
#         self.dropout = nn.Dropout(p=0.1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.mp1(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.mp2(x)
#
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu3(x)
#         x = self.mp3(x)
#
#         x = self.flatten(x)
#
#         x = F.relu(self.dense1(x))
#         x = F.relu(self.dense1_1(x))
#         x = F.relu(self.dense2(x))
#         x = F.relu(self.dense2_1(x))
#         x = self.dropout(x)
#         x = self.dense3(x)
#
#         return F.log_softmax(x, dim=1)

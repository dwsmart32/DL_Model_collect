import torch.nn as nn

class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(32, 3, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)

        self.ups1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.ups2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.ups3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.th = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)  # batch_size x 32 x 48 x 48
        x = self.relu(x)

        x = self.conv2(x)  # batch_size x 64 x 24 x 24
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv3(x)  # bach_size x 128 x 12 x 12
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv4(x)  # batch_size x 128 x 12 x 12
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv5(x)  # batch_size x 128 x 12 x 12
        x = self.bn4(x)
        x = self.relu(x)

        x = self.ups1(x)
        x = self.conv6(x)  # batch_size x 64 x 24 x 24
        x = self.bn5(x)
        x = self.relu(x)

        x = self.ups2(x)
        x = self.conv7(x)  # batch_size x 32 x 48 x 48
        x = self.bn6(x)
        x = self.relu(x)

        x = self.ups3(x)
        x = self.conv8(x)  # batch_size x 3 x 96 x 96
        x = self.th(x)


        return x


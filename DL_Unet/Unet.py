import torch.nn as nn

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.ups1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv6 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.ups2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv7 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.ups3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv8 = nn.Conv2d(32, 3, 3, stride=1, padding=1)

        self.th = nn.Tanh()
        self.relu = nn.ReLU()

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, x):
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        x1 = self.conv1(x)  # batch_size x 32 x 48 x 48
        x1 = self.relu(x1)

        x2 = self.conv2(x1)  # batch_size x 64 x 24 x 24
        x2 = self.bn1(x2)
        x2 = self.relu(x2)

        x3 = self.conv3(x2)  # bach_size x 128 x 12 x 12
        x3 = self.bn2(x3)
        x3 = self.relu(x3)

        x4 = self.conv4(x3)  # batch_size x 128 x 12 x 12
        x4 = self.bn3(x4)
        x4 = self.relu(x4)

        x5 = self.conv5(x4)  # batch_size x 128 x 12 x 12
        x5 = self.bn4(x5)
        x5 = self.relu(x5)

        x5 = x5 + x3

        x6 = self.ups1(x5)
        x6 = self.conv6(x6)  # batch_size x 64 x 24 x 24
        x6 = self.bn5(x6)
        x6 = self.relu(x6)

        x6 = x6 + x2

        x7 = self.ups2(x6)
        x7 = self.conv7(x7)  # batch_size x 32 x 48 x 48
        x7 = self.bn6(x7)
        x7 = self.relu(x7)

        x7 = x7 + x1

        x8 = self.ups3(x7)
        x8 = self.conv8(x8)  # batch_size x 3 x 96 x 96
        x8 = self.th(x8)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return x8


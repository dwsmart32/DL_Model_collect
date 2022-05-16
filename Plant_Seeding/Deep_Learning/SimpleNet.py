import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 맨처음에 3개가 들어오잖아 RGB, 32개필터를 만들어버려, 필터사이즈는 3이야
        # self.conv1 = nn.Conv2d(input, output, filtersize, stride=2, padding=1)
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.fc = nn.Linear(512 * 3 * 3, 12)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x  # batch_size x 3 x 96 x 96

        x = self.conv1(x)  # batch_size x 32 x 48 x 48
        x = self.relu(x)

        x = self.conv2(x)  # batch_size x 64 x 24 x 24
        x = self.relu(x)

        x = self.conv3(x)  # bach_size x 128 x 12 x 12
        x = self.relu(x)

        x = self.conv4(x)  # batch_size x 256 x 6 x 6
        x = self.relu(x)

        x = self.conv5(x)  # batch_size x 512 x 3 x 3
        x = self.relu(x)

        x = x.view(-1, 512 * 3 * 3)  # batch_size x 512 * 3 * 3
        x = self.fc(x)  # batch_size x 12

        return x

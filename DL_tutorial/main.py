
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import torch
import torch.nn as nn
from tqdm import tqdm


from train import train
from test import test
from utils import Dataset, Dataloader
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))

    batch_size = 32


    trainset = Dataset(train=True)
    trainloader = Dataloader(trainset, batch_size, train=True)

    testset = Dataset(train=False)
    testloader = Dataloader(testset, batch_size, train=False)

    # # get some random training images
    # dataiter = iter(trainloader)
    # images, label = dataiter.next()
    #
    # # show images
    # ####imshow(torchvision.utils.make_grid(images[:8]))
    # print("Label index: {0}".format(label[:8]))
    # print(images.size())


    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            # 맨처음에 3개가 들어오잖아 RGB, 32개필터를 만들어버려, 필터사이즈는 3이야
            # self.conv1 = nn.Conv2d(input, output, filtersize, stride=2, padding=1)
            self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(128, 256, 4, stride=1, padding=0)  # 총 256장의 필터를 만들었어
            self.fc = nn.Linear(256, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x  # batch_size x 3 x 32 x 32 batch size의 개수만큼 있는데 각자 3,32,32짜리임

            x = self.conv1(x)  # batch_size x 32 x 16 x 16
            x = self.relu(x)

            x = self.conv2(x)  # batch_size x 64 x 8 x 8
            x = self.relu(x)

            x = self.conv3(x)  # bach_size x 128 x 4 x 4
            x = self.relu(x)

            x = self.conv4(x)  # batch_size x 256 x 1 x 1
            x = self.relu(x)

            x = x.view(-1, 256)  # batch_size x 256
            x = self.fc(x)  # batch_size x 10
            return x

    net = SimpleNet() # 객체 만들기
    net = net.to(device)
    print(net)

    epoch = 10
    lr = 0.001
    beta1 = 0.5
    beta2 = 0.999
    criterion = nn.CrossEntropyLoss()
    model_path = './cifar10_simple.pth'
    loss_list1 = []
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))
    train(net, trainloader, trainset, epoch, criterion, optimizer, model_path, loss_list1)

    test(net, testloader, testset, criterion)




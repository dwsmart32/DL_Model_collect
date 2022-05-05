import torch.optim

from Model import *
from train import *
from test import *
from torch.utils.tensorboard import SummaryWriter
import platform
from torchsummary import summary
from torchvision import models
import torch

if __name__ == '__main__':
    # seed fix
    torch.manual_seed(10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))

    predict_csv_path = './sample_submission.csv'
    train_data_path = './plant-seedlings-classification/train/'
    test_data_path = './plant-seedlings-classification/test/'
    #print_traindataset(train_data_path)

    batch_size = 32
    train_dataset, valid_dataset = Dataset(train_data_path, "train")
    test_dataset = Dataset(test_data_path, "test")

    trainloader = Dataloader(train_dataset, batch_size, "train")
    validloader = Dataloader(valid_dataset, batch_size, "valid")
    testloader = Dataloader(test_dataset, None, "test")

    #transfer_learning (resnet50)
    net = models.resnet50(pretrained=True)
    for p in net.parameters():
        p.requires_grad = False
    fc_input_dim = net.fc.in_features
    net.fc = nn.Linear(fc_input_dim, 12)

    epoch = 30
    learning_rate = 0.0001
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.99))

    net = net.to(device)
    model_path = './Model.ckpt'
    loss_list = []


    train_losses, train_acc, val_acc = train_net(net, trainloader, validloader, optimizer, epoch, device, loss_function)

    #tensorboard
    writer = SummaryWriter('logs/')
    for i in range(0, epoch):
        writer.add_scalars(f'train_losses', train_losses[i], i)
        writer.add_scalars(f'train_acc', train_acc[i], i)
        writer.add_scalars(f'val_acc', val_acc[i], i)

    #csv update 완료
    test_net(net, test_data_path, predict_csv_path)

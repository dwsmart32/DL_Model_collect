import shutil

import torch.optim
from train import *
from test import *
from torch.utils.tensorboard import SummaryWriter
import platform
from torchsummary import summary
from torchvision import models
import torch
from Unet import *

if __name__ == '__main__':

    # seed fix
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))

    #Dataset path
    Dataset_path = './plant-seedlings-classification/'
    predict_csv_file = Dataset_path + 'sample_submission.csv'
    train_data_path = Dataset_path + 'train/'
    test_data_path = Dataset_path + 'test/'

    # Print Dataset description
    #print_traindataset(train_data_path)

    batch_size = 32
    train_dataset, valid_dataset, _ = Dataset(train_data_path, "train")
    _, _, test_dataset = Dataset(test_data_path, "test")
    print('[Dataset description]')
    trainloader = Dataloader(train_dataset, batch_size, "train")
    validloader = Dataloader(valid_dataset, batch_size, "valid")

    #test_dataset for Kaggle testing
    testloader = Dataloader(test_dataset, batch_size, "test")

    # #transfer_learning (resnet50)
    # net = models.resnet50(pretrained=True)
    # for p in net.parameters():
    #     p.requires_grad = False
    # fc_input_dim = net.fc.in_features
    # net.fc = torch.nn.Linear(fc_input_dim, 12)

    net = Unet()
    epoch = 30
    learning_rate = 0.0002
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.99), weight_decay=0.1)
    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, betas=(0.5, 0.99), weight_decay=0.1)
    #optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate, betas=(0.5, 0.99), weight_decay=0.1)

    net = net.to(device)
    loss_list = []


    train_losses, train_acc, val_acc = train_net(net, trainloader, validloader, optimizer, epoch, device, loss_function)

    #tensorboard
    if os.path.isdir('./logs'): shutil.rmtree('./logs/')
    writer = SummaryWriter('logs/')

    for i in range(0, epoch):
        writer.add_scalars(f'Accuracy/train_acc+valid_acc', {'train_acc' : train_acc[i], 'valid_acc' : val_acc[i]}, i)
        writer.add_scalar(f'Loss/train_loss', train_losses[i], i)


    # test, csv update done
    test_net(net, testloader, test_data_path, predict_csv_file,device)
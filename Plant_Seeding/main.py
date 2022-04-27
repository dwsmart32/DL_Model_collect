import torch.optim

from Model import *
from train import *
from test import *
from torch.utils.tensorboard import SummaryWriter
import platform
from torchsummary import summary
from torchvision import models


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))

    predict_csv_path = './sample_submission.csv'
    train_data_path = './plant-seedlings-classification/train/'
    test_data_path = './plant-seedlings-classification/test/'
    #print_traindataset(train_data_path)

    batch_size = 32
    train_dataset, valid_dataset = Dataset(train_data_path, "train")
    #test_dataset = Dataset(test_data_path, "test")

    trainloader = Dataloader(train_dataset, batch_size, "train")
    validloader = Dataloader(valid_dataset, batch_size, "valid")
    #testloader = Dataloader(test_dataset, batch_size, "test")

    #transfer learning
    net = models.resnet50(pretrained=True)
    for p in net.parameters():
        p.requires_grad = False
    fc_input_dim = net.fc.in_features
    net.fc = nn.Linear(fc_input_dim, 12)

    epoch = 10
    learning_rate = [0.00001, 0.0001, 0.001]
    # beta1 = 0.5
    # beta2 = 0.999
    loss_function = [nn.CrossEntropyLoss().to(device), nn.MSELoss().to(device), nn.GaussianNLLLoss().to(device)]

    net = net.to(device)
    model_path = './Model.ckpt'
    loss_list = []


    #tensorboard
    writer = SummaryWriter('logs/')
    train_loss=[]
    train_accuracy=[]
    validation_accuracy=[]

    for lr in range(len(learning_rate)):
        for loss_fn in range(len(loss_function)):
            Optimizer = [torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.99)), torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)]
            for opt in(Optimizer):
                train_losses, train_acc, val_acc = train_net(net, trainloader, validloader, opt, epoch, device, loss_function)
                train_loss.append(np.round(np.array(train_losses), 4))
                train_accuracy.append(np.round(np.array(train_acc), 4))
                validation_accuracy.append(np.round(np.array(val_acc), 4))

    graph_n = len(loss_function)*len(Optimizer)
    for i in range(0, graph_n):
        for j in range(0, epoch):
            if i/2 ==0:
                writer.add_scalars(f'valid_acc/opt:Adam, loss_fn:{loss_fn}',
                                   {f'lr:{learning_rate[0]}': validation_accuracy[graph_n*i][j],
                                    f'lr:{learning_rate[1]}': validation_accuracy[graph_n*i + 1][j],
                                    f'lr:{learning_rate[2]}': validation_accuracy[graph_n*i + 2][j]}, j)
            else:
                writer.add_scalars(f'valid_acc/opt:SGD, loss_fn:{loss_fn}',
                                  {f'lr:{learning_rate[0]}': validation_accuracy[graph_n * i][j],
                                   f'lr:{learning_rate[1]}': validation_accuracy[graph_n * i + 1][j],
                                   f'lr:{learning_rate[2]}': validation_accuracy[graph_n * i + 2][j]}, j)

    for i in range(0, graph_n):
        for j in range(0, epoch):
            if i/2 ==0:
                writer.add_scalars(f'train_loss/opt:Adam, loss_fn:{loss_fn}',
                                   {f'lr:{learning_rate[0]}': train_loss[graph_n*i][j],
                                    f'lr:{learning_rate[1]}': train_loss[graph_n*i + 1][j],
                                    f'lr:{learning_rate[2]}': train_loss[graph_n*i + 2][j]}, j)
            else:
                writer.add_scalars(f'train_loss/opt:SGD, loss_fn:{loss_fn}',
                                  {f'lr:{learning_rate[0]}': train_loss[graph_n * i][j],
                                   f'lr:{learning_rate[1]}': train_loss[graph_n * i + 1][j],
                                   f'lr:{learning_rate[2]}': train_loss[graph_n * i + 2][j]}, j)



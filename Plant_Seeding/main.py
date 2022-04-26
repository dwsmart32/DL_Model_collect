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

    epoch = 20
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    criterion = nn.L1Loss()

    net = net.to(device)
    model_path = './Model.ckpt'
    loss_list = []
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))
    train_net(net, trainloader, validloader, optimizer, epoch, device)
    #loss_list2=train(net, device, trainloader, validloader, epoch, criterion, optimizer, model_path, loss_list2)

    # writer = SummaryWriter('logs/')
    # for i in range(len(loss_list2)):
    #     writer.add_scalars(f'loss_comparing', {
    #         'net': loss_list2[i],
    #     }, i)
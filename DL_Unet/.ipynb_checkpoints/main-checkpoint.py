import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from PIL import Image
from utils import Datadownload,Dataset, Dataloader
from utils import *
from Unet import *
from ColorizationNet import *
from train import *
from test import *
from torch.utils.tensorboard import SummaryWriter
import platform

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {}".format(device))

    if platform.system()[0:5] == 'Window':
        isMac = False
    elif platform.system()[0:5] == 'Darwin': # macbook
        isMac = True
    else :
        isMac = False
        print('error occurred')

    Datadownload(isMac)
    batch_size = 32
    train_dataset = Dataset("train")
    valid_dataset = Dataset("valid")
    test_dataset = Dataset("test")

    trainloader = Dataloader(train_dataset, batch_size, "train")
    validloader = Dataloader(valid_dataset, batch_size, "valid")
    testloader = Dataloader(test_dataset, batch_size, "test")

    # #####img print###
    # dataiter = iter(trainloader)
    # images, _ = dataiter.next()
    # grays = rgb_to_grayscale(images)
    # w, h = images.size(2), images.size(3)
    #
    # # show images
    # imshow(torchvision.utils.make_grid(grays[:8]))
    # imshow(torchvision.utils.make_grid(images[:8]))
    #
    # print("Size of image: {0}x{1}".format(w, h))
    # #####img print###

    epoch = 20
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    criterion = nn.L1Loss()

    net1 = ColorizationNet()
    net1 = net1.to(device)

    model_path = './colorization_net.ckpt'
    loss_list1 = []
    optimizer = torch.optim.Adam(net1.parameters(), lr=lr, betas=(beta1, beta2))
    loss_list1=train(net1, device, trainloader, validloader, epoch, criterion, optimizer, model_path, loss_list1)

    net2 = Unet()
    net2 = net2.to(device)
    model_path = './Unet2.ckpt'
    loss_list2 = []
    net2 = Unet()
    net2 = net2.to(device)
    optimizer = torch.optim.Adam(net2.parameters(), lr=lr, betas=(beta1, beta2))
    loss_list2=train(net2, device, trainloader, validloader, epoch, criterion, optimizer, model_path, loss_list2)

    writer = SummaryWriter('logs/')
    for i in range(len(loss_list2)):
        writer.add_scalars(f'loss_comparing', {
            'net1': loss_list1[i],
            'net2': loss_list2[i],
        }, i)
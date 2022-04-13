
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch

def imshow(img):  # function to show an image
    img = img.cpu().detach()
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(24, 12))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()

def Dataset(train=True):
    ##
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = None
    if train == True:
        dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=train_transform)
    else:
        dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=test_transform)

    return dataset


def Dataloader(dataset, batch_size, train=True):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    ##print values
    if train==True:
        print("Training data size : {}".format(len(dataset)))
    else:
        print("Test data size : {}".format(len(dataset)))

    return dataloader
import gdown
import zipfile
import os
import torchvision
import torch
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import numpy as np
import os

def Datadownload(isMac):
    if isMac:
        url = 'https://drive.google.com/uc?id=1iJEx2EzLVv51hi-K1oEFUOqw8LeKj5EK'
        output_name = 'flower102.zip'
        if not os.path.isfile(output_name):
            gdown.download(url, output_name, quiet=False)
            with zipfile.ZipFile('./' + output_name, 'r') as zip_ref:
                zip_ref.extractall('./')
        else:
            print("The dataset file already exist.")
    else:
        if os.isdir('./flower102'):
            print('datafile exists')
        else:
            print('this platform is Window, and No datafile exists')
            print('you need to download datafiles by your own.')


def Dataset(datatype):
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((96, 96), interpolation=InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((96, 96), interpolation=InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if datatype == "train":
        dataset = torchvision.datasets.ImageFolder('./flower102/train', transform=train_transform)
    elif datatype =="valid":
        dataset = torchvision.datasets.ImageFolder('./flower102/train', transform=test_transform)
    else:
        dataset = torchvision.datasets.ImageFolder('./flower102/test', transform=test_transform)

    return dataset


def Dataloader(dataset, batch_size, datatype):

    if datatype == "train":
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        print("Training data size : {}".format(len(dataset)))
    elif datatype == "valid":
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        print("Validation data size : {}".format(len(dataset)))
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        print("Test data size : {}".format(len(dataset)))

    return dataloader


def imshow(img):  # function to show an image
    img = img.cpu().detach()
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(12, 6))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()


def rgb_to_grayscale(batch):  # function to convert rgb to grayscale
    batch = batch.cpu().detach()
    batch = batch / 2 + 0.5
    grayimg_list = []
    for i in range(batch.size(0)):
        npimg = batch[i].numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        r, g, b = npimg[:, :, 0], npimg[:, :, 1], npimg[:, :, 2]
        grayimg = 0.2989 * r + 0.5870 * g + 0.1140 * b
        grayimg = grayimg[:, :, np.newaxis]
        grayimg = np.transpose(grayimg, (2, 0, 1))
        grayimg_list.append(grayimg)
    graybatch = np.array(grayimg_list)
    graybatch = graybatch * 2 - 1.0
    graybatch = torch.tensor(graybatch)

    return graybatch



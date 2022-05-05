import zipfile
import os
import torchvision
import torch
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import numpy as np
import os



def Dataset(root_dir, datatype):
    resize_factor=(640,640)
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(resize_factor, interpolation=InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.RandomHorizontalFlip(p=0.9),
            #torchvision.transforms.RandomCrop((10, 10), padding=0),
            torchvision.transforms.Normalize((0.3, 0.3, 0.3), (0.1, 0.1, 0.1))])

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(resize_factor, interpolation=InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.RandomHorizontalFlip(p=0.9),
            #torchvision.transforms.RandomCrop((10,10), padding=0),
            torchvision.transforms.Normalize((0.3, 0.3, 0.3), (0.1, 0.1, 0.1))])

    split_dataset = None

    if datatype == "train":
        dataset = torchvision.datasets.ImageFolder(root_dir, transform=train_transform)
        train_count = int(0.8*len(dataset))
        valid_count = len(dataset)-train_count
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_count, valid_count])

    elif datatype == "test":
        dataset = torchvision.datasets.ImageFolder(root_dir, transform=test_transform)

    if split_dataset == None:
        print(f'split_dataset is None.')

    return train_dataset,valid_dataset


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

def print_traindataset(train_data_path):
    total_img = 0
    print('[train_dataset]')
    for name in os.listdir(train_data_path):
        path, dirs, files = next(os.walk(train_data_path + '/' + name))
        file_count = len(files)
        print(f'{name}: {file_count}')
        total_img = total_img + file_count
    print(f'number_of_total_img : {total_img}')



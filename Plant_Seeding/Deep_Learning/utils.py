
import torchvision
import torch
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from Preprocessing_utils import *



class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def Dataset(root_dir, datatype):
    resize_factor=(96,96)
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(resize_factor, interpolation=InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.RandomHorizontalFlip(p=0.5),
            #torchvision.transforms.RandomCrop((1, 1), padding=0),
            torchvision.transforms.Normalize((0.3, 0.3, 0.3), (0.1, 0.1, 0.1))])

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(resize_factor, interpolation=InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.RandomHorizontalFlip(p=0.5),
            #torchvision.transforms.RandomCrop((10,10), padding=0),
            torchvision.transforms.Normalize((0.3, 0.3, 0.3), (0.1, 0.1, 0.1))])


    test_dataset=None;train_dataset=None;valid_dataset=None;

    if datatype == "train":
        dataset = torchvision.datasets.ImageFolder(root_dir, transform=train_transform)
        train_count = int(0.8*len(dataset))
        valid_count = len(dataset)-train_count
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_count, valid_count])

    elif datatype == "test":
        source_from = root_dir
        source_to = root_dir + 'dummy/'
        if not os.path.isdir(source_to): # if dummy folder not exist
            # get files(.png)
            get_files = os.listdir(source_from)
            # make dummy folder
            os.makedirs(source_to)

            for g in get_files:
                shutil.move(source_from + g, source_to)
            print('dummy folder is made successfully')
        else:
            print('dummy folder is already made in test_dataset')

        test_dataset = ImageFolderWithPaths(source_from,
                                            transform=test_transform
                                            )


    return train_dataset,valid_dataset, test_dataset


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


def DeleteAllFiles(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)
        print('[tensorboard] Remove logs folder for update')

    else:
        print('[tensorboard] do not need to logs delete filder')
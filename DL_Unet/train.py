import time

from torch import device
from utils import *



def train(net, device, trainloader, validloader, num_epoch, criterion, optimizer, model_path, loss_list):  # Function to train the network
    for epoch in range(num_epoch):
        start_time = time.time()
        running_loss = 0.0
        total = 0
        for i, data in enumerate(trainloader, 0):
            images, label = data
            grays = rgb_to_grayscale(images)
            images = images.to(device)
            grays = grays.to(device)

            optimizer.zero_grad()
            outputs = net(grays)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

        # print training stats
        average_loss = running_loss / total
        with torch.no_grad():
            validation_loss = val(net, device, epoch, validloader, criterion)
            loss_list.append(validation_loss)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print('Epoch: %d / Training Loss: %.4f / Validation Loss: %.4f / Time : %.2f (s)' %
              (epoch + 1, average_loss, validation_loss, elapsed_time))

        # if (epoch % 10 == 9) or (epoch == 0):  # show training samples per 10 epoch
        #     imshow(torchvision.utils.make_grid(grays[:8]))
        #     imshow(torchvision.utils.make_grid(outputs[:8]))
        #     imshow(torchvision.utils.make_grid(images[:8]))

    print('Finished Training')
    state = {'net': net.state_dict(), 'loss_list': loss_list, }
    torch.save(state, model_path)
    print('Saved Trained Model')
    return loss_list

def val(net, device, current_epoch, validloader, criterion):  # Function to validate the network
    net.eval()
    running_loss = 0.0
    total = 0
    for i, data in enumerate(validloader, 0):
        images, label = data
        grays = rgb_to_grayscale(images)
        images = images.to(device)
        grays = grays.to(device)
        outputs = net(grays)
        loss = criterion(outputs, images)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    average_loss = running_loss / total
    net.train()

    return average_loss
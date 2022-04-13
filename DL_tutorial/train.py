
import time
import torch
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(net, trainloader, trainset, num_epoch, criterion, optimizer, model_path, loss_list):  # Function to train the network
    net.train()
    writer = SummaryWriter('logs/')
    for epoch in range(num_epoch):
        start_time = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):  # 50000개에 대해서 loop를 돈다.
            # 즉 50000개 중 하나는 3,32,32 의 size를 가지고 있기에 배치 개념이 들어가진 않았다
            # Load data
            images, label = data
            images = images.to(device)
            label = label.to(device)

            # Feed-forward the input
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, label)

            # Backpropagation
            loss.backward()  # 백 프롶을 시켜서 loss를 작게 만들어
            optimizer.step()

            running_loss += loss.item()

        # print training stats
        average_loss = running_loss / len(trainset)
        loss_list.append(average_loss)  # epoch당 loss 값 기록.
        end_time = time.time()
        elapsed_time = end_time - start_time
        writer.add_scalar("Loss/train",average_loss,epoch)
        print('Epoch: %d / Training Loss: %.3f / Time : %.2f (s)' %
              (epoch + 1, average_loss, elapsed_time))

    print('Finished Training')
    state = {'net': net.state_dict(), 'loss_list': loss_list, }
    torch.save(state, model_path)
    print('Saved Trained Model')
    writer.close()
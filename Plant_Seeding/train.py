import time

import torch.nn as nn
from utils import *
import tqdm


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


def train_net(net, trainloader, test_loader, optimizer, epoch, device, loss_fn):
    train_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(epoch):
        running_loss = 0.0
        # 신경망을 훈련 모드로 설정
        net.train()

        total = 0
        n_acc = 0
        # 시간이 많이 걸리므로 tqdm을 사용해서 진행 바를 표시
        for i, (img, label) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
            img = img.to(device)
            label = label.to(device)
            h = net(img)
            loss = loss_fn(h, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = img.size(0)
            running_loss += loss.item()
            total += batch_size

            _, y_pred = h.max(1)
            n_acc += (label == y_pred).float().sum().item()
        train_losses.append(running_loss / i)
        # 훈련 데이터의 예측 정확도
        train_acc.append(n_acc / total)

        # 검증 데이터의 예측 정확도
        val_acc.append(eval_net(net, test_loader, device))
        # epoch의 결과 표시
        print(f'epoch: {epoch}, train_loss:{train_losses[-1]}, train_acc:{train_acc[-1]}'
              f',val_acc: {val_acc[-1]}', flush=True)


def eval_net(net, data_loader, device):
    # Dropout 및 BatchNorm을 무효화
    net.eval()
    ys = []
    ypreds = []
    for x, y in data_loader:
        # to 메서드로 계산을 실행할 디바이스 전송
        x = x.to(device)
        y = y.to(device)
        # 확률이 가장 큰 분류를 예측
        # 여기선 forward(추론) 계산이 전부이므로 자동 미분에 필요한 처리는 off로 설정해서 불필요한 계산을 제한다.
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)

    # 미니 배치 단위의 예측 결과 등을 하나로 묶는다
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    # 예측 정확도 계산
    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()




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
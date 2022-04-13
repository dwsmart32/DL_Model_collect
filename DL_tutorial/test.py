
import time
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(net, testloader, testset, criterion):  # Function to test the network
    start_time = time.time()
    running_loss = 0.0
    total = 0
    correct = 0
    net.eval()

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # Load data
            images, label = data
            images = images.to(device)
            label = label.to(device)

            # Feed-forward the input
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            # Test results
            total += label.size(0)
            correct += (predicted == label).sum().item()
            loss = criterion(outputs, label)
            running_loss += loss.item()

    # Print test stats
    accuracy = correct / total
    average_loss = running_loss / len(testset)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print('Test Loss: %.3f / Test Accuracy: %.2f / Time : %.2f (s)' %
          (average_loss, accuracy, elapsed_time))

    print('Finished Test')
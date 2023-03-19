import torch
import torch.nn.functional as F


def train(num_epoch, conv_model, exp_lr_scheduler, train_loader, optimizer, criterion):
    conv_model.train()
    exp_lr_scheduler.step()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.unsqueeze(1)
        data, target = data, target

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = conv_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            pass
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                num_epoch, (batch_idx + 1) *
                len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.data))


def evaluate(data_loader, conv_model):
    conv_model.eval()
    loss = 0
    correct = 0

    for data, target in data_loader:
        data = data.unsqueeze(1)
        data, target = data, target

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = conv_model(data)

        loss += F.cross_entropy(output, target, size_average=False).data

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(data_loader.dataset)

    print('\nAverage Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

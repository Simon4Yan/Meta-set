from __future__ import print_function

import sys

sys.path.append(".")
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from FD.regression import RegNet
from FD.regression import REG

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (var, mean, target, fid) in enumerate(train_loader):
        var, mean, target, fid = var.to(device), mean.to(device), target.to(device), fid.to(device)
        optimizer.zero_grad()
        output = model(var, mean, fid)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(var), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    pred_acc = []
    target_acc = []
    with torch.no_grad():
        for var, mean, target, fid in test_loader:
            var, mean, target, fid = var.to(device), mean.to(device), target.to(device), fid.to(device)
            output = model(var, mean, fid)
            pred_acc.append(output.cpu())
            target_acc.append(target.cpu())
            test_loss += F.smooth_l1_loss(output, target, reduction='sum').item()  # sum up batch loss

    R2 = r2_score(torch.cat(target_acc).numpy(), torch.cat(pred_acc).numpy())
    RMSE = mean_squared_error(torch.cat(target_acc).numpy(), torch.cat(pred_acc).numpy(), squared=False)
    MAE = mean_absolute_error(torch.cat(target_acc).numpy(), torch.cat(pred_acc).numpy())

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f} R2 :{:.4f} RMSE: {:.4f} MAE: {:.4f}\n'.format(test_loss, R2, RMSE, MAE))


def main():
    '''
    few tips:
    1) batch size: 8 or 16, small number of sample sets might use 8;
    2) step_size in scheduler: 20 or 30, small number of sample sets might use 30;
    3) epochs: please use integer multiples of step_size, like 8 * step_size;
               Because this project uses Adadelta as optimizer;
    4) how many sample sets? --> more than 2000 would be good for learning RegNet
    '''
    parser = argparse.ArgumentParser(description='PyTorch NN Regression')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=210, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='M',
                        help='Learning rate step gamma (default: 0.8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    data = sorted(os.listdir('./dataset_bg/'))
    acc = np.load('./learn/accuracy_mnist.npy')
    fid = np.load('./FD/fd_mnist.npy')
    feature_path = './FD/dataset_feature/'

    # select some samplet sets for validation (also used in Linear regression)
    index = 30

    train_data = data[index:]
    train_acc = acc[index:]
    train_fid = fid[index:]

    test_data = data[:index]
    test_acc = acc[:index]
    test_fid = fid[:index]

    train_loader = torch.utils.data.DataLoader(
        REG(feature_path, train_data, train_acc, train_fid),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        REG(feature_path, test_data, test_acc, test_fid),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = RegNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=args.gamma)

    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()
    if args.save_model:
        torch.save(model.state_dict(), "./FD/mnist_regnet.pt")


if __name__ == '__main__':
    main()

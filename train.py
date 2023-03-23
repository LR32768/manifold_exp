'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from dataset import CatDogDataset
from utils import progress_bar, mixup_data
from resnet import *
from vgg import *
import MLP

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training
def train(epoch, dataloader, net, optimizer, criterion, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # noise = torch.randn(inputs.shape).to(device)
        # inputs, lam = mixup_data(inputs, noise, args.alpha * (200-epoch) / 200+1, use_cuda=torch.cuda.is_available())
        # targets_a, targets_b = targets, 10*torch.ones(targets.shape).long().to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        #loss = mixup_criterion(criterion,outputs, targets_a, targets_b, lam)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/(batch_idx+1), 100.*correct/total

def test(args, net, generator, uncon, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx in range(args.test_num):
            inputs, targets = generator.sample(args.bs, uncon=uncon)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, args.test_num, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print(f"Acc is {acc}")

    return test_loss / (batch_idx+1), acc
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, f'./checkpoint/{args.name}/ckpt.pth')
    #     best_acc = acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--name', '-n', type = str, default='exp', help='name of the experiment')

    # Dataset hyperparameters
    parser.add_argument('--man_dim', type=int, default=5, help='manifold dimensionality')
    parser.add_argument('--num_imgs', type=int, default=1000, help='manifold dimensionality')
    parser.add_argument('--seed', type=int, default=None, help='seed for generating the dataset')
    parser.add_argument('--dir', type=str, default='./tmp', help='temporary directory for storing dataset')

    # Training settings
    parser.add_argument('--g_id', type=int, default=0, help='image generatator device id')
    parser.add_argument('--t_id', type=int, default=1, help='training device id')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--use_aug', action='store_true', help='whether to use data augmentation')
    parser.add_argument('--use_bn', action='store_true')

    # Training model
    parser.add_argument('--model', type=str, default='vgg13')
    parser.add_argument('--test_num', type=int, default=100, help='num of batches for testing')

    args = parser.parse_args()

    if not os.path.exists(f'./checkpoint/{args.name}'):
        os.mkdir(f'./checkpoint/{args.name}')

    g_device = f'cuda:{args.g_id}'
    t_device = f'cuda:{args.t_id}'

    # Data
    print('==> Preparing data..')

    generator, dataset = CatDogDataset(man_dim=args.man_dim, num_images=args.num_imgs, out_dir=args.dir,
                            use_aug=args.use_aug, seed=args.seed, device=g_device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=2)

    # Model
    print('==> Building model..')
    if args.model == 'vgg13_bn':
        net = vgg13_bn(num_classes=2)
    elif args.model == 'vgg13':
        net = vgg13(num_classes=2)
    elif args.model == 'resnet34':
        net = resnet18(num_classes=2)
    elif args.model.startswith('MLP'):
        net = MLP.__dict__[args.model]()
    net = net.to(t_device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    for epoch in range(0, args.num_epochs):
        train_loss, train_acc = train(epoch, dataloader, net, optimizer, criterion, t_device)
        scheduler.step()

    intest_loss, intest_acc = test(args, net, generator, False, t_device)
    outtest_loss, outtest_acc = test(args, net, generator, True, t_device)

    with open(f'{args.model}_log_{args.man_dim}.txt', 'a') as ff:
        ff.write(f"num sample: {args.num_imgs}\n")
        ff.write(f"{train_loss} {train_acc} {intest_loss} {intest_acc} {outtest_loss} {outtest_acc}\n")

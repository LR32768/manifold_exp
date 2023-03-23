'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from dataset import CatDogDataset
from utils import progress_bar, mixup_data
from resnet import *
from vgg import *
import MLP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--name', '-n', type = str, default='exp', help='name of the experiment')

    # Dataset hyperparameters
    parser.add_argument('--man_dim', type=int, default=5, help='manifold dimensionality')
    parser.add_argument('--num_imgs', type=int, default=1000, help='manifold dimensionality')
    parser.add_argument('--seed', type=int, default=None, help='seed for generating the dataset')
    parser.add_argument('--dir', type=str, default='./tmp', help='temporary directory for storing dataset')
    parser.add_argument('--skip_data', action='store_true', help='skip the data generation process')

    # Training settings
    parser.add_argument('--g_id', type=int, default=1, help='image generatator device id')
    parser.add_argument('--t_id', type=int, default=1, help='training device id')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--use_aug', action='store_true', help='whether to use data augmentation')
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--reg', type=float, default=0.0)
    parser.add_argument('--scale', type=float, default=0.5)

    # Training model
    parser.add_argument('--model', type=str, default='vgg13')
    parser.add_argument('--test_num', type=int, default=50, help='num of batches for testing')

    args = parser.parse_args()

    g_device = f'cuda:{args.g_id}'
    t_device = f'cuda:{args.t_id}'

    # Data
    print('==> Preparing data..')

    generator, dataset = CatDogDataset(man_dim=args.man_dim, num_images=args.num_imgs, out_dir=args.dir,
                            use_aug=args.use_aug, seed=args.seed, device=g_device)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    train_data = None
    train_label = []
    for i in range(len(dataset)):
        img, lbl = dataset[i]
        if train_data is not None:
            train_data = torch.cat((train_data, img.view(1,-1)), dim=0)
        else:
            train_data = img.view(1, -1)

        train_label.append(lbl)

    train_label = torch.tensor(train_label)

    print(train_data.size())
    print(train_label)

    # Model
    from ntk import ntk_train, ntk_pred

    alpha = ntk_train(train_data, train_label, chunk_size=40, reg=args.reg)
    print(f"The alpha is {alpha}")

    # Calculate the RKHS norm
    gap = torch.sqrt(2 * train_label.float().dot(alpha))
    print(gap)

    test_data = None
    avg_loss = 0
    correct = 0
    for batch_idx in range(args.test_num):
        img, lbl = generator.sample(args.bs, uncon=False)
        test_data = img.view(args.bs, -1).cpu()

        test_label = lbl
        pred = ntk_pred(train_data, test_data, alpha)

        loss = torch.square(pred.cpu() - test_label.cpu()).mean()
        avg_loss += loss

        pred_label = (pred > 0.5) * 1.
        correct += ((pred_label.cpu() == test_label.cpu()) * 1.).sum()
        torch.cuda.empty_cache()

    in_loss = avg_loss / args.test_num
    in_acc = correct / (args.bs*args.test_num) * 100
    print(f"in loss is {in_loss}")
    print(f"in acc is {in_acc}")

    test_data = None
    avg_loss = 0
    correct = 0
    for batch_idx in range(args.test_num):
        img, lbl = generator.sample(args.bs, uncon=True)
        test_data = img.view(args.bs, -1).cpu()
        test_label = lbl

        pred = ntk_pred(train_data, test_data, alpha)
        loss = torch.square(pred.cpu() - test_label.cpu()).mean()
        avg_loss += loss

        pred_label = (pred > 0.5) * 1.
        correct += ((pred_label.cpu() == test_label.cpu()) * 1.).sum()
        torch.cuda.empty_cache()

    out_loss = avg_loss / args.test_num
    out_acc = correct / (args.bs * args.test_num) * 100
    print(f"out loss is {out_loss}")
    print(f"out acc is {out_acc}")

    # intest_loss, intest_acc = test(args, net, generator, False, t_device)
    # outtest_loss, outtest_acc = test(args, net, generator, True, t_device)
    #
    with open(f'NTK_log_{args.man_dim}.txt', 'a') as ff:
        ff.write(f"num sample: {args.num_imgs}\n")
        ff.write(f"{in_loss} {in_acc} {out_loss} {out_acc} {torch.sqrt(2 * train_label.float().dot(alpha)).item()}\n")

import numpy as np
import torch
import math
import argparse
from dataset import *
from tqdm.auto import tqdm

def _ntk(x, y):
    """
    define the two layer infinitely wide ReLU network's neural tangent kernel.
    x (m x D): torch float tensor
    y (n x D): torch float tensor

    result: m x n
    """
    assert x.size(-1) == y.size(-1)

    # Send both tensor to cuda
    x = x.cuda().reshape(x.size(0), -1)
    y = y.cuda().reshape(y.size(0), -1)


    # normalize each tensor, we assume that they both have unit norm
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)

    xTy = x @ y.t()
    xTy = torch.clip(xTy, -1, 1)
    # Important! To avoid nan result because of precision (i.e. acos(1.0000001))

    res = xTy * (math.pi - torch.acos(xTy)) / (2 * math.pi)
    res = res.cpu()

    # Clear the cache to avoid OOM.
    torch.cuda.empty_cache()

    return res

def _RBFK(x, y, gamma=2):
    assert x.size(-1) == y.size(-1)

    # Send both tensor to cuda
    x = x.cuda().reshape(x.size(0), -1)
    y = y.cuda().reshape(y.size(0), -1)

    # normalize each tensor, we assume that they both have unit norm
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)

    d = 2 * x @ y.t() - 2
    d = d * gamma

    res = torch.exp(d).cpu()

    torch.cuda.empty_cache()
    return res


def kernel(X1, X2, k='ntk', chunk_size=1000):
    L1, L2 = X1.size(0), X2.size(0)
    res = torch.zeros(L1, L2)
    pbar = tqdm(total = L1//chunk_size)

    for i in range((L1-1) // chunk_size+1):
        for j in range((L2-1) // chunk_size+1):
            st1, ed1 = i * chunk_size, min((i + 1) * chunk_size, L1)
            st2, ed2 = j * chunk_size, min((j + 1) * chunk_size, L2)
            #print(f"Calculating ({st1},{ed1}) x ({st2},{ed2}

            if k == 'ntk':
                blk_res = _ntk(X1[st1:ed1], X2[st2:ed2])
            elif k == 'rbfk':
                blk_res = _RBFK(X1[st1:ed1], X2[st2:ed2])

            res[st1:ed1, st2:ed2] = blk_res

        pbar.update(1)

    pbar.close()
    return res

def kernel_train(X, y, k='ntk', reg=0.1, chunk_size=20):
    G = kernel(X, X, k=k, chunk_size=chunk_size)
    Gi = torch.inverse(G + torch.eye(X.size(0)) * reg)
    lbl = (y==1) * 1.0
    alpha = Gi @ lbl
    return alpha

def kernel_pred(X_train, X_test, alpha, k='ntk', chunk_size=20):
    K = kernel(X_train, X_test, k=k, chunk_size=chunk_size)
    pred = alpha @ K
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--name', '-n', type=str, default='exp', help='name of the experiment')

    # Dataset hyperparameters
    parser.add_argument('--man_dim', type=int, default=5, help='manifold dimensionality')
    parser.add_argument('--num_imgs', type=int, default=1000, help='manifold dimensionality')
    parser.add_argument('--seed', type=int, default=None, help='seed for generating the dataset')
    parser.add_argument('--dir', type=str, default='./tmp', help='temporary directory for storing dataset')

    # # Training model
    # parser.add_argument('--model', type=str, default='vgg13')
    # parser.add_argument('--test_num', type=int, default=100, help='num of batches for testing')
    #
    # args = parser.parse_args()
    #
    # x0 = torch.tensor([[1., 0.]])
    # theta = torch.arange(-1, 1, 0.01) * math.pi
    # #x = torch.cat((torch.cos(theta).view(-1,1), torch.sin(theta).view(-1,1)), dim=1)
    # x = torch.cat((torch.ones(200, 1), theta.view(-1, 1)), dim=1)
    #
    # z = _ntk(x0, x).squeeze().cpu().numpy()
    #
    # xx = theta.numpy()
    # import matplotlib.pyplot as plt
    # plt.plot(xx, z)
    # plt.grid()
    # plt.savefig("NTK_1d.png")
    # plt.close()

    trainset, testset = MNIST_dataset()
    data = trainset.data
    lbl = trainset.targets

    test_data = testset.data
    test_lbl = testset.targets

    data = data[:1000].float() / 255
    test_data = test_data[:100].float() / 255
    #print(lbl[:1000])
    G = ntk(data, data, chunk_size=1000)
    K = ntk(data, test_data, chunk_size=100)

    Gi = torch.inverse(G + torch.eye(data.size(0))*0.1)
    # Like ridge regression, add regularization to avoid singularity.

    y = (lbl[:1000]==0)*1.0 # if digit is 0 then become 1
    alpha = Gi @ y

    pred = alpha @ K
    np.set_printoptions(suppress=True, precision=3)
    print(f"label {(test_lbl[:100]==0)*1}")
    print(f"pred {pred.numpy()}")

    # print(lbl[:100])
    # print((lbl[:100]==5)*1)

    # print(img_tensor.max())
    # print(testset.data.numpy().shape)
    # print(img_tensor.shape)
    # print(lbl)
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from causalpy.neural_networks import L0Mask, FCNet


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fc = FCNet(3, 20, 10, 1).to(dev)
    x = torch.rand(10000, 3).to(dev)
    y = (50 * x[:, 1]).view(-1, 1)
    dataset = TensorDataset(x, y)
    l0 = L0Mask(fc).to(dev)
    # params = [fc.parameters(), l0.parameters()]
    # params = [fc.parameters()]
    optimizer = torch.optim.Adam(itertools.chain(l0.parameters()), lr=0.001)
    optimizer.zero_grad()
    loss_f = torch.nn.MSELoss()
    losses = []
    # print(*l0.sample_mask(1, deterministic=True), sep="\n")
    # print("FC params")
    # print(*fc.parameters(), sep="\n")
    for i in tqdm(range(1000)):
        for data, target in DataLoader(dataset, 10000):
            loss = 0
            for i in range(1000):
                loss += loss_f(fc(data), target)
            loss /= 1000
            loss += 0.01 * l0.l0_regularization()
            # loss = loss_f(fc(data), target)
            print(loss)
            losses.append(loss.detach())
            loss.backward()
            # for pa in l0.log_alphas:
            #     print(pa.grad)
            optimizer.step()
    # print(*l0.sample_mask(1, deterministic=True), sep="\n")
    # print("FC params")
    # print(*fc.parameters(), sep="\n")
    plt.plot(losses)
    plt.show()

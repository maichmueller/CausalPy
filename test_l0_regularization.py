import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from causalpy.neural_networks import L0Mask, FCNet



if __name__ == "__main__":

    fc = FCNet(3, 20, 10,  1).cuda()
    x = torch.rand(10000, 3).cuda()
    y = (5 * x[:, 1]).view(-1, 1)
    dataset = TensorDataset(x, y)
    l0 = L0Mask(fc.parameters()).cuda()
    params = [fc.parameters(), l0.parameters()]
    # params = [fc.parameters()]
    optimizer = torch.optim.Adam(itertools.chain(*params), lr=0.001)
    optimizer.zero_grad()
    loss_f = torch.nn.MSELoss()
    losses = []
    # print(*l0.sample_mask(1, deterministic=True), sep="\n")
    # print("FC params")
    # print(*fc.parameters(), sep="\n")
    for i in tqdm(range(100)):
        for data, target in DataLoader(dataset, 128):
            loss = l0.compute_loss(
                data,
                target,
                loss_f,
                fc,
                10,
            )
            # loss += l0.l0_regularization()
            # loss = loss_f(fc(data), target)
            losses.append(loss.detach())
            loss.backward()
            for pa in l0.log_alphas:
                print(pa.grad)
            optimizer.step()
    # print(*l0.sample_mask(1, deterministic=True), sep="\n")
    # print("FC params")
    # print(*fc.parameters(), sep="\n")
    plt.plot(losses)
    plt.show()
from causalpy.neural_networks import L0Mask, FCNet
import numpy as np
import torch


if __name__ == "__main__":
    fc = FCNet(1, 4, 1)
    x = (
        torch.tensor(np.random.default_rng().normal(0, 10, size=10000))
        .float()
        .view(-1, 1)
    )
    y = x ** 2
    mask = L0Mask(fc)
    optim = torch.optim.Adam(mask.parameters())
    loss_f = torch.nn.MSELoss()

    for epoch in range(200):

        l = mask.estimate_loss(
            target=y, loss_func=loss_f, data_in=x, mc_sample_size=100
        )
        # print([p for p in fc.parameters()][0])
        l.backward()
        # print([p for p in fc.parameters()][0])

        optim.zero_grad()
        print("LOSS", l.item())

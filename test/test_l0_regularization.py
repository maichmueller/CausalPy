import numpy as np
import torch
from causalpy.neural_networks import L0Mask, FCNet


def test_l0_mask():
    fc = FCNet(3, 4, 5)
    l0 = L0Mask(fc.parameters())
    print(l0.parameters())
    print(l0.l0_regularization())
    print(l0.compute_loss(torch.rand(128, 3), torch.ones(128, 5), torch.nn.CrossEntropyLoss, fc, 10))
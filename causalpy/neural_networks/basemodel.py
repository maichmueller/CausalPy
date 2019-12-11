import torch


class NeuralBaseNet(torch.nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int):
        self.dim_in = dim_in
        self.dim_out = dim_out
        super().__init__()

    def forward(self, tensor_in):
        return

import torch
from abc import abstractmethod


class NeuralBaseNet(torch.nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_out: int):
        self.dim_in = dim_in
        self.dim_out = dim_out
        super().__init__()

    @torch.no_grad()
    def evaluate(self, data):
        """
        Function to evaluate the input in the neural net without accumulating gradients.

        Parameters
        ----------
        data: Tensor,
            the data to evaluate for the network

        Returns
        -------
        Tensor,
            the evaluated data.
        """
        return self(data)

import torch


class FCNet(torch.nn.Module):
    """ Standard feed forward neural network with fully connected layers and ReLUs"""

    def __init__(self, *layer_nodes):
        self.dim_in = layer_nodes[0]
        self.dim_out = layer_nodes[-1]
        self.net = torch.nn.Sequential()

        # Skip Layer -- just a linear mapping
        self.skip_layer = torch.nn.Linear(layer_nodes[0], layer_nodes[-1])

        i = 0
        for l1, l2 in zip(layer_nodes, layer_nodes[1:-1]):
            self.net.add_module("Linear Layer {}".format(i), torch.nn.Linear(l1, l2))
            self.net.add_module("ReLU Layer {}".format(i), torch.nn.ReLU())
            i += 1

        self.net.add_module(
            "Linear Layer {}".format(i),
            torch.nn.Linear(layer_nodes[-2], layer_nodes[-1]),
        )
        super().__init__()

    def forward(self, x):
        return self.net(x)

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

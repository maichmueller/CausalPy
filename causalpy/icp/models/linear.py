import torch


class LinearNet(torch.nn.Module):
    """ Standard feed forward neural network with fully connected layers and ReLUs"""
    def __init__(self, layer_nodes):
        super().__init__()
        self.net = torch.nn.Sequential()

        # Skip Layer -- just a linear mapping
        self.skip_layer = torch.nn.Linear(layer_nodes[0], layer_nodes[-1])

        i = 0
        for l1, l2 in zip(layer_nodes, layer_nodes[1:-1]):
            self.net.add_module("Linear Layer {}".format(i), torch.nn.Linear(l1, l2))
            self.net.add_module("ReLU Layer {}".format(i), torch.nn.ReLU())
            i += 1

        self.net.add_module("Linear Layer {}".format(i), torch.nn.Linear(layer_nodes[-2], layer_nodes[-1]))

        print(self.net)

    def forward(self, x):
        return self.net(x)  # + self.skip_layer(x)
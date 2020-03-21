import torch
from torch import nn, distributions, Tensor


class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1 = nn.Linear(feature_size + class_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 400)
        self.fc4 = nn.Linear(400, feature_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def normalizing_flow(self, x: Tensor, c: Tensor):
        z_mu, z_var = self.encode(x, c)
        return torch.distributions.Normal(z_mu, z_var).rsample(x.size(0))

    def generating_flow(self, z: Tensor, c: Tensor):
        return self.decode(z, c)

    def encode(self, x: Tensor, c: Tensor):
        """
        Q(z|x, c)

        x: (bs, feature_size)
        c: (bs, class_size)
        """
        inputs = torch.cat([x, c], 1)  # (bs, feature_size+class_size)
        h1 = self.relu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparametrize(self, mu: Tensor, logvar: Tensor):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            eps.requires_grad_ = True
            return eps.mul(std) + mu
        else:
            return mu

    def decode(self, z: Tensor, c: Tensor):
        """
        P(x|z, c)

        z: (bs, latent_size)
        c: (bs, class_size)
        """
        inputs = torch.cat([z, c], 1)  # (bs, latent_size+class_size)
        h3 = self.relu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x: Tensor, c: Tensor):
        mu, logvar = self.encode(x.view(-1, 28 * 28), c)
        z = self.reparametrize(mu, logvar)
        return self.decode(z, c), mu, logvar

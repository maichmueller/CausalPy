import torch
import numpy as np
from causalpy.neural_networks import FCNet
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    #
    fcc = FCNet(3, 4)
    fcc.to("cpu")

    def y(x):
        output = torch.empty(x.size(0), 4)
        output[:, 0] = x[:, 0] * 2 + x[:, 1] * (-1)
        output[:, 1] = x[:, 0] * 2 + x[:, 2] * (-10)
        output[:, 2] = x[:, 1] ** 2
        output[:, 3] = x[:, 2] ** 2 + x[:, 2] * (-10)
        return output

    def get_jacobian(net, x):
        dim_in = net.dim_in
        dim_out = net.dim_out
        x = x.squeeze()
        n = x.size(0)
        x = x.repeat(dim_out, 1)
        x.requires_grad_(True)
        z = y(x)
        extraction_matrix = torch.zeros((dim_out * n, dim_out))
        # extraction_matrix = torch.empty(2 * n, 2)

        for j in range(dim_out):
            extraction_matrix[j * n : (j + 1) * n, j] = 1

        # z.backward(torch.eye(2).repeat(n, 1))
        grad_data = torch.autograd.grad(z, x, grad_outputs= extraction_matrix, retain_graph=True, create_graph=True)[0]

        output = torch.zeros((n, dim_in, dim_out))
        # output = torch.zeros((n, 2, 2))
        # for var in range(network.dim_out):
        for entry_input in range(n):
            # output[entry_input] = torch.cat((grad_data[entry_input + var * n] for var in range(network.dim_out)), 0)
            output[entry_input] = torch.cat(
                tuple(grad_data[entry_input + j * n].view(-1, 1) for j in range(dim_out)),
                1,
            )
        return output

    xx, yy, zz = tuple(
        map(
            lambda x: torch.Tensor(x).to("cpu"),
            np.meshgrid(
                np.linspace(0, 2, 3), np.linspace(0, 2, 3), np.linspace(0, 2, 3)
            ),
        )
    )
    # points = np.append(xx.reshape(-1, 1), yy.reshape(-1, 1), axis=1)
    x = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)), 1).to(
        "cpu"
    )
    target = y(x).to("cpu")
    train_ds = TensorDataset(x, target)
    train_loader = DataLoader(train_ds, batch_size=1024)


    # def compute_jacobian(f, x, output_dims):
    #     '''
    #     Normal:
    #         f: input_dims -> output_dims
    #     Jacobian mode:
    #         f: output_dims x input_dims -> output_dims x output_dims
    #     '''
    #     f.to("cpu")
    #     repeat_dims = tuple(output_dims) + (1,) * len(x.shape)
    #     jac_x = x.repeat(*repeat_dims)
    #     jac_x.requires_grad_(True).to("cpu")
    #     jac_y = f(jac_x).to("cpu")
    #
    #     ml = torch.meshgrid([torch.arange(dim) for dim in output_dims])
    #     index = [m.flatten() for m in ml]
    #     gradient = torch.zeros(output_dims + output_dims).to("cpu")
    #     gradient.__setitem__(tuple(index) * 2, 1)
    #
    #     jac_y.backward(gradient)
    #
    #     return jac_x.grad.data

    for val, eval in zip(x.repeat(2, 1), get_jacobian(fcc, x)):
        print(val)
        print(eval)


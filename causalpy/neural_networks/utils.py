import torch


class prohibit_model_grad(object):
    """
    Context manager cutting of the passed model from the graph generation of the contained pytorch computations.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def __enter__(self):
        for p in self.model.parameters():
            p.requires_grad = False
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for p in self.model.parameters():
            p.requires_grad = True
        return self


def get_jacobian(
        network: torch.nn.Module,
        x: torch.Tensor,
        dim_in: int = None,
        dim_out: int = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu",),
        **kwargs
):
    r"""
    Computes the Jacobian matrix for a batch of input data x with respect to the network of the class.

    Notes
    -----
    The output Jacobian for this method is potentially the transpose of the definition one might be familiar with.
    That is, the output Jacobian J will be of the form:

    .. math:: J_{ij} = d_i f_j

    or in matrix form:

    .. math:: | d_1 f_1 \quad d_1 f_2 \quad \dots \quad d_1 f_m |
    .. math:: | d_2 f_1 \quad d_2 f_2 \quad \dots \quad d_2 f_m |
    .. math:: | ... \quad ... \quad ... \quad ... \quad ...  \quad ... \quad ... \quad ... \quad ... |
    .. math:: | d_n f_1 \quad d_n f_2 \quad \dots \quad d_n f_m |


    Parameters
    ----------
    network: torch.nn.Module,
        the network, of which to compute the Jacobian for.
    x: torch.Tensor,
        the data tensor of appropriate shape for then network. Can also be supplied in batches.
    dim_in: int,
        the input dimension for data intended for the neural network.
        If not provided by the user, will be inferred from the network.
    dim_out: int,
        the output dimension of the data returned by neural network.
        If not provided by the user, will be inferred from the network.
    device: (optional) torch.device,
        the device to use for the tensor. If not provided, will fall back to cuda, if available,
        otherwise falls back onto cpu.

    Returns
    -------
    Jacobian: torch.Tensor,
        the jacobian matrix for every entry of data in the batch.
    """
    x = x.squeeze()
    batch_size = x.size(0)

    # ``torch.autograd.grad`` only returns the product J @ v with J = Jacobian, v = gradient vector, so as to
    # allow the user to provide external gradients.
    # Therefore we need to build a matrix of basis vectors v_i = (0, ..., 0, 1, 0, ..., 0), with 1 being on the
    # i-th position, in order to get each column of J (and with this every derivation of each input variable x_i
    # and each output function f_j).
    #
    # The output of the ``autograd.grad`` method apparently returns a tensor of shape (..., ``dim_out), which
    # must mean, that the torch definition of J must be J_ij = d_i f_j, thus we need basis vectors of length
    # ``dim_out`` to get the correct aforementioned matrix product.

    # Unfortunately this also means that we will need to copy the input data ``dim_out`` times, in order to
    # cover all derivatives. This could be memory costly, if the input is big and might is replaced by iterative
    # calls to ``grad`` instead in case the memory allocation fails. However, if the data fits into memory, this
    # should be the faster than an iterative python call to ``autograd.grad``.
    x = x.repeat(dim_out, 1)
    x.requires_grad_(True)

    z = network(x, **kwargs)

    unit_vec_matrix = torch.zeros((dim_out * batch_size, dim_out), device=device)

    for j in range(dim_out):
        unit_vec_matrix[j * batch_size: (j + 1) * batch_size, j] = 1

    grad_data = torch.autograd.grad(
        z,
        x,
        grad_outputs=unit_vec_matrix,
        retain_graph=True,
        create_graph=True,
    )[0]

    # we now have the gradient data, but all the derivation vectors D f_j are spread out over ``dim_out``
    # many repetitive computations. Therefore we need to find all the deriv. vectors belonging to the same
    # function f_j and put them into the jacobian batch tensor of shape (batch_size, dim_in, dim_out).
    # This will provide a Jacobian matrix for every batch data entry.
    jacobian = torch.zeros((batch_size, dim_in, dim_out))
    for batch_entry in range(batch_size):
        jacobian[batch_entry] = torch.cat(
            tuple(
                grad_data[batch_entry + j * batch_size].view(-1, 1)
                for j in range(dim_out)
            ),
            1,
        )
    return jacobian



if __name__ == "__main__":

    from causalpy.neural_networks import FCNet
    from torch.utils.data import DataLoader, TensorDataset

    from tqdm import tqdm

    # x = torch.stack(
    #     [
    #         torch.linspace(-100, 100, 1000),
    #         torch.linspace(-100, 100, 1000),
    #     ],
    #     dim=1
    # )
    def y(x):
        return torch.stack(
            [
                x[:, 0] ** 2 + x[:, 1],
                x[:, 1] - x[:, 0],
                x[:, 1]],
            dim=1
        ).cuda()
    print(get_jacobian(y, torch.tensor([[1, 1]], dtype=torch.float), dim_in=2, dim_out=3))

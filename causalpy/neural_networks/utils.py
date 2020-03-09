from typing import Optional

import torch
from torch import Tensor
import numpy as np
from torch.utils.data import Sampler
from sklearn.model_selection import StratifiedShuffleSplit


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


class StratifiedSampler(Sampler):
    """
    Stratified Sampling
    Provides equal representation of target classes in each batch.
    """

    def __init__(self, data_source, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        super().__init__(data_source=data_source)
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def rbf(
    X: Tensor, Y: Optional[Tensor] = None, sigma: Optional[float] = None
):
    # for computing the general 2-pairwise norm ||x_i - y_j||_2 ^ 2 for each row i and j of the matrices X and Y:
    # the numpy code looks like the following:
    #   XY = X @ Y.transpose()
    #   XX_d = np.ones(XY.shape) * np.diag(X @ X.transpose())[:, np.newaxis]  # row wise mult of diagonal with mat
    #   YY_d = np.ones(XY.shape) * np.diag(Y @ Y.transpose())[np.newaxis, :]  # col wise mult of diagonal with mat
    #   pairwise_norm = XX_d + YY_d - 2 * XY
    if Y is not None:
        if X.dim() == 1:
            X.unsqueeze_(1)
        if Y.dim() == 1:
            Y.unsqueeze_(1)
        # adapted for torch:
        XY = X @ Y.t()
        XY_ones = torch.ones_like(XY)
        XX_d = XY_ones * torch.diagonal(X @ X.t()).unsqueeze(1)
        YY_d = XY_ones * torch.diagonal(Y @ Y.t()).unsqueeze(0)
        pairwise_norm = XX_d + YY_d - 2 * XY
    else:
        if X.dim() == 1:
            X.unsqueeze_(1)
        # one can save some time by not recomputing the same values in some steps
        XX = X @ X.t()
        XX_ones = torch.ones_like(XX)
        XX_diagonal = torch.diagonal(XX)
        XX_row_diag = XX_ones * XX_diagonal.unsqueeze(1)
        XX_col_diag = XX_ones * XX_diagonal.unsqueeze(0)
        pairwise_norm = XX_row_diag + XX_col_diag - 2 * XX

    if sigma is None:
        try:
            mdist = torch.median(pairwise_norm[pairwise_norm != 0])
            sigma = torch.sqrt(mdist)
        except RuntimeError:
            sigma = 1.0

    gaussian_rbf = torch.exp(pairwise_norm * (-0.5 / (sigma ** 2)))
    return gaussian_rbf


def hsic(X, Y, batch_size):
    """ Hilbert Schmidt independence criterion -- kernel based measure for how dependent X and Y are"""

    def centering(
        K: Tensor,
        device: Optional[torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        n = K.shape[0]
        unit = torch.ones(n, n, device=device)
        I = torch.eye(n, device=device)
        Q = I - unit / n
        return torch.mm(torch.mm(Q, K), Q)

    out = (
        torch.sum(
            centering(rbf(X, X), device=X.device)
            * centering(rbf(Y, Y), device=Y.device)
        )
        / batch_size
    )
    return out


def mmd_multiscale(x: Tensor, y: Tensor, normalize_j=False):
    """ MMD with rationale kernel"""
    # Normalize Inputs Jointly
    if normalize_j:
        xy = torch.cat((x, y), 0).detach()
        sd = torch.sqrt(xy.var(0))
        mean = xy.mean(0)
        x = (x - mean) / sd
        y = (y - mean) / sd

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * zz

    XX, YY, XY = (
        torch.zeros_like(xx),
        torch.zeros_like(xx),
        torch.zeros_like(xx),
    )

    for a in [6e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1, 1.2, 1.5, 1.8, 2, 2.5]:
        XX += a ** 2 * (a ** 2 + dxx) ** -1
        YY += a ** 2 * (a ** 2 + dyy) ** -1
        XY += a ** 2 * (a ** 2 + dxy) ** -1

    return torch.mean(XX + YY - 2.0 * XY)


def moments(X: Tensor, Y: Tensor, order=2):
    """ Compares Expectation and Variance between two samples """
    if order == 2:
        a1 = (X.mean() - Y.mean()).abs()
        a2 = (X.var() - Y.var()).abs()

        return (a1 + a2) / 2


def wasserstein(x: Tensor, y: Tensor, normalize: bool = False):
    if normalize:
        x, y = normalize_jointly(x, y)
    sort_sq_diff = (torch.sort(x, dim=0)[0] - torch.sort(y, dim=0)[0]).pow(2)
    std_prod = torch.std(x) * torch.std(y)
    return torch.mean(sort_sq_diff) / std_prod


def normalize_jointly(x: Tensor, y: Tensor):
    xy = torch.cat((x, y), 0).detach()
    sd = torch.sqrt(xy.var(0))
    mean = xy.mean(0)
    x = (x - mean) / sd
    y = (y - mean) / sd
    return x, y


def alpha_moment(data: Tensor, alpha: float):
    assert alpha > 0, f"Alpha needs to be > 0. Provided: alpha={alpha}."
    return torch.pow(torch.pow(data, alpha).mean(dim=0), 1 / alpha)


def get_jacobian(
    network: torch.nn.Module,
    x: Tensor,
    dim_in: int = None,
    dim_out: int = None,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu",
    ),
    **kwargs,
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
        unit_vec_matrix[j * batch_size : (j + 1) * batch_size, j] = 1

    grad_data = torch.autograd.grad(
        z, x, grad_outputs=unit_vec_matrix, retain_graph=True, create_graph=True,
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
            [x[:, 0] ** 2 + x[:, 1], x[:, 1] - x[:, 0], x[:, 1]], dim=1
        ).cuda()

    print(
        get_jacobian(y, torch.tensor([[1, 1]], dtype=torch.float), dim_in=2, dim_out=3)
    )

import numpy as np
from numpy.polynomial import polynomial
from scipy.special import expit
from typing import List, Union, Dict, Type, Callable, TypeVar
from abc import ABC, abstractmethod


class BaseAssignment(ABC):
    """
    An abstract base class for assignments. Assignments are supposed to be functions with a fixed call scheme:
    The first positional argument will be the evaluation of a noise variable N.
     Any further inputs are the evaluations of parent variables, which the associated variable of the assignment
     depends on.

     Any inheriting class will need to implement two functions (aside the __init__ constructor):
    - '__call__': A call method to evaluate the function.
    - 'str': A conversion of the assignment to print out its mathematical behaviour in the form of:
        f(N, x_0, ..., x_k) = ...

    See details of these functions for further explanation.
    """

    @abstractmethod
    def __call__(self, noise: Union[float, np.ndarray], *args, **kwargs):
        """
        The call method to evaluate the assignment (function). The interface only enforces an input of the noise
        variable for each inheriting class, as these are essential parts of an SCM, that can't be omitted.

        It is implicitly necessary in the current framework, that the implemented call method is vectorized, i.e. is
        able to evaluate numpy arrays of consistent shape correctly. This improves sampling from these functions,
        avoiding python loops in favor of numpy C calculations.
        :param noise: float or np.ndarray, the input of the noise variable of a single sample or of an array of samples.
        :param args: any arguments needed by the subclass.
        :param kwargs: any keyword arguments needed by the subclass.
        :return: float or np.ndarray, the evaluated function at all provided spots.
        """
        raise NotImplementedError(
            "Assignment subclasses must provide '__call__' method."
        )

    @abstractmethod
    def __len__(self):
        """
        Method to return the number of variables the assignment takes.
        :return:
        """
        raise NotImplementedError(
            "Assignment subclasses must provide '__len__' method."
        )

    @abstractmethod
    def function_str(self, variable_names=None):
        """
        Method to convert the assignment functor to console printable output of the form: f(N, x_0,...) = ...

        :param variable_names: (optional) List[str], a list of string names for each of the input variables in sequence.
        Each position will be i of the list will be the name of the ith positional variable argument in __call__.
        :return: str, the converted identifier of the function.
        """
        raise NotImplementedError(
            "Assignment subclasses must provide 'function_str' method."
        )

    def __str__(self):
        return self.str()

    def str(self, variable_names=None):
        if variable_names is None:
            variable_names = [f"x_{i}" for i in range(len(self))]
        variable_names = ["N"] + variable_names
        assignment = self.function_str(variable_names)
        prefix = f"f({', '.join(variable_names)}) = "
        return prefix + assignment


AssignmentType = TypeVar("AssignmentType", bound=BaseAssignment)


class LinearAssignment(BaseAssignment):
    def __init__(self, noise_factor, offset=0, *coefficients):
        self.noise_factor = noise_factor
        self.offset = offset
        self.coefficients = (
            np.asarray(coefficients) if len(coefficients) > 0 else np.array([])
        )

    def __call__(self, noise, *args):
        return self.offset + self.noise_factor * noise + self.coefficients @ args

    def __len__(self):
        return 1 + len(self.coefficients)

    def function_str(self, variable_names=None):
        rep = f"{f'{round(self.offset, 2)} + ' if self.offset != 0 else ''}{round(self.noise_factor, 2)} N"
        for i, c in enumerate(self.coefficients):
            if c != 0:
                rep += f" + {round(c, 2)} {variable_names[i + 1]}"
        return rep


class PolynomialAssignment(BaseAssignment):
    def __init__(self, *coefficients_list: List[float]):
        polynomials = []
        if len(coefficients_list) > 0:
            for coefficients in coefficients_list:
                polynomials.append(polynomial.Polynomial(coefficients))
        self.polynomials: np.ndarray = np.asarray(polynomials)

    def __call__(self, *args):
        assert len(args) == len(self.polynomials)
        # args[0] is assumed to be the noise
        return sum((poly(arg) for poly, arg in zip(self.polynomials, args)))

    def __len__(self):
        return len(self.polynomials)

    def function_str(self, variable_names=None):
        assignment = []
        for poly, var in zip(self.polynomials, variable_names):
            coeffs = poly.coef
            this_assign = []
            for deg, c in enumerate(coeffs):
                if c != 0:
                    this_assign.append(
                        f"{round(c, 2)} {var}{f'**{deg}' if deg != 1 else ''}"
                    )
            assignment.append(" + ".join(this_assign))
        return " + ".join(assignment)


class LinkerAssignment(BaseAssignment):
    def __init__(self, linker_func: Callable, assignment_func: AssignmentType):
        self.linker = linker_func
        self.assign_func = assignment_func

    def __call__(self, *args, **kwargs):
        return self.linker(self.assign_func(*args, **kwargs))

    def __len__(self):
        return len(self.assign_func)

    def function_str(self, variable_names=None):
        return f"{self.linker.__name__}({self.assign_func.str(variable_names)})"


def identity(x):
    return x


def sigmoid(x):
    return expit(x / 2)

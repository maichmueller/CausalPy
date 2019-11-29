import numpy as np
from numpy.polynomial import polynomial
from typing import Union, TypeVar
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


Assignment = TypeVar('Assignment', bound=BaseAssignment)

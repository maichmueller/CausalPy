import numpy as np
from abc import ABC, abstractmethod
from typing import Union, TypeVar, Optional, List, Collection, Dict, Hashable


class Assignment(ABC):
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

    def __init__(self):
        self.has_named_args = False
        self._named_arg_position: Dict[Hashable, int] = {}

    @abstractmethod
    def __call__(self, noise: Union[float, np.ndarray], *args, **kwargs):
        """
        The call method to evaluate the assignment (function). The interface only enforces an input of the noise
        variable for each inheriting class, as these are essential parts of an SCM, that can't be omitted.

        It is implicitly necessary in the current framework, that the implemented call method is vectorized, i.e. is
        able to evaluate numpy arrays of consistent shape correctly. This improves sampling from these functions,
        avoiding python loops in favor of numpy C calculations.

        Parameters
        ----------
        noise: float or np.ndarray,
            the input of the noise variable of a single sample or of an array of samples.
        args: any positional arguments needed by the subclass.
        kwargs: any keyword arguments needed by the subclass.

        Returns
        -------
        float or np.ndarray, the evaluated function for all provided inputs.
        """
        raise NotImplementedError(
            "Assignment subclasses must provide '__call__' method."
        )

    @abstractmethod
    def __len__(self):
        """
        Method to return the number of variables the assignment takes.
        """
        raise NotImplementedError(
            "Assignment subclasses must provide '__len__' method."
        )

    @abstractmethod
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        """
        Method to convert the assignment functor to console printable output of the form:
            f(N, x_0,...) = math_rep_of_func_as_string_here...

        Notes
        -----
        The inheriting class only needs to implement the actual assignment part of the string!
        This means, that the prefix 'f(N, X_0, X1,...) =' is added by the base class (with correct number of variables),
        and the child class only needs to provide the string representation of the right hand side of the assignment.

        Parameters
        ----------
        variable_names: (optional) List[str],
            a list of string names for each of the input variables in sequence.
            Each position will be i of the list will be the name of the ith positional variable argument in __call__.

        Returns
        -------
        str, the converted identifier of the function.
        """
        raise NotImplementedError(
            "Assignment subclasses must provide 'function_str' method."
        )

    def __str__(self):
        return self.str()

    def str(self, variable_names: Optional[Collection[str]] = None):
        if variable_names is None:
            if self.has_named_args:
                variable_names = list(
                    name
                    for name, position in sorted(
                        self._named_arg_position.items(),
                        key=lambda x: x[1],
                    )
                )
            else:
                variable_names = [f"X_{i}" for i in range(len(self))]
        variable_names = ["N"] + variable_names
        assignment = self.function_str(variable_names)
        prefix = f"f({', '.join(variable_names)}) = "
        return prefix + assignment

    def set_names_for_args(self, names_collection: Collection[Hashable]):
        """
        Set the positional relation of input arguments with names.

        This method will help assignment function calls later on to provide kwargs with the names of
        causal parents in the graph, without having specifically named these parents in the function
        definition of the assignment already.
        In short: enables dynamic kwarg dispatch on assignments if desired.
        """
        for position, name in enumerate(names_collection):
            self._named_arg_position[name] = position
        self.has_named_args = True

    def parse_call_input(self, *args, **kwargs):
        """
        This method will parse the provided args and kwargs to return only args, that have been
        rearranged according to the order previously set for named args.

        Combined passing of args and kwargs is forbidden and raises a `ValueError`.
        """
        if kwargs:
            if args:
                raise ValueError(
                    "Either only args or only kwargs can be provided to assignment call."
                )

            elif not self.has_named_args:
                raise ValueError(
                    "Kwargs provided, but assignment doesn't have named arguments."
                )

            else:
                args = tuple(
                    val
                    for key, val in sorted(
                        kwargs.items(),
                        key=lambda key_val_pair: self._named_arg_position[key_val_pair[0]],
                    )
                )

        return args

"""
Module containing functions to differentiate functions using JAX.
"""

import functools

try:
    import jax
    from jax import jvp, grad, jit, config
    import jax.numpy as jnp

    config.update("jax_enable_x64", True)
except ImportError:
    jax = None

import numpy as np

from ._backend import Backend
from .. import make_tracing_backend_decorator
from ...tools import bisect_sequence, unpack_singleton_sequence_return_value


class _JaxBackend(Backend):
    def __init__(self):
        super().__init__("JAX")

    @staticmethod
    def is_available():
        return jax is not None

    @Backend._assert_backend_available
    def is_compatible(self, objective, argument):
        return (callable(objective) and isinstance(argument, (list, tuple)) and
                len(argument) > 0)

    @Backend._assert_backend_available
    def compile_function(self, function, arguments):
        jitted_function = jit(function)

        @functools.wraps(jitted_function)
        def wrapper(*args):
            return np.asarray(jitted_function(*map(jnp.asarray, args)))

        return wrapper

    @Backend._assert_backend_available
    def compute_gradient(self, function, arguments):
        num_arguments = len(arguments)
        jitted_gradient = jit(grad(function, argnums=list(range(num_arguments))))

        @functools.wraps(jitted_gradient)
        def wrapper(*args):
            return list(map(np.asarray, jitted_gradient(*args)))

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(wrapper)
        return wrapper

    @Backend._assert_backend_available
    def compute_hessian_vector_product(self, function, arguments):
        num_arguments = len(arguments)
        hvp = functools.partial(jvp, grad(function, argnums=list(range(num_arguments))))
        jitted_hvp = jit(hvp)

        @functools.wraps(jitted_hvp)
        def wrapper(*args):
            points, vectors = bisect_sequence(args)
            return list(map(np.asarray, jitted_hvp(points, vectors)[1]))

        if num_arguments == 1:
            return unpack_singleton_sequence_return_value(wrapper)

        return wrapper


Jax = make_tracing_backend_decorator(_JaxBackend)

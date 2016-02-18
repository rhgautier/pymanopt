"""
Module containing steepest descent (gradient descent) algorithm based on
steepestdescent.m from the manopt MATLAB package.
"""
from __future__ import print_function

import time

from pymanopt.solvers import linesearch as default_linesearchers
from pymanopt.solvers.solver import Solver


class SteepestDescent(Solver):
    def __init__(self, linesearch=None, *args, **kwargs):
        super(SteepestDescent, self).__init__(*args, **kwargs)

        if linesearch is None:
            self._searcher = default_linesearchers.LineSearch()
        else:
            self._searcher = linesearch

    # Function to solve optimisation problem using steepest descent.
    def solve(self, problem, x=None):
        """
        Perform optimization using gradient descent with linesearch.
        This method first computes the gradient (derivative) of obj
        w.r.t. arg, and then optimizes by moving in the direction of
        steepest descent (which is the opposite direction to the gradient).
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .man attribute specifying the manifold to optimize
                over, as well as a cost and enough information to compute
                the gradient of that cost.
            - x=None
                Optional parameter. Starting point on the manifold. If none
                then a starting point will be randomly generated.
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
        """
        man = problem.man
        verbosity = problem.verbosity
        objective = problem.cost
        gradient = problem.grad

        # If no starting point is specified, generate one at random.
        if x is None:
            x = man.rand()

        # Initialize iteration counter and timer
        iter = 0
        time0 = time.time()

        if verbosity >= 2:
            print(" iter\t\t   cost val\t    grad. norm")
        while True:
            # Calculate new cost, grad and gradnorm
            cost = objective(x)
            grad = gradient(x)
            gradnorm = man.norm(x, grad)
            iter = iter + 1

            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e" % (iter, cost, gradnorm))

            # Descent direction is minus the gradient
            desc_dir = -grad

            # Perform line-search
            step_size, x = self._searcher.search(objective, man, x, desc_dir,
                                                 cost, -gradnorm**2)

            stop_reason = self._check_stopping_criterion(
                time0, stepsize=step_size, gradnorm=gradnorm, iter=iter)

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

        return x

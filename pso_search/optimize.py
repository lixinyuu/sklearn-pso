# -*- coding: utf-8 -*-

import logging
import numpy as np
import multiprocessing as mp

from pyswarms.backend.operators import compute_pbest, compute_objective_function
from pyswarms.backend.topology import Star
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from pyswarms.base import SwarmOptimizer
from pyswarms.utils.reporter import Reporter

class PSOoptimizer(SwarmOptimizer):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        bounds=None,
        bh_strategy="periodic",
        velocity_clamp=None,
        vh_strategy="unmodified",
        center=1.00,
        ftol=-np.inf,
        init_pos=None,
    ):
        """
        A custom optimizer modified from pyswarms.single.global_best
        https://github.com/ljvmiranda921/pyswarms/blob/master/pyswarms/single/global_best.py
        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        options : dict with keys :code:`{'c1', 'c2', 'w'}`
            a dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
        bounds : tuple of numpy.ndarray, optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        bh_strategy : str
            a strategy for the handling of out-of-bounds particles.
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity and
            the second entry is the maximum velocity. It sets the limits for
            velocity clamping.
        vh_strategy : str
            a strategy for the handling of the velocity of out-of-bounds particles.
        center : list (default is :code:`None`)
            an array of size :code:`dimensions`
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence. Default is :code:`-np.inf`
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        """
        super(PSOoptimizer, self).__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
            center=center,
            ftol=ftol,
            init_pos=init_pos,
        )

        # Initialize logger
        self.rep = Reporter(logger=logging.getLogger(__name__))
        # Initialize the resettable attributes
        self.reset()
        # Initialize the topology
        self.top = Star()
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.name = __name__
        
        # Populate memory of the handlers    
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position
        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        
        # Set reached requirement
        self.reached_requirement = 0
        
    def get_current_pos(self):
        return self.swarm.position        

    def update(self, iters, current_cost, **kwargs):
        """
        Optimize the swarm for one iteration by providing its cost
        manually.
        Parameters
        ----------
        iters : int
            the current iterations
        current_cost : ndarray
            the current cost which should be provided
        """

        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=logging.DEBUG,
        )
     
        self.swarm.current_cost = current_cost
        self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
        # Set best_cost_yet_found for ftol
        best_cost_yet_found = self.swarm.best_cost
        self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)
        # fmt: on
        # self.rep.hook(best_cost=self.swarm.best_cost)
        # Save to history
        hist = self.ToHistory(
            best_cost=self.swarm.best_cost,
            mean_pbest_cost=np.mean(self.swarm.pbest_cost),
            mean_neighbor_cost=self.swarm.best_cost,
            position=self.swarm.position,
            velocity=self.swarm.velocity,
        )
        self._populate_history(hist)
        
        # Verify stop criteria based on the relative acceptable cost ftol
        relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
        if (
            np.abs(self.swarm.best_cost - best_cost_yet_found) < relative_measure
            ):
            self.reached_requirement = 1
            
        # Perform velocity and position updates
        self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds
            )
        self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh
            )
 
    
    def finalize(self):
        """
        Obtain the final best_cost and the final best_position
        """
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()].copy()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=logging.INFO,
        )
        return (final_best_cost, final_best_pos)
    
    def optimize(self, objective_func, iters,  **kwargs):
        # 
        for _iter in range(iters):
            swarm_pos = self.get_current_pos()
            current_cost = objective_func(swarm_pos, **kwargs)
            self.update(_iter, current_cost)
        return self.finalize()
        

# -*- coding: utf-8 -*-
from __future__ import division
import os
import warnings
import logging

import time
import numpy as np
import random
from collections import defaultdict
from sklearn.base import clone, is_classifier
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._search import BaseSearchCV, check_cv, _check_param_grid
from sklearn.metrics.scorer import check_scoring, _check_multimetric_scoring
from sklearn.utils.validation import _num_samples, indexable
from itertools import product
import logging
from joblib import Parallel, delayed
import pyswarms as ps
import pyswarms.backend as P
from pyswarms.backend.operators import compute_pbest
from pyswarms.backend.topology import Star
from collections import OrderedDict
from .utils import Bound, LogSpace
from .optimize import PSOoptimizer

class PSOSearchCV(BaseSearchCV):
    def __init__(self, estimator, param_grid, scoring=None, cv=None,refit=True, verbose=0, 
                 n_particles=64, iterations =20, pso_c1 = 0.5, pso_c2 = 0.3, 
                 pso_w = 0.9, n_jobs=None,  iid = True,pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):
        super().__init__(estimator=estimator, scoring=scoring,cv=cv,refit=refit,verbose=verbose,
             n_jobs=n_jobs, iid=iid, pre_dispatch=pre_dispatch, error_score=error_score, 
            return_train_score=return_train_score)
        self.n_particles=n_particles
        self.iterations = iterations
        self.param_grid = param_grid
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.pso_w = pso_w
        self._check_params()
    
    def _check_params(self):
        self.fixed_param = OrderedDict()
        self.eval_param = OrderedDict()
        self.eval_logbase = OrderedDict()
        for k, v in self.param_grid.items():
            if isinstance(v, Bound):
                self.eval_param[k] = [v.low, v.high]
                self.eval_logbase[k] = 1
            elif isinstance(v, LogSpace):
                self.eval_param[k] = [v.low, v.high]
                self.eval_logbase[k] = v.logbase
            elif isinstance(v, list):
                assert len(v) == 1, "Attention, PSO search currently does not support discrete param list"
                self.fixed_param[k] = v[0]
            else:
                self.fixed_param[k] = v
        
    def fit(self, X, y=None, groups = None, **fit_params):
        X, y, groups = indexable(X, y, groups)
        self.best_estimator_ = None
        self.best_mem_score_ = float("-inf")
        self.best_mem_params_ = None
        base_estimator = clone(self.estimator)
        n_splits = self.cv.get_n_splits(X, y)
        self.scorers, self.multimetric_ = _check_multimetric_scoring(
                                        self.estimator, scoring=self.scoring)
        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, str) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers) and not callable(self.refit):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key or a "
                                 "callable to refit an estimator with the "
                                 "best parameter setting on the whole "
                                 "data and make the best_* attributes "
                                 "available for that metric. If this is "
                                 "not needed, refit should be set to "
                                 "False explicitly. %r was passed."
                                 % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'
        results = self._fit(X, y, groups)
        
        
        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0 or
                   self.best_index_ >= len(results["params"])):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = results["rank_test_%s"
                                           % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][
                                           self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(clone(base_estimator).set_params(
                **self.best_params_))
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = self.scorers if self.multimetric_ else self.scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

                
    def _fit(self, X, y=None, groups = None, parameter_dict= {}):
        self._cv_results = None  # To indicate to the property the need to update
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        n_samples = _num_samples(X)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
        cv = check_cv(self.cv, y=y, classifier=is_classifier(self.estimator))
        
        swarm_keys = list(self.eval_param.keys())
        min_bound = [self.eval_param[v][0] for v in self.eval_param.keys()]
        max_bound = [self.eval_param[v][1] for v in self.eval_param.keys()]
        logbase = [self.eval_param[v][1] for v in self.eval_param.keys()]
        bounds = (min_bound, max_bound)
  
        options = {'c1': self.pso_c1, 'c2': self.pso_c1, 'w': self.pso_c1} # arbitrarily set
        optimizer = PSOoptimizer(n_particles=self.n_particles, 
                                 dimensions=len(min_bound),
                                 bounds=bounds,  
                                 options=options)  
        
        n_splits = self.cv.get_n_splits(X, y)
        all_candidate_params = []
        all_out = []
        results = {}
        for i in range(self.iterations):
            swarm_pos = optimizer.get_current_pos()
            swarm_pos = self.get_log_position(swarm_pos)
            pos_parameters = [{swarm_keys[idx]:params[idx] for idx in range(len(min_bound))} for params in swarm_pos]
            pos_parameters = [{**n, **self.fixed_param} for n in pos_parameters]
            candidate_params, out = self._evalFunction(X, y, groups, pos_parameters)
            all_candidate_params.extend(candidate_params)
            all_out.extend(out)
            if self.return_train_score:
                _, test_score_dicts, _, _,_ = zip(*out)
            else:
                test_score_dicts, _, _, _ = zip(*out)
            test_scores = [n["score"] for n in test_score_dicts]
            test_scores = -np.array(test_scores).reshape((len(pos_parameters), n_splits)).mean(axis = 1)

            optimizer.update(i, test_scores)
            print('Iteration: {} | avg: {:.4f} | min:{:.4f} | std: {:.4f}'.format(i+1, np.mean(test_scores), np.min(test_scores), np.std(test_scores)))
            if optimizer.reached_requirement == 1:
                print("Accuracy Requirement reached, optimization stop.")
        final_best_cost, final_best_pos = optimizer.finalize()
        print('The best cost found by pso is: {:.4f}'.format(final_best_cost))
        print('The best position found by pso is: {}'.format(self.get_log_position(final_best_pos)))
        
        results = self._format_results(
                    all_candidate_params, self.scorers, n_splits, all_out)
        return results
    
    def get_log_position(self, pos):
        
        logbases = [self.eval_logbase[v]for v in self.eval_param.keys()]
        new_pos = pos.copy()
        if len(pos.shape) == 2:
            for i in range(len(logbases)):
                if logbases[i] != 1:
                    new_pos[:, i] =  logbases[i] ** pos[:, i]
        elif len(pos.shape) == 1:
            for i in range(len(logbases)):
                if logbases[i] != 1:
                    new_pos[i] =  logbases[i] ** pos[i]
        return new_pos
        
        
    def _evalFunction(self, X, y , groups, pos_parameters, **fit_params):
        n_splits = self.cv.get_n_splits(X, y)
        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                            pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(scorer=self.scorers,
                                    fit_params=fit_params,
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True,
                                    return_times=True,
                                    return_parameters=False,
                                    error_score=self.error_score,
                                    verbose=self.verbose)
        with parallel:
            def evaluate_candidates(candidate_params):
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print("Fitting {0} folds for each of {1} candidates,"
                          " totalling {2} fits".format(
                              n_splits, n_candidates, n_candidates * n_splits))

                out = parallel(delayed(_fit_and_score)(clone(base_estimator),
                                                       X, y,
                                                       train=train, test=test,
                                                       parameters=parameters,
                                                       **fit_and_score_kwargs)
                               for parameters, (train, test)
                               in product(candidate_params,
                                          self.cv.split(X, y, groups)))

                if len(out) < 1:
                    raise ValueError('No fits were performed. '
                                     'Was the CV iterator empty? '
                                     'Were there no candidates?')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError('cv.split and cv.get_n_splits returned '
                                     'inconsistent results. Expected {} '
                                     'splits, got {}'
                                     .format(n_splits,
                                             len(out) // n_candidates))
                return candidate_params, out

            candidate_params, out = evaluate_candidates(pos_parameters)
        return candidate_params, out
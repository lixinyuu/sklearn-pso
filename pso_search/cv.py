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
from .utils import Bound, LogSpace, BaseMapFunction
from .optimize import PSOoptimizer

class PSOSearchCV(BaseSearchCV):
    """
    Partical swar search of best hyperparameters, based on Genetic
    Algorithms
    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (`str`) as keys and space of
        parameter settings to try as values.
    n_particles:: int
        The number of parameters set which will be tested in one PSO iteration
    iterations:: int
        The iterations of PSO optimization
    pso_c1:: float
        PSO optimization parameters
    pso_c2:: float
        PSO optimization parameters
    pso_w:: float
        PSO optimization parameters
    scoring : str, callable, list/tuple or dict, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.
        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.
        See :ref:`multimetric_grid_search` for an example.
        If None, the estimator's score method is used.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    pre_dispatch : int, or str, default=n_jobs
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    iid : bool, default=False
        If True, return the average score across folds, weighted by the number
        of samples in each test set. In this case, the data is assumed to be
        identically distributed across the folds, and the loss minimized is
        the total loss per sample, and not the mean loss across the folds.
        .. deprecated:: 0.22
            Parameter ``iid`` is deprecated in 0.22 and will be removed in 0.24
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.
        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.
        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.
        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.
        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.
        See ``scoring`` parameter to know more about multiple metric
        evaluation.
        .. versionchanged:: 0.20
            Support for callable added.
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.
    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.
    """
    def __init__(self, estimator, param_grid, scoring=None, cv=None, refit=True, 
                 verbose=0, n_particles=64, iterations =20, pso_c1 = 0.5, pso_c2 = 0.3, 
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
        """ Only those parameters defined as Bound or LogSpace will be searched in PSO """
        self.fixed_param = OrderedDict()
        self.eval_param = OrderedDict()
        self.eval_func = OrderedDict()
        for k, v in self.param_grid.items():
            if isinstance(v, BaseMapFunction):
                self.eval_param[k] = [v.low, v.high]
                self.eval_func[k] = v.map_func
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
        cv_orig = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)
        self.cv = cv_orig
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
            swarm_pos = self.get_real_position(swarm_pos)
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
        final_best_pos  = self.get_real_position(final_best_pos)
        final_best_pos = {swarm_keys[idx]:final_best_pos[idx] for idx in range(len(min_bound))}
        final_best_pos = {**final_best_pos, **self.fixed_param}
        print('The best cost found by pso is: {:.4f}'.format(final_best_cost))
        print('The best position found by pso is: {}'.format(final_best_pos))
        
        results = self._format_results(
                    all_candidate_params, self.scorers, n_splits, all_out)
        
        return results
    
    def get_real_position(self, pos):
        
        funcs = [self.eval_func[v]for v in self.eval_param.keys()]
        new_pos = pos.copy().astype(object) # astype object to preserve int
        if len(pos.shape) == 2:
            for i in range(len(funcs)):
                new_pos[:, i] =  funcs[i](pos[:, i])
        elif len(pos.shape) == 1:
            for i in range(len(funcs)):
                new_pos[i] =  funcs[i](pos[i])
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

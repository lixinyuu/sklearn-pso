import sys
import os
# Insert system path to use script
sys.path.append(os.path.dirname(os.getcwd()))
from pso_search.optimize import PSOoptimizer
from pso_search import PSOSearchCV, LogSpace, Bound
from pso_search.utils import BaseMapFunction
import sklearn.datasets
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import unittest
import random

def func(x, m=1., z=False):
    if len(x.shape) == 1:
        return m * (np.exp((x[0]**2 + x[1]**2)) + float(z))
    else:     
        return m * (np.exp((x[:, 0]**2 + x[:, 1]**2)) + float(z))

def readme():
    data = sklearn.datasets.load_digits()
    X = data["data"]
    y = data["target"]
    
    paramgrid = {"kernel": ["rbf"],
             "C"     : LogSpace(10., -9., 9.), 
                       # LogSpace means the value x is between [-9, 9] in 
                       # PSO optimization, but the actual C = 10 ** x in SVC
             "gamma" : LogSpace(10., -9., 9.)}

    random.seed(1)

    from pso_search import PSOSearchCV
    cv = PSOSearchCV(estimator=SVC(),
                   param_grid=paramgrid,
                   scoring="accuracy",
                   n_particles=32, 
                   cv=StratifiedKFold(n_splits=4),
                   verbose=0,
                   iterations = 2, 
                   n_jobs=4)
    cv.fit(X, y)
    return cv

class TestPSOSearch(unittest.TestCase):
    def test_logspace(self):
        logspace = LogSpace(1/2, -10, 10)
        self.assertEqual(logspace.map_func(-10), (1/2)**-10)
        
    def test_bound(self):
        bound = Bound(-10., 10.)
        self.assertEqual(bound.map_func(-10.), -10.)
        
    def test_basemapfunc(self):
        basemap = BaseMapFunction(lambda x:1/2*x, -10., 10.)
        self.assertEqual(basemap.map_func(-10.), -5.)
    
    def test_cv(self):
        def try_with_params(**kwargs):
            cv = readme()
            cv_results_ = cv.cv_results_
            print("CV Results:\n{}".format(cv_results_))
            self.assertIsNotNone(cv_results_, msg="cv_results is None.")
            self.assertNotEqual(cv_results_, {}, msg="cv_results is empty.")
            self.assertAlmostEqual(cv.best_score_, 1., delta=.05,
                msg="Did not find the best score. Returned: {}".format(cv.best_score_))

        try_with_params()

    def test_optimize(self):
        """ Simple hill climbing optimization with some twists. """
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        bounds = ([-10, -10], [10, 10])
        pso = PSOoptimizer(n_particles=64, dimensions=2 ,options = options, bounds=bounds)
        best_score, best_params = pso.optimize(func, iters = 100)

        print("Score Results:\n{}".format(best_score))
        self.assertAlmostEqual(best_params[0],0., 3)
        self.assertAlmostEqual(best_params[1],0., 3)
        self.assertAlmostEqual(best_score, 1., 5)

if __name__ == "__main__":
    unittest.main()
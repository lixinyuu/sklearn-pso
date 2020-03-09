# sklearn-pso
Use partical swarm optimization instead of gridsearch in scikit-learn. There is a similar package sklearn-deap which use evolutionary algortihms to find the parameters. However, evolutionary algorithms do not work on continus parameters.

It's implemented using pyswarms library: https://github.com/ljvmiranda921/pyswarms \
Heavily inspired by:\
sklearn-deap: https://github.com/rsteca/sklearn-deap \
scikit-learn https://github.com/scikit-learn/scikit-learn \

Install
-------

**To do

Usage examples
--------------

```python
import sklearn.datasets
import numpy as np
import random

data = sklearn.datasets.load_digits()
X = data["data"]
y = data["target"]

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from pso_search.utils import Bound, LogSpace

paramgrid = {"kernel": ["rbf"],
             "C"     : LogSpace(10., -9., 9.),
             "gamma" : LogSpace(10., -9., 9.)}

random.seed(1)

from pso_search import PSOSearchCV
cv = PSOSearchCV(estimator=SVC(),
                                   param_grid=paramgrid,
                                   scoring="accuracy",
                                   n_particles=32, 
                                   cv=StratifiedKFold(n_splits=4),
                                   verbose=0,
                                   iterations = 5, 
                                   n_jobs=4)
cv.fit(X, y)
```

Output:

        Iteration: 1 | avg: -0.2584 | min:-0.9655 | std: 0.3028
        Iteration: 2 | avg: -0.2988 | min:-0.9716 | std: 0.3433
        Iteration: 3 | avg: -0.4924 | min:-0.9805 | std: 0.3897
        Iteration: 4 | avg: -0.6691 | min:-0.9805 | std: 0.4020
        Iteration: 5 | avg: -0.6961 | min:-0.9805 | std: 0.3804
        The best cost found by pso is: -0.9805
        The best position found by pso is: [1.18500897e+01 5.64562914e-04]
        
To do list
----------

1, Discrete Parameters Search

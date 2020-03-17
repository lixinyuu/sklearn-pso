# sklearn-pso
Use partical swarm optimization instead of gridsearch in scikit-learn. There is a similar package sklearn-deap which use evolutionary algortihms to find the parameters. However, PSO should works better in continuous parameters search.

It's implemented using pyswarms library: https://github.com/ljvmiranda921/pyswarms \
Also heavily inspired by:\
sklearn-deap: https://github.com/rsteca/sklearn-deap \
scikit-learn https://github.com/scikit-learn/scikit-learn 

Install
-------

Download manually

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
from pso_search import Bound, LogSpace

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
                 iterations = 5, 
                 n_jobs=4)
cv.fit(X, y)
```

Output:

    Iteration: 1 | avg: -0.1869 | min:-0.9610 | std: 0.2276
    Iteration: 2 | avg: -0.2778 | min:-0.9777 | std: 0.3127
    Iteration: 3 | avg: -0.5939 | min:-0.9805 | std: 0.4043
    Iteration: 4 | avg: -0.7939 | min:-0.9805 | std: 0.3212
    Iteration: 5 | avg: -0.9031 | min:-0.9805 | std: 0.2037
    The best cost found by pso is: -0.9805
    The best position found by pso is: {'C': 346108044.60263175, 'gamma': 0.0005722350238635339, 'kernel': 'rbf'}
        
To do list
----------

1, Discrete Parameters Search

# `Machine Learning`

- `ML` gives computer the ability to learn without being explicitly programmed.
- `ML` can extract structure | pattern from data. 
- `ML` solve problems that are too difficullt for humans to solve.

### Supervised Learning

- Supervised Learning is the most common form of ML.
- Supervised Learning model learns a relationship between `feature matrix` and `target vector`
- It can make `predictions` for unseen or future data.
- `Regression` : Predicts a `continuous` value.
- `Classification` : Predicts a `categorical` value.

### Format data for Scikit Learn

- Before `fitting` | training a model with `scikit learn`, your data has to be in a numeric format.
- `Scikit Learn` expects feature matrix ( Independent features ) and target vector ( Dependent feature )

### Import important libraries:
```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
```

### Load dataset:
```python
data = load_iris()

# Feature matrix:
df = pd.DataFrame(data.data, columns=data.feature_names) 

# Target vector:
df['species'] = data.target
```

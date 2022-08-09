# `Train Test Split`

- The goal of `ML` is to build a model that performs well on new data.
- Split the dataset into `training set` and `testing set`
- `Train` the model on `training set`
- `Test` the model on `testing set` and evaluate the performance.

### `Import` Libraries:
```python
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

### `Load` Dataset:
```python
data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Feature matrix and target vector:
X = df.loc[:, ['RM', 'LSTAT', 'PTRATION']].values
y = df.loc[:, 'target'].values
```      

### `Train Test Split` : 
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```
- `X_train` = Featue matrix used for training.
- `y_train` = Target vector used for training.
- `X_test` = Feature matrix used for testing.
- `y_test` = Target vector used for evaluation.

### `Linear Regression Model` :
```python
# Create model instance:
model = LinearRegression(fit_intercept=True)

# Train the model with training set:
model.fit(X_train, y_train)
```  

### Measuring model performance:
```python 
# Model performance on train set:
model.score(X_train, y_train)

# Model performance on test set:
model.score(X_test, y_test)
```

- By comparing the model performance on `train set` and `test set` we can identify `overfitting`
- Model performance on `train set` states how well the model is trained.
- Model performance on `test set` helps us to find well the model is performing on new dataset.

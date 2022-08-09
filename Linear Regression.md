# `Linear Regression`

### `Import` important libraries:
```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd

# Imports for Linear Regression:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

### `Load` the dataset:
```python
df.pd.read_csv('Dataset/Linear.csv')
```

### `Linear Regression` is very sensitive to `outliers`, `missing values` :
```python
# Check for missing values:
df.isnull().sum()

# Handle missing values:
df.dropna(axis=0, how='any') or df.fillna('0')
```    

### Arrange data into `feature matrix` and `target vector` :
```python
X = df.loc[:, ['x']].values
y = df.loc[:, ['y']].values
```

### `Linear Regression Model` :
```python
# Create instance of the model:
model = LinearRegression(fit_intercept=True)

# Fit the model:
model.fit(X, y)

# Predict for one observation:
model.predict(X[0].reshape(-1, 1))

# Predict for multiple observations:
model.predict([0:10])
```

### Measuring model performance:
```python
model.score(X, y)

# Observe attribute slope: 
m = model.coef_[0]

# Observe attribute intercept:
c = model.intercept_

# Slope intercept formula y = mx + c:
print(f'y = {m:.2f}x + {c:.2f}')
```

### Plot the best fit regression line:
```python
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,7));

ax.scatter(X, y, color='black');
ax.plot(X, model.predict(X), color='red', linewidth=2);
ax.grid(True, axis='both', zorder=0, linestyle=':', color='k')
ax.tick_params(labelsize = 18)
ax.set_xlabel('Independent', fontsize = 24)
ax.set_ylabel('Dependent', fontsize = 24)
ax.set_title(f'Linear Regression: y = {m:.2f}x + {c:.2f}', fontsize = 20)

fig.tight_layout()
```

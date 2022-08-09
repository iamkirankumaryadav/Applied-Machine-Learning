# `Decision Tree` 🌳

- A very `interpretable` ML algorithm.
- Help in both classification and regression ( predict continuous value )

### `Import` Libraries:
```python
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
```

### `Load` Dataset:
```python
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target
```      

### `Split` the dataset into train and test set:
```python
X_train, X_test, y_train, y_test = train_test_split(df[data.feature_names], df['species'], random_state=42)
```

`Decision Tree` doesn't need standardization.

### `Decision Tree` Model:
```python
# Create instance of model: ( max_depth: number of splits a tree makes )
model = DecisionTreeClassifier(max_depth=2, random_state=42)

# Fit the model:
model.fit(X_train, y_train)
```

### Prediction:
```python
# Predict for one observation:
model.predict(X_test.iloc[0].values.reshapr(1, -1))

# Predict for multiple observations:
model.predict(X_test[0, 10])
```

### Measure Model Performance:
```python
model.score(X_test, y_test)
```

### Find Optimal `max_depth` :
```python
max_depth_range = list(range(1,6))

# List to store average RMSE for each value of max_depth:
accuracy=[]

for depth in max_depth_range:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    accuracy.append(score)
```    

### Find the best max_depth value:
```python
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7));

ax.plot(max_depth_range, accuracy, lw=2, color='k')
ax.set_xlim([1, 5])
ax.set_ylim([0.50, 1.00])
ax.grid(True, axis='both', zorder=0, linestyle=':', color='k')

y_ticks = ax.get_yticks()

y_ticklist = []
for tick in yticks:
    yticklist.append(str(tick).ljust(4, '0')[0:4])
ax.set_yticklabels(y_ticklist)
ax.tick_params(labelsize=16)
ax.set_xticks([1,2,3,4,5])
ax.set_xlabel('Max Depth', fontsize=18)
ax.set_ylabel('Accuracy', fontsize=18)
fig.tight_layout()
```

### Visualize `Decision Tree` using `Matplotlib` 
```python
from sklearn import tree

# Make tree more interpretable:
featurename = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']
classname = ['Setosa', 'Versicolor', 'Virginica']

fig, axes = plt.subplot(nrows=1, ncols=1, figsize=(4,4), dpi=300)
tree.plot_tree(model, feature_names=featurename, class_names=classname, filled=True)

fig.savefig('Image/DecisionTree.png')

# Simply plots the Decision Tree:
tree.plot_tree(model);
```

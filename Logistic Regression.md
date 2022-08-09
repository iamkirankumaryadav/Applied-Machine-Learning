# `Logistic Regression`

- A model used for `classification`
- Model training and predictions are relatively fast.
- No tuning is usually required.
- Performs well on small number of observations.

## `Binary` Classification

### `Import` Libraries:
```python
%matplotlib inline

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
```

### `Load` Dataset:
```python
df = pd.read_csv('Data/Classification.csv')
```

### Split the dataset into train and test set:
```python
X_train, X_test, y_train, y_test = train_test_split(df[['Petal Length (cm)']], df['Species'], random_state=42)
```

- `Logistic Regression` is sensitive towards scale of data.
- Dataset's features should be standardized for better performance ( mean = 0 and variance = 1 )
- `StandardScaler` helps to standardize the dataset features.
- Important: `fit` on train set and `transform` on train and test set ( Never apply fit on test set )
- Important:  No need to apply standardization on target vector ( Only apply on X i.e. feature matrix )

### Standardize Data:
```python
# Create instance for standard scaler:
scaler = StandardScaler()

# Fit on train set:
scaler.fit(X_train)

# Transform on train and test set:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

### `Logistic Regression` Model:
```python
# Create model instance: ( We can tune hyperparameters if required ) 
model = LogisticRegression()

# Fit the model: ( Model is learning the relationship between X features and y labels )
model.fit(X_train, y_train)
```

### Classification:
```python
# Check classification for one observation:
print(f'Prediction: model.predict(X_test[0].reshape(1, -1))[0]')

# Check classification probability:
print(f'Probability: model.predict_proba(X_test[0].reshape(1, -1))')
```

### Measure Model Performance:
```python
model.score(X_test, y_test)
```

### `Metrics` for Classification Model: ( Confusion Matrix )
```python
matrix = metrics.confusion_matrix(y_test, model.predict(X_test))

plt.figure(figsize=(9,9))
sns.heatmap(matrix, annot=True, fmt='.0f', linewidth=0.5, square=True, cmap='Blues');
plt.ylabel('Actual Label', fontsize=16);
plt.xlabel('Predicted Label', fontsize=16);
plt.title(f'Accuracy: {model.score(X_test, y_test)}', size=16);
plt.tick_params(labelsize=16)
```      

## `Multiclass` Classification

- Split the task into multiple binary classification datasets.
- Fit a binary classification model on each

### Better Approach: `OVR` or `OVA`

- A technique that extends binary classification to multi class problems.
- Train one classifier per class, one class is treated as positive class and other is considered as negative class.

1. Binary Classification Problem 1: `digit 0` vs `digits 1, 2, 3`
2. Binary Classification Problem 2: `digit 1` vs `digits 0, 2, 3`
3. Binary Classification Problem 3: `digit 2` vs `digits 0, 1, 3`
4. Binary Classification Problem 4: `digit 3` vs `digits 0, 1, 2`

### `Import` Libraries:
```python
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
```

### `Load` Dataset:
```python
df = pd.read_csv('Data/Multiclass.csv')
```

### `Visualize` Digits:
```python
pixel_colname = df.columns[:-1]

# Create visualization of digits:
plt.figure(figsize=(10, 4))

for index in range(0,5):
    plt.subplot(1, 5, 1+index)
    image_value = df.loc[index, pixel_colname].values # Get all columns except the label column.
    image_label = df.loc[index, 'digit']
    plt.imshow(image_value.reshape(8,8), cmap='gray')
    plt.title(f'Label: {str(image_label)}')
```

### `Split` the dataset into train and test set:
```python
X_train, X_test, y_train, y_test = train_test_split(df[pixel_colname], df['digit'], random_state=42)
```

### `Standardize` the data:
```python
scaler = StandardScaler()

# Fit on train set:
scaler.fit(X_train)

# Transform on train and test set:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

### `Logistic Regression` Model:
```python
model = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=42)

# Fit the model:
model.fit(X_train, y_train)
```

### Measure Model Performance:
```python
model.score(X_test, y_test)

# Check attributes of model: ( You will get 4 different coefficient matrices )
model.intercept_    # You will find 4 intercepts 1 for each class
model.coef_
```

### Classifications
```python
# Check classification probability for one observation:
model.predict_proba(X_test[0:1])

model.predict(X_test[0:1])
```

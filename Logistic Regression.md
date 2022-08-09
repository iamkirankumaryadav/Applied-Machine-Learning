# `Logistic Regression`

- A model used for `classification`
- Model training and predictions are relatively fast.
- No tuning is usually required.
- Performs well on small number of observations.

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

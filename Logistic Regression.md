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


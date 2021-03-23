# Applied Machine Learning

### Model = Algorithm ( Parameters = Values ) + Data

### Essential Steps 

### 1. Explore the Data 
- Data Type 
- Information 
- Rows and Columns 
- Numerical or Categorical 
- Find **Missing** Data 
- Find Relation between Features and Labels 
- Visual Data 

### 2. Clean the Data 
- **Fill** or **Drop** Missing 
- **Encode** Categorical Data 

### 3. Split Datset into Train Set, Validate Set and Test Set 

### 4. Fit | Train an Initial Model and Evaluate 
- Use **K Fold Cross Validation** to get **Better Accuracy** and **Observe** the **Cross Validation Score**

### 5. Tune Hyperparameters by using Grid Search Cross Validation
- Apply **Grid Search Cross Validation** to Find **Optimal Hyperparameters** of a Model which results in the most **Accurate Predictions**
- Find **Best Parameters**

### 6. Evaluate on Validation Set
- Evaluate the Results on Validation Set using the **Best Performing Parameters**
- Create more than one Model to Find **Best Performing Model** for **Test Set** 

### 7. Select and Evaluate the Final Model on Test Set
- Select the Final **Best** Performing Model on Test Set for Evaluation.

Model | Type | Train Speed | Predict Speed | Performance
:--- | :--- | :--- | :--- | :---
Logistic Regression | Classification | Fast | Fast | Medium | Low
Support Vector Machine  |  Classification | Slow | Moderate | Low | Medium
Multi Layer Perceptron | Both | Slow | Moderate | Low
Random Forest | Both | Moderate | Moderate | Low
Boosted Tree | Both | Slow | Fast |Low




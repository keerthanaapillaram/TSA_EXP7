## Devloped by: P Keerthana
## Register Number: 212223240069
## Date: 29-04-2025

# Ex.No: 07-AUTO-REGRESSIVE MODEL

### AIM:
To Implementat an Auto Regressive Model using Python

### ALGORITHM :

### Step 1 :

Import necessary libraries.

### Step 2 :

Read the CSV file into a DataFrame.

### Step 3 :

Perform Augmented Dickey-Fuller test.

### Step 4 :

Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags.

### Step 5 :

Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF).

### Step 6 :

Make predictions using the AR model.Compare the predictions with the test data.

### Step 7 :

Calculate Mean Squared Error (MSE).Plot the test data and predictions.

### PROGRAM

#### Import necessary libraries :

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```

#### Read the CSV file into a DataFrame :

```python
data = pd.read_csv('stock_data.csv',parse_dates=['Date'],index_col='Date')
```

#### Perform Augmented Dickey-Fuller test :

```python
result = adfuller(data['generated_price']) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

#### Split the data into training and testing sets :

```python
x=int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]
```

#### Fit an AutoRegressive (AR) model with 13 lags :

```python
lag_order = 13
model = AutoReg(train_data['generated_price'], lags=lag_order)
model_fit = model.fit()
```

#### Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF) :

```python
plt.figure(figsize=(10, 6))
plot_acf(data['generated_price'], lags=20, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()
plt.figure(figsize=(10, 6))
plot_pacf(data['generated_price'], lags=20, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
```

#### Make predictions using the AR model :

```python
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
```

#### Compare the predictions with the test data :

```python
mse = mean_squared_error(test_data['generated_price'], predictions)
print('Mean Squared Error (MSE):', mse)
```

#### Plot the test data and predictions :

```python

plt.figure(figsize=(12, 6))
plt.plot(test_data['generated_price'], label='Test Data - stock_price')
plt.plot(predictions, label='Predictions - Stock Price',linestyle='--')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid()
plt.show()

```

### OUTPUT:

Dataset:

![image](https://github.com/user-attachments/assets/b8779e77-4d05-4fb9-96fd-9a96ec1594ba)


ADF test result:

![image](https://github.com/user-attachments/assets/97f2067a-8b55-466f-9cd5-7bf6c8abe3c1)

PACF plot:

![image](https://github.com/user-attachments/assets/d008eb1c-9da8-49af-a5ac-02d9d469cb79)


ACF plot:

![image](https://github.com/user-attachments/assets/b72f6e11-b0de-41e8-9c13-1fff913a8a19)


Accuracy:

![image](https://github.com/user-attachments/assets/8f3fd0f7-a8fa-4a3e-9cb7-f87704afb866)


Prediction vs test data:

![image](https://github.com/user-attachments/assets/564fd195-b3ad-4e52-820f-e07e85263f89)


### RESULT:
Thus we have successfully implemented the auto regression function using python.

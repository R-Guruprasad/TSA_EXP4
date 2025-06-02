### DEVELOPED BY: R Guruprasad
### REGISTER NO: 212222240033
### DATE:

# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

# AIM:
To implement ARMA model in python.
# ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

# PROGRAM:
```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('books.csv', nrows =200)

# Use the 'Close' price column
close_prices = data['ratings_count'].dropna()

plt.rcParams['figure.figsize'] = [10, 7.5]

# Simulate ARMA(1,1) Process
ar1 = np.array([1, 0.33])
ma1 = np.array([1, 0.9])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=len(close_prices))
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plot_pacf(ARMA_1)

# Simulate ARMA(2,2) Process
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=len(close_prices) * 10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 200])
plt.show()

# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plot_pacf(ARMA_2)
```
# OUTPUT:
## SIMULATED ARMA(1,1) PROCESS:

![Screenshot 2024-09-13 093954](https://github.com/user-attachments/assets/77aef347-03c0-4d6d-89cb-a1f5f689002b)


## Partial Autocorrelation

![Screenshot 2024-09-13 094010](https://github.com/user-attachments/assets/3495b0fd-92bd-49ef-9ff9-f5f8e854f106)


## Autocorrelation

![Screenshot 2024-09-13 094018](https://github.com/user-attachments/assets/6c23cd84-7a31-47c5-9f75-0321ff1e65e1)


## SIMULATED ARMA(2,2) PROCESS:

![Screenshot 2024-09-13 094026](https://github.com/user-attachments/assets/23a79b09-44a3-4895-b61e-22f53f62b66d)


## Partial Autocorrelation

![Screenshot 2024-09-13 094033](https://github.com/user-attachments/assets/7d6dd74f-b6be-4570-8470-67a6a9f5ce5a)


## Autocorrelation

![Screenshot 2024-09-13 094042](https://github.com/user-attachments/assets/94403f46-8d51-4655-a2d2-fe54ce1444b2)


RESULT:
Thus, a python program is created to fir ARMA Model successfully.

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import math
from arch import arch_model
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter("ignore")

# create data

ticker = "AAPL"

data = yf.download(ticker, period ="5y", interval="1d")

dt = data[['Close']].rename(columns={"Close":"Price"})
dt['return'] = dt['Price'].pct_change()

dt.index = pd.to_datetime(data.index)

# train test split

cutoff_date = dt.index.max() - pd.DateOffset(years=1)


train = dt[dt.index < cutoff_date]
test = dt[dt.index >= cutoff_date]

train['returnLag1'] = train['return'].shift(1)
train['returnLag2'] = train['return'].shift(2)

train = train.dropna(subset=[('return', '')])



# create AR function

class AR:
    def __init__(self, y, p):
        self.p = p
        self.y = y
        self.T = len(y)
        self.y_data, self.X_data = self.create_data()
        self.params_hat = self.run_AR()

    def create_data(self):
        dt = pd.DataFrame({'y':self.y})
        for s in range(1, self.p+1):
            dt[f"lag{s}"] = dt['y'].shift(s)

        dt = dt.dropna()
        y_data = dt['y']
        X_data = dt[[col for col in dt.columns if col != 'y']]
        X_data = sm.add_constant(X_data)

        return y_data, X_data
    
    def run_AR(self):
        model = sm.OLS(self.y_data, self.X_data)
        results = model.fit()

        return results.params
    
    def get_error(self):
        y_hat = self.X_data @ self.params_hat
        dt = pd.DataFrame({'y': self.y, 'y_hat': y_hat})
        dt['eps_hat'] = dt['y'] - dt['y_hat']
        dt = dt.dropna()
        self.error = np.dot(dt['eps_hat'], dt['eps_hat'])/dt.shape[0]
        print(dt)
        return dt

    def plot(self):
        data = self.get_error().reset_index()
        data = data.head(20)
        plt.plot(data.index, data['y'], color='blue')
        plt.plot(data.index, data['y_hat'], color='red')
        plt.title(f"AR{self.p}")
        plt.show()


# create MA function

class MA:
    def __init__(self, y, q):
        self.q = q
        self.y = y
        self.T = len(y)

        self.params0 = [self.y.mean()] + [1e-5]*q + [self.y.std()]
        self.bounds = [(None, None)] + [(-0.999, 0.999)]*self.q + [(1e-6, None)]

        self.params_hat = self.fit()


    def MA_likelihood(self, params):
        y = self.y
        mu = params[0]
        theta = params[1:-1]
        sigma = params[-1]
        T = len(y)

        if sigma <= 0 or sum([(abs(x) >= 1) for x in theta]) > 0:
            return np.inf
        
        eps = np.zeros_like(y)
        for t in range(self.q, len(y)):
            eps[t] = y[t] - mu - np.dot(np.array(list(theta)), np.array(eps[t-self.q:t]))

        ll = -0.5 * (
            np.log(2*np.pi*sigma**2) +
            eps[1:]**2 / sigma**2
        ).sum()

        return -ll 
    
    def fit(self, method="L-BFGS-B"):
        results = minimize(
            self.MA_likelihood,
            self.params0,
            method=method,
            bounds=self.bounds
        )

        return results.x
    
    def get_error(self):
        dt = pd.DataFrame({'y': self.y}).reset_index()
        params_hat = self.params_hat
        mu_hat = params_hat[0]
        theta_hat = params_hat[1:-1]
        sigma_hat = params_hat[-1]

        dt['y_hat'] = np.nan
        dt['eps_hat'] = np.nan

        for idx in dt.index:
            if idx < self.q:
                dt.loc[dt.index[idx], 'y_hat'] = mu_hat
                dt.loc[dt.index[idx], 'eps_hat'] = dt.loc[dt.index[idx], 'y'] -dt.loc[dt.index[idx], 'y_hat']
            else:
                dt.loc[dt.index[idx], 'y_hat'] = mu_hat + np.dot(np.array(theta_hat), np.array(dt.loc[dt.index[idx-self.q: idx], 'eps_hat']))
                dt.loc[dt.index[idx], 'eps_hat'] = dt.loc[dt.index[idx], 'y'] -dt.loc[dt.index[idx], 'y_hat']

        self.error = np.dot(dt['eps_hat'], dt['eps_hat'])/len(self.y)

        return dt
    

    def plot(self):
        data = self.get_error()
        data = data.head(20)
        plt.plot(data.index, data['y'],color='blue')
        plt.plot(data.index, data['y_hat'], color='red')
        plt.title(f"MA{self.q}")
        plt.show()


MAs = list(range(1, 10))
MA_error = []
ARs = list(range(1, 10))
AR_error = []

for q in MAs:
    model = MA(train['return'], q)
    model.get_error()
    MA_error.append(model.error)


for p in ARs:
    model = AR(train['return'], p)
    model.get_error()
    AR_error.append(model.error)

MA_results = pd.DataFrame({'q': MAs, 'error': MA_error})
MA_results = MA_results.sort_values(['error'])

print(MA_results)

AR_results = pd.DataFrame({'p': ARs, 'error': AR_error})
AR_results = AR_results.sort_values(['error'])

print(AR_results)


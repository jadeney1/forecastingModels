import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import math
from arch import arch_model
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

import warnings

#warnings.simplefilter("ignore")

# create data

tickers = ["AAPL", "MSFT", "HOG", "LUV", "PLAY", "GOOGL"]

ticker = "PLAY"

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
        return dt

    def plot(self, **kwargs):
        data = self.get_error().reset_index()
        plt.plot(data.index, data['y'], color='blue')
        plt.plot(data.index, data['y_hat'], color='red')
        plt.title(f"AR{self.p}")
        save = kwargs.get('save')
        if save:
            plt.savefig(f"plots/AR{self.p}")
        else:
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
    

    def plot(self, **kwargs):
        save = kwargs.get('save')
        data = self.get_error()
        data = data.head(20)
        plt.plot(data.index, data['y'],color='blue')
        plt.plot(data.index, data['y_hat'], color='red')
        plt.title(f"MA{self.q}")
        if save:
            plt.savefig(f"plots/MA{self.q}")
        else:
            plt.show()


# create ARMA function

class ARMA:
    def __init__(self, p, q, y):
        self.y = y
        self.p = p
        self.q = q
        self.T = len(self.y)
        mu0 = self.y.mean()
        sigma0 = self.y.std()
        phi0 = [0]*p
        theta0 = [0]*q
        self.params0 = [mu0] + phi0 + theta0 + [sigma0]
        self.bounds = [(None, None)] + [(-0.999, 0.999)]*(p+q) + [(1e-5, None)]

        self.params_hat = self.fit().x


    def ARMA_loglike(self, params):
        y = self.y
        p = self.p
        q = self.q
        mu = params[0]
        phi = params[1:1+p]
        theta = params[1+p:1+p+q]
        sigma = params[-1]
        
        if sigma <= 0:
            return np.inf
        
        T = len(y)
        eps = np.zeros(T)
        
        for t in range(T):
            ar_term = sum(phi[i] * (y[t-1-i]-mu) for i in range(min(p, t)))
            ma_term = sum(theta[j] * eps[t-1-j] for j in range(min(q, t)))
            eps[t] = y[t] - mu - ar_term - ma_term
        
        ll = -0.5 * np.sum(np.log(2*np.pi*sigma**2) + eps**2 / sigma**2)

        return -ll
    
    def fit(self, method="L-BFGS-B"):
        results = minimize(
            self.ARMA_loglike,
            self.params0,
            method=method,
            bounds=self.bounds
        )

        return results
    
    def forecast(self):
        params = self.params_hat
        mu_hat = params[0]
        phi_hat = params[1:1+self.p]
        theta_hat = params[1+self.p:1+self.p+self.q]
        sigma_hat = params[-1]

        eps = np.zeros(self.T)
        y_hat = np.zeros(self.T)
        for t in range(self.T):
            ar_term = sum(phi_hat[i] * (self.y[t-1-i]-mu_hat) for i in range(min(self.p, t)))
            ma_term = sum(theta_hat[j] * eps[t-1-j] for j in range(min(self.q, t)))
            y_hat[t] = mu_hat + ar_term + ma_term
            eps[t] = self.y[t] - y_hat[t]

        return pd.DataFrame({'y':self.y, 'y_hat':y_hat, 'eps':eps})
    

    def plot(self, **kwargs):
        data = self.forecast()
        save = kwargs.get('save')
        plt.plot(range(data.shape[0]), data['y'], color='blue')
        plt.plot(range(data.shape[0]), data['y_hat'], color='red')
        if save:
            plt.savefig(f"plots/ARMA({self.p},{self.q})")
        else:
            plt.show()


    def errors(self):
        data = self.forecast()
        y_hat_var = data['y_hat'].var()
        y_var = data['y'].var()
        print('yhat variance', y_hat_var)
        print('actual variance', y_var)
        average_MSE = (data['eps'] @ data['eps'])/len(data['eps'])
        print('average MSE', average_MSE)
        data['direction'] = data['y_hat']*data['y'] > 0
        direction = data['direction'].mean()
        print('proportion direction correct', direction)



# create ARCH function

class ARCH:
    def __init__(self, y, d, p, q, coefs):
        self.y = list(y)
        self.d = d
        self.p = p
        self.q = q
        self.coefs = list(coefs)
        self.T = len(self.y)

        self.residuals = self.calculate_residuals()

        self.params0 = [1e-1]*(self.d+1)
        self.bounds = [(1e-6, None)]*(self.d+1)

        self.sigma_coefs_hat = self.fit().x
        


    def calculate_residuals(self):
        y = self.y
        eps = np.zeros(self.T)
        y_hat = np.zeros(self.T)

        mu_hat = self.coefs[0]
        phi_hat = self.coefs[1:1+self.p]
        theta_hat = self.coefs[1+self.p:1+self.p+self.q]
        sigma_hat = self.coefs[-1]

        for t in range(self.T):
            ar_term = sum(phi_hat[i] * (self.y[t-1-i]-mu_hat) for i in range(min(self.p, t)))
            ma_term = sum(theta_hat[j] * eps[t-1-j] for j in range(min(self.q, t)))
            y_hat[t] = mu_hat + ar_term + ma_term
            eps[t] = self.y[t] - y_hat[t]

        return pd.DataFrame({'y': y, 'y_hat': y_hat, 'eps': eps})


    def ARCH_loglike(self, params, eps):
        eps = list(eps)
        coefs = params

        if sum([(x < 0) for x in params]) > 0:
            return np.inf

        sigma2 = np.zeros(self.T)
        sigma2[0] = np.var(eps)
        
        for t in range(self.T):
            sigma2[t] = coefs[0] + sum([(a**2)*b for a, b in zip(eps[max(0,t-self.d):t], coefs[1:min(t, self.d)+1])])

        ll = -0.5 * np.sum(np.log(2*np.pi) + np.log(np.array(sigma2)) + np.array(eps)**2 / np.array(sigma2))

        return -ll
    

    def fit(self, method="L-BFGS-B"):
        residuals = list(self.residuals['eps'])
        results = minimize(
            self.ARCH_loglike,
            self.params0,
            args=(residuals, ),
            method=method,
            bounds=self.bounds
        )

        return results
    
    def forecast(self):
        eps = self.residuals['eps'].tolist()
        params = self.sigma_coefs_hat

        sigma2 = np.ones(self.T)

        for t in range(self.T):
            sigma2[t] = params[0] + np.array(eps[max(t-self.d, 0):t])**2 @ np.array(params[1:min(t,self.d)+1])
        
        return sigma2




test_case = (3,3,3)

testARMA = ARMA(p=test_case[0], q=test_case[1], y = train['return'])
testData = testARMA.forecast()
sig2 = testData['eps'].var()
testARMA.errors()
testParams = testARMA.params_hat
testARCH = ARCH(train['return'], d=test_case[2], p=test_case[0], q=test_case[1], coefs=testParams)
testData['sigma2'] = testARCH.forecast()
testData['probPositive'] = testData.apply(lambda row: norm.cdf(row['y_hat']/math.sqrt(row['sigma2'])), axis=1)
testData['probPositiveNaive'] = testData.apply(lambda row: norm.cdf(row['y_hat']/math.sqrt(sig2)), axis=1)
testData['correct'] = testData['y']*(testData['probPositive']-0.5) > 0
testData['correctNaive'] = testData['y']*(testData['probPositive']-0.5) > 0
testData['absFrom50'] = abs(testData['probPositive']-0.5)
testData.to_csv(f"test.csv", index=True)

bounds = np.linspace(0, testData['probPositive'].max(), 1000)

propC = []
profit = []
pInvestments = []
for b in bounds:
    dt2 = testData.copy()
    dt2['buy'] = dt2['probPositive'] > b
    pInvestments.append(dt2['buy'].sum()/len(dt2['y']))
    profit.append(dt2[dt2['buy']]['y'].sum())


print(testData['y'].sum())

plt.title(f"{ticker}")
plt.plot(pInvestments, profit, color='red', label='profit')
plt.plot(pInvestments, [testData['y'].sum()]*len(bounds), color='grey', linestyle='--')
plt.show()


# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:25:56 2024

@author: abc
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

file = 'D:/backup/intdiv/stockdatanew.csv'

df = pd.read_csv(file)

df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d').dt.to_period('M')


df.set_index('Date', inplace=True)
df.index = df.index.to_timestamp()


def adf_statistics(time_series):
    """
    Augmented Dickey-Fuller test for stationarity
    """
    result = adfuller(time_series.values)
    if result[1] < 0.0500:             # result[1] contains the p-value
        return 0                       # returns 0 value if p-value of test is under 5%
    else:
        return 1
    
def adf_tests(df):
    """
    Augmented Dickey-Fuller test applied to every column in DataFrame
    """
    results = df.apply(adf_statistics, axis=0) # Output is a Pandas series
    if sum(results)==0:
        print('Null hypothesis of non-stationarity is rejected for ALL series with p-values < 5%')
    else:
        for i, v in results.items():
            if v == 1:
                print(f'Null hypothesis of non-stationarity of {i} series is NOT rejected')
            else:
                print(f'Null hypothesis of non-stationarity of {i} series is rejected')   
                
                
def transform1(column):
    """
    Log first difference multiplied by 100 transformation (i.e. approximate percent change)
    """
    column = np.log(column).diff() * 100       
    return column

def transform2(column):
    """
    First difference multiplied by 100 transformation
    """
    column = column.diff() * 100       
    return column


def remove_outliers(df):
    """
    Remove outliers (setting their value to missing), defined as observations that are more than 5X 
    the interquartile range from the series mean
    """    
    # Compute the mean and interquartile range
    mean = df.mean()
    iqr = df.quantile([0.25, 0.75]).diff().T.iloc[:, 1]
    
    # Replace entries that are more than 10 times the IQR
    # away from the mean with NaN (denotes a missing entry)
    mask = np.abs(df) > mean + 5 * iqr
    treated = df.copy()
    treated[mask] = np.nan

    return treated

df.head(10)

adf_tests(df)

df.isnull().values.any()

df_station = remove_outliers(df)

endog_dj = df_station.reset_index(drop=True)

#https://www.chadfulton.com/topics/statespace_large_dynamic_factor_models.html

factors = {
 'USA': ['Global', 'NAmerica'],
 'CAN': ['Global', 'NAmerica'],
 'MEX': ['Global', 'NAmerica'],
 'COL': ['Global', 'SAmerica'],
 'PER': ['Global', 'SAmerica'],
 'BRA': ['Global', 'SAmerica'],
 'CHL': ['Global', 'SAmerica'],
 'ARG': ['Global', 'SAmerica'],
 'GBR': ['Global', 'Europe'],
 'IRL': ['Global', 'Europe'],
 'NLD': ['Global', 'Europe'],
 'BEL': ['Global', 'Europe'],
 'FRA': ['Global', 'Europe'],
 'CHE': ['Global', 'Europe'],
 'ESP': ['Global', 'Europe'],
 'PRT': ['Global', 'Europe'],
 'DEU': ['Global', 'Europe'],
 'POL': ['Global', 'Europe'],
 'AUT': ['Global', 'Europe'],
 'CZE': ['Global', 'Europe'],
 'ITA': ['Global', 'Europe'],
 'GRC': ['Global', 'Europe'],
 'ROU': ['Global', 'Europe'],
 'FIN': ['Global', 'Europe'],
 'SWE': ['Global', 'Europe'],
 'NOR': ['Global', 'Europe'],
 'DNK': ['Global', 'Europe'],
 'ZAF': ['Global', 'MENA'],
 'TUR': ['Global', 'MENA'],
 'EGY': ['Global', 'MENA'],
 'ISR': ['Global', 'MENA'],
 'QAT': ['Global', 'MENA'],
 'KAZ': ['Global', 'AP'],
 'CHN': ['Global', 'AP'],
 'KOR': ['Global', 'AP'],
 'JPN': ['Global', 'AP'],
 'IND': ['Global', 'AP'],
 'PAK': ['Global', 'AP'],
 'THA': ['Global', 'AP'],
 'MYS': ['Global', 'AP'],
 'SGP': ['Global', 'AP'],
 'PHL': ['Global', 'AP'],
 'IDN': ['Global', 'AP'],
 'AUS': ['Global', 'AP'],
 'NZL': ['Global', 'AP'],
}

factors2 = {
 'USA': ['Global'],
 'CAN': ['Global'],
 'MEX': ['Global'],
 'COL': ['Global'],
 'PER': ['Global'],
 'BRA': ['Global'],
 'CHL': ['Global'],
 'ARG': ['Global'],
 'GBR': ['Global'],
 'IRL': ['Global'],
 'NLD': ['Global'],
 'BEL': ['Global'],
 'FRA': ['Global'],
 'CHE': ['Global'],
 'ESP': ['Global'],
 'PRT': ['Global'],
 'DEU': ['Global'],
 'POL': ['Global'],
 'AUT': ['Global'],
 'CZE': ['Global'],
 'ITA': ['Global'],
 'GRC': ['Global'],
 'ROU': ['Global'],
 'FIN': ['Global'],
 'SWE': ['Global'],
 'NOR': ['Global'],
 'DNK': ['Global'],
 'ZAF': ['Global'],
 'TUR': ['Global'],
 'EGY': ['Global'],
 'ISR': ['Global'],
 'QAT': ['Global'],
 'KAZ': ['Global'],
 'CHN': ['Global'],
 'KOR': ['Global'],
 'JPN': ['Global'],
 'IND': ['Global'],
 'PAK': ['Global'],
 'THA': ['Global'],
 'MYS': ['Global'],
 'SGP': ['Global'],
 'PHL': ['Global'],
 'IDN': ['Global'],
 'AUS': ['Global'],
 'NZL': ['Global'],
}



factor_multiplicities = {'Global': 2}

model = sm.tsa.DynamicFactorMQ(
    endog_dj,
    factors=factors,
    factor_multiplicities=factor_multiplicities)

model.summary()

results = model.fit(disp=10)

fitted = results.resid
residual = endog_dj - fitted

fitted.to_excel('D:/backup/intdiv/pfitted.xlsx')
residual.to_excel('D:/backup/intdiv/presidual.xlsx')

#######################################################################3



factor_names = ['Global.1', 'Global.2', 'Europe','AP','MENA','SAmerica','NAmerica']
mean = results.factors.smoothed[factor_names]

# Compute 95% confidence intervals
from scipy.stats import norm
std = pd.concat([results.factors.smoothed_cov.loc[name, name]
                 for name in factor_names], axis=1)
crit = norm.ppf(1 - 0.05 / 2)
lower = mean - crit * std
upper = mean + crit * std

with sns.color_palette('deep'):
    fig, ax = plt.subplots(figsize=(14, 3))
    mean.plot(ax=ax)
    
    for name in factor_names:
        ax.fill_between(mean.index, lower[name], upper[name], alpha=0.3)
    
    ax.set(title='Estimated factors: smoothed estimates and 95% confidence intervals')
    fig.tight_layout();
    
    
rsquared = results.get_coefficients_of_determination(method='individual')

top_ten = []
for factor_name in rsquared.columns[:7]:
    top_factor = (rsquared[factor_name].sort_values(ascending=False)
                                       .iloc[:10].round(2).reset_index())
    top_factor.columns = pd.MultiIndex.from_product([
        [f'Top ten variables explained by {factor_name}'],
        ['Variable', r'$R^2$']])
    top_ten.append(top_factor)
pd.concat(top_ten, axis=1)


with sns.color_palette('deep'):
    fig = results.plot_coefficients_of_determination(method='individual', figsize=(14, 9))
    fig.suptitle(r'$R^2$ - regression on individual factors', fontsize=14, fontweight=600)
    fig.tight_layout(rect=[0, 0, 1, 0.95]);
    
#dir(results)

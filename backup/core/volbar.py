import numpy as np
import pandas as pd


def re_bar(x):
    n = 48
    x.loc[:,'cumvol'] = x.loc[:,'vol'].cumsum().copy()
    x.loc[:,'amount'] = (x.loc[:,'vol'] * x.loc[:,'price'] / 10000).copy()
    allVol = x.loc[:,'vol'].sum() + 1
    ruler = np.linspace(0, allVol, n+1)
    x.loc[:,'group'] = x.loc[:,'cumvol'].apply(lambda x:np.sum(x>=ruler)).copy()
    x = x.loc[:, ['vol', 'amount', 'group']].groupby('group').sum()
    x.loc[:,'price'] = (x.loc[:, 'amount'] / x.loc[:,'vol']).copy()
    return(x)

# factor0 - factor0
# **********************************************************
# *********************  based on snapshot  ****************
# **********************************************************







# **********************************************************
# ******************  based on transaction  ****************
# **********************************************************
# Assume that the number of bars per stock is fixed at 48.
# 5min bar    48
def volatility(data):
    def f(x):
        x = re_bar(x)
        x = x[x.price != 0].copy()
        return(np.log(x.price).diff().var())

    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)


# 2
def bar_skew(data):
    def f(x):
        x = re_bar(x)
        x = x[x.price != 0].copy()
        return(np.log(x.price).diff().skew())

    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# 3
def max_gain(data):
    def f(x):
        x = re_bar(x)
        x = x[x.price != 0].copy()
        return np.log(x.loc[:,'price']).diff().max()

    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# 4
def max_loss(data):
    def f(x):
        x = re_bar(x)
        x = x[x.price != 0].copy()
        return np.log(x.loc[:,'price']).diff().min()

    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# 5
def break_high(data):
    def f(x):
        x = re_bar(x)
        x = x[x.price != 0].copy()
        high = np.maximum.accumulate(x.loc[:,'price'])
        
        return(np.sum(high.diff() > 0))

    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# 6
def break_low(data):
    def f(x):
        x = re_bar(x)
        x = x[x.price != 0].copy()
        high = np.minimum.accumulate(x.loc[:,'price'])
        
        return(np.sum(high.diff() < 0))

    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# 7
def normal(data):
    def f(x):
        x = re_bar(x)
        x = x[x.price != 0].copy()
        if x.price.std() == 0:
            return 0
        else:
            return x.price.mean()/x.price.std()
    
    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# 8
def cross(data):
    def f(x):
        x = re_bar(x)
        x = x[x.price != 0].copy()
        if x.empty:
            return 0
        elif x.price.max() == x.price.min():
            return 0
        else:
            return (x.price.iloc[-1] - x.price.iloc[0]) / (x.price.max() - x.price.min())
    
    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# 9
def abs_road(data):
    def f(x):
        x = re_bar(x)
        x = x[x.price != 0].copy()
        x = np.log(x.price).diff()[1:].copy()
        if (np.abs(x)).sum() == 0:
            return 0
        else:
            return x.sum()/(np.abs(x)).sum()
    
    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

'''
# 10
def rank_ret_median(data):
    def f(x):
        x = re_bar(x)
        x = x[x.price != 0].copy()
        if x.shape[0]>1:
            x = np.log(x.price).diff()[1:].copy()
        if x.empty:
            return 0
        else:
            return x.median()
    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return factor
'''
# Done.

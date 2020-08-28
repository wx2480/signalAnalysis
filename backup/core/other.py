import numpy as np
import pandas as pd

# factor0 - factor0

# **********************************************************
# *********************  based on snapshot  ****************
# **********************************************************

# factor0
def corr_mean(data):
    factor = pd.pivot_table(data, index = ['code'], columns=data.index, values = ['last'])
    cols = factor.index
    factor.fillna(method = 'ffill', axis = 1, inplace = True)
    factor.fillna(0, inplace = True)
    factor = np.corrcoef(factor)
    factor = pd.DataFrame(factor, columns = cols, index = cols)
    factor = factor.mean()
    factor.fillna(0, inplace = True)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)
    
# factor1
def corr_var(data):
    factor = pd.pivot_table(data, index = ['code'], columns=data.index, values = ['last'])
    cols = factor.index
    factor.fillna(method = 'ffill', axis = 1, inplace = True)
    factor.fillna(0, inplace = True)
    factor = np.corrcoef(factor)
    factor = pd.DataFrame(factor, columns = cols, index = cols)
    factor = factor.var()
    factor.fillna(0, inplace = True)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)
# factor2
def corr_skew(data):
    factor = pd.pivot_table(data, index = ['code'], columns=data.index, values = ['last'])
    cols = factor.index
    factor.fillna(method = 'ffill', axis = 1, inplace = True)
    factor.fillna(0, inplace = True)
    factor = np.corrcoef(factor)
    factor = pd.DataFrame(factor, columns = cols, index = cols)
    factor = factor.skew()
    factor.fillna(0, inplace = True)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor3
def corr_kurt(data):
    factor = pd.pivot_table(data, index = ['code'], columns=data.index, values = ['last'])
    cols = factor.index
    factor.fillna(method = 'ffill', axis = 1, inplace = True)
    factor.fillna(0, inplace = True)
    factor = np.corrcoef(factor)
    factor = pd.DataFrame(factor, columns = cols, index = cols)
    factor = factor.kurt()
    factor.fillna(0, inplace = True)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor4
def consist_trade(data):
    alpha = 0.9
    f = lambda x:np.sum(np.abs(x['open'] - x['last']) <= alpha * (x['high'] - x['low']))/x.shape[0]
    factor = data.groupby('code').apply(f)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

'''
'''

# **********************************************************
# ******************  based on transaction  ****************
# **********************************************************

# factor5
def enormous_bprice_location(data):
    def f(x):
        x = x[x.price != 0].copy()
        maxp = np.max(x.loc[:,'price'])
        minp = np.min(x.loc[:,'price'])
        ans = x.loc[:,'price'][(x.loc[:,'amount'] > 1000000) & (x.loc[:,'flag2'] == 'B')].mean()
        if maxp == minp:
            return(0)
        if np.isnan(ans):
            return(0)
        else:
            return (ans - minp)/(maxp - minp)

    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    factor = data.groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor6
def enormous_sprice_location(data):
    def f(x):
        x = x[x.price != 0].copy()
        maxp = np.max(x.loc[:,'price'])
        minp = np.min(x.loc[:,'price'])
        ans = x.loc[:,'price'][(x.loc[:,'amount'] > 1000000) & (x.loc[:,'flag2'] == 'S')].mean()
        if maxp == minp:
            return(0)
        if np.isnan(ans):
            return(0)
        else:
            return (ans - minp)/(maxp - minp)

    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    factor = data.groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor7
def vol_mean_price(data):
    def f(x):
        ans = x.loc[:,'amount'].sum()/x.loc[:,'vol'].sum()
        return ans
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    factor = data.groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)
'''
# factor8
def 
'''

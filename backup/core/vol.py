import numpy as np
import pandas as pd

# factor0 - factor7

# **********************************************************
# *********************  based on snapshot  ****************
# **********************************************************

# factor 0
# Day volatility
# freq: daily
# based on snapshot data
def daily_var(data):
    factor = data[['code','last']].groupby('code').var()
    factor = factor/np.power(10, 8)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor 1
# Upward volatility
# freq: daily
# based on snapshot data
def daily_up_var(data):
    f = lambda x: np.sum(np.square(x['last'][x['last']-x['last'].mean() > 0]-x['last'].mean()))/(x.shape[0]-1)
    factor = data[['code','last']].groupby('code').apply(f)
    factor = factor/np.power(10, 8)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor 2
# Downward volatility
# freq: daily
# based on snapshot data
def daily_down_var(data):
    f = lambda x: np.sum(np.square(x['last'][x['last']-x['last'].mean() < 0]-x['last'].mean()))/(x.shape[0]-1)
    factor = data[['code','last']].groupby('code').apply(f)
    factor = factor/np.power(10, 8)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor 3
# Day volatility
# freq: 5min - day
# based on snapshot data
def fivemins_var(data):
    f = lambda x:x['last'].resample('5min').mean().var()
    factor = data[['code','last']].groupby('code').apply(f)
    factor = factor/np.power(10, 8)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor 4
# Upward volatility
# freq: 5min - day
# based on snapshot data
def fivemins_up_var(data):
    f = lambda x: np.sum(np.square(x[x-x.mean() > 0]-x.mean()))/(x.shape[0]-1)
    factor = data[['code', 'last']].groupby('code').apply(lambda x:x[['last']].resample('5min').mean()).groupby('code').apply(f)
    factor = factor/np.power(10, 8)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor 5
# Downward volatility
# freq: 5min - day
# based on snapshot data
def fivemins_down_var(data):
    f = lambda x: np.sum(np.square(x[x-x.mean() < 0]-x.mean()))/(x.shape[0]-1)
    factor = data[['code', 'last']].groupby('code').apply(lambda x:x[['last']].resample('5min').mean()).groupby('code').apply(f)
    factor = factor/np.power(10, 8)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)
    
# factor 6
# avebid's vol
# freq: 5min - day
# based on snapshot data
def daily_var_avebid(data):
    f = lambda x:x['avebid'].resample('5min').mean().var()
    factor = data[['code','avebid']].groupby('code').apply(f)
    factor = factor/np.power(10, 8)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor 7
# aveoff's vol
# freq: 5min - day
# based on snapshot data
def daily_var_aveoff(data):
    f = lambda x:x['aveoff'].resample('5min').mean().var()
    factor = data[['code','aveoff']].groupby('code').apply(f)
    factor = factor/np.power(10, 8)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor8
def daily_skew(data):
    factor = data.loc[:,['code','last']].groupby('code').skew()

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor9
def daily_kurt(data):
    factor = data.loc[:,['code', 'last']].groupby('code').apply(lambda x:x.loc[:,'last'].kurt())

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)


# factor10
def daily_ret_skew(data):
    def f(x):
        ans = (np.log(x.loc[:,'last'][x.loc[:,'last'] > 0]).diff()).skew()
        return ans
    factor = data.loc[:,['code','last']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor11
def daily_ret_kurt(data):
    def f(x):
        ans = (np.log(x.loc[:,'last'][x.loc[:,'last'] > 0]).diff()).kurt()
        return ans
    factor = data.loc[:,['code', 'last']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# *************************************************************************
# factor12
def daily_ret_var(data):
    def f(x):
        factor = (np.log(x.loc[:,'last'][x.loc[:,'last'] != 0]).diff()).var()
        return factor

    factor = data[['code','last']].groupby('code').apply(f)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor13
def daily_ret_up_var(data):
    def f(x):
        ret = np.log(x.loc[:,'last'][x.loc[:,'last'] != 0]).diff()
        factor = np.sum(np.square(ret[ret>0]))/(ret.shape[0] - 1)
        return factor
    
    factor = data[['code','last']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor14
def daily_ret_down_var(data):
    def f(x):
        ret = np.log(x.loc[:,'last'][x.loc[:,'last'] != 0]).diff()
        factor = np.sum(np.square(ret[ret<0]))/(ret.shape[0] - 1)
        return factor
    
    factor = data[['code','last']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor 15
def fivemins_ret_var(data):
    def f(x):
        ret = x.loc[:,'last'].resample('5min').mean()
        ret = np.log(ret[ret != 0]).diff()
        return ret.var()
    
    factor = data[['code','last']].groupby('code').apply(f)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor 16
def fivemins_ret_up_var(data):
    def f(x):
        ret = np.log(x.loc[:,'last'][x.loc[:,'last'] != 0]).diff()
        factor = np.sum(np.square(ret[ret>0]))/(ret.shape[0] - 1)
        return factor
    
    factor = data[['code', 'last']].groupby('code').apply(lambda x:x[['last']].resample('5min').mean()).groupby('code').apply(f)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor 17
def fivemins_ret_down_var(data):
    def f(x):
        ret = np.log(x.loc[:,'last'][x.loc[:,'last'] != 0]).diff()
        factor = np.sum(np.square(ret[ret<0]))/(ret.shape[0] - 1)
        return factor
    
    factor = data[['code', 'last']].groupby('code').apply(lambda x:x[['last']].resample('5min').mean()).groupby('code').apply(f)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)
    
# factor 18
def daily_var_avebid_pct(data):
    def f(x):
        ret = x['avebid'].resample('5min').mean()
        return np.log(ret[ret != 0]).diff().var()
    
    factor = data[['code','avebid']].groupby('code').apply(f)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor 19
def daily_var_aveoff_pct(data):
    def f(x):
        ret = x['aveoff'].resample('5min').mean()
        return np.log(ret[ret != 0]).diff().var()

    factor = data[['code','aveoff']].groupby('code').apply(f)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# **********************************************************
# ******************  based on transaction  ****************
# **********************************************************

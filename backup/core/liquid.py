import numpy as np
import pandas as pd
import statsmodels.api as sm

# factor0 - factor6

# **********************************************************
# *********************  based on snapshot  ****************
# **********************************************************

# factor0
# ask_bid_spread/np.log(price) mean()
def bid_ask_spread(data):
    f = lambda x:((x['ask5'] - x['bid5'])/np.log(x['last']/10000 + 0.0001)).mean()
    factor = data[['code', 'last', 'bid5', 'ask5']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor1
# amount always means money
def market_depth_amount(data):
    f = (
        lambda x:
        (
            x['bid1'] * x['bid1vol'] +
            x['bid2'] * x['bid2vol'] +
            x['bid3'] * x['bid3vol'] +
            x['bid4'] * x['bid4vol'] +
            x['bid5'] * x['bid5vol'] +
            x['ask1'] * x['ask1vol'] +
            x['ask2'] * x['ask2vol'] +
            x['ask3'] * x['ask3vol'] +
            x['ask4'] * x['ask4vol'] +
            x['ask5'] * x['ask5vol']
            ).mean()
        )
    data = data[[
        'code','bid1','bid1vol','ask1','ask1vol','bid2','bid2vol','ask2','ask2vol','bid3',
        'bid3vol','ask3','ask3vol','bid4','bid4vol','ask4','ask4vol','bid5','bid5vol','ask5','ask5vol'
        ]]
    factor = data.groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor2
# 
def market_depth_bidamount(data):
    f = (
        lambda x:
        (
            x['bid1'] * x['bid1vol'] +
            x['bid2'] * x['bid2vol'] +
            x['bid3'] * x['bid3vol'] +
            x['bid4'] * x['bid4vol'] +
            x['bid5'] * x['bid5vol'] 
            ).mean()
        )
    data = data[[
        'code','bid1','bid1vol','ask1','ask1vol','bid2','bid2vol','ask2','ask2vol','bid3',
        'bid3vol','ask3','ask3vol','bid4','bid4vol','ask4','ask4vol','bid5','bid5vol','ask5','ask5vol'
        ]]
    factor = data.groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor3
# amount always means money
def market_depth_askamount(data):
    f = (
        lambda x:
        (
            x['ask1'] * x['ask1vol'] +
            x['ask2'] * x['ask2vol'] +
            x['ask3'] * x['ask3vol'] +
            x['ask4'] * x['ask4vol'] +
            x['ask5'] * x['ask5vol']
            ).mean()
        )
    data = data[[
        'code','bid1','bid1vol','ask1','ask1vol','bid2','bid2vol','ask2','ask2vol','bid3',
        'bid3vol','ask3','ask3vol','bid4','bid4vol','ask4','ask4vol','bid5','bid5vol','ask5','ask5vol'
        ]]
    factor = data.groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# **********************************************************
# ******************  based on transaction  ****************
# **********************************************************

# factor4
# the number of askid
def askid_num(data):
    # data = data[data['flag1'] == '0']
    data.dropna(inplace = True)
    f = lambda x: x['askID'].unique().shape[0]
    factor = data[['code','askID']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor5
def bidid_num(data):
    # data = data[data['flag1'] == '0']
    data.dropna(inplace = True)
    f = lambda x: x['bidID'].unique().shape[0]
    factor = data[['code','bidID']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor6
def askid_bidid_ratio(data):
    # data = data[data['flag1'] == '0']
    data.dropna(inplace = True)
    f = lambda x: (x['askID'].unique().shape[0])/(x['bidID'].unique().shape[0])
    factor = data[['code','askID','bidID']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)
'''
# factor7
def liquid_shock_elasticity(data):
    def f(x):
        if x.price.empty:
            return 0
        if x.loc[:,'amount'].shape[0] < 2:
            return 0
        
        _x = x.loc[:,'amount'][1:].copy()
        _y = np.log(x.loc[:,'price']).diff()[1:].copy()

        model = sm.OLS(_y, _x)
        r = model.fit()
        return r.params.iloc[0]

    data = data[data.loc[:, 'flag1'] == '0']
    data.loc[:,'flag2'] = data.loc[:,'flag2'].apply(lambda x:1 if x == 'B' else -1)
    data.loc[:,'amount'] = (data.loc[:,'price'] * data.loc[:,'vol'] * data.loc[:,'flag2'] / 10000).copy()
    factor = data.groupby('code').apply(f)
 
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)
'''

# factor8
def price2vol_shock(data):
    def f(x):
        x = x[(x.price != 0) & (x.vol != 0)].copy()
        if x.loc[:,'price'].shape[0] < 2:
            return 0
        p = np.log(x.loc[:,'price']).diff()
        q = np.log(x.loc[:,'vol'])
        ans = np.sum(np.abs(p / q))
        return ans

    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor9
def withdraw_ratio(data):
    f = lambda x:np.sum(x.loc[:,'flag1'] == 'C')/x.shape[0]
    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor10
def pareto_buy(data):
    def f(x):
        x = x.loc[:,'vol'][x.loc[:,'flag2'] == 'B']
        ans = x.quantile(0.75)/x.quantile(0.25)
        return ans
    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)


def bar_volatility(data):
    def f(x):
        x = x[(x.loc[:,'price'] != 0) & (x.loc[:,'flag1'] == '0')].copy()
        return np.log(x['price']).diff().var()
    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

def bar_skew_second(data):
    def f(x):
        x = x[(x.loc[:,'price'] != 0) & (x.loc[:,'flag1'] == '0')].copy()
        return np.log(x['price']).diff().skew()
    factor = data.groupby('code').apply(f)
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

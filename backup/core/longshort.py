import numpy as np
import pandas as pd

# factor0 - factor19
# **********************************************************
# *********************  based on snapshot  ****************
# **********************************************************

# factor0
# mostly ls means longshort
def bid_ask_ls(data):
    f = (
        lambda x:
        (
            (
                x['bid1'] * x['bid1vol'] +
                x['bid2'] * x['bid2vol'] +
                x['bid3'] * x['bid3vol'] +
                x['bid4'] * x['bid4vol']
                ).sum() /
            ((
                x['ask1'] * x['ask1vol'] +
                x['ask2'] * x['ask2vol'] +
                x['ask3'] * x['ask3vol'] +
                x['ask4'] * x['ask4vol']
            ).sum() + 1)
            )
        )
    data = data[[
        'code','bid1','bid1vol','ask1','ask1vol','bid2','bid2vol','ask2','ask2vol','bid3',
        'bid3vol','ask3','ask3vol','bid4','bid4vol','ask4','ask4vol','bid5','bid5vol','ask5','ask5vol'
        ]]
    factor = data.groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor1
# one day's maxdrawdown ->short strength
def daily_maxdrawdown(data):
    f = lambda x: np.max(1 - x['last'][x['last'] != 0].dropna()/np.maximum.accumulate(x['last'][x['last'] != 0].dropna()))
    factor = data[['code','last']].groupby('code').apply(lambda x: x[['last']].resample('5min').mean()).groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor2
# one day's meandrawdown ->short strength
def daily_meandrawdown(data):
    f = lambda x: np.mean(1 - x['last'][x['last'] != 0].dropna()/np.maximum.accumulate(x['last'][x['last'] != 0].dropna()))
    factor = data[['code','last']].groupby('code').apply(lambda x: x[['last']].resample('5min').mean()).groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor3
# the max stock gain of the day ->long strength
def daily_maxgain(data):
    f = lambda x: np.max(x['last'][x['last'] != 0].dropna()/np.minimum.accumulate(x['last'][x['last'] != 0].dropna()) - 1)
    factor = data[['code','last']].groupby('code').apply(lambda x: x[['last']].resample('5min').mean()).groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor4
def daily_meangain(data):
    f = lambda x: np.mean(x['last'][x['last'] != 0].dropna()/np.minimum.accumulate(x['last'][x['last'] != 0].dropna()) - 1)
    factor = data[['code','last']].groupby('code').apply(lambda x: x[['last']].resample('5min').mean()).groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor5
# break through previous highpoint
def break_highpoint(data):
    f = lambda x: np.sum((np.maximum.accumulate(x['last'][x['last'] != 0].dropna())).diff() > 0)
    factor = data[['code','last']].groupby('code').apply(lambda x: x[['last']].resample('5min').mean()).groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor6
# low
def break_lowpoint(data):
    f = lambda x: np.sum((np.minimum.accumulate(x['last'][x['last'] != 0].dropna())).diff() < 0)
    factor = data[['code','last']].groupby('code').apply(lambda x: x[['last']].resample('5min').mean()).groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor21
def fivemins_max_gains(data):
    factor = data[['code', 'last']].groupby('code').apply(lambda x:np.max(np.log(x['last'][x['last'] != 0]).diff(100)))

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor22
def price_vol_corr(data):
    f = lambda x:np.corrcoef(x.loc[:,'cumamount'].diff()[1:], x.loc[:,'last'][1:])[1,0]
    factor = data.loc[:,['code', 'last', 'cumamount']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor23
def trend_strength(data):
    def f(x):
        delta = np.exp(-6)
        y = x.loc[:, 'last'][x.loc[:, 'last'] != 0].diff()[1:]
        _y = np.abs(y)
        return np.sum(y) / (np.sum(_y) + delta)
    factor = data.loc[:,['code','last']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)


# factor24
def open_call_auction(data):
    def f(x):
        y = x.loc[:,'cumamount'][x.loc[:,'cumamount'] != 0]
        if y.empty:
            return(0)
        else:
            return(y.iloc[0] / y.iloc[-1])
    
    factor = data.loc[:,['code','cumamount']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor25
def close_call_auction(data):
    def f(x):
        y = x.loc[:,'cumamount'][x.loc[:,'time'] >= 145700000]
        if y.empty:
            return(0)
        
        if y.iloc[-1] == 0:
            return(0)
        return(1 - y.iloc[0]/y.iloc[-1])

    factor = data.loc[:,['code','time','cumamount']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor26
def noon_call_auction(data):
    def f(x):
        y = x.loc[:,'cumamount'][(x.loc[:,'time'] >= 130000000) & (x.loc[:,'time'] <= 130500000)]
        if y.empty:
            return 0
        if y.iloc[-1] == 0:
            return(0)
        return(1 - y.iloc[0]/y.iloc[-1])

    factor = data.loc[:,['code','time','cumamount']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)
    

# **********************************************************
# ******************  based on transaction  ****************
# **********************************************************

# factor7
def small_order(data):
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    factor = data[['code','amount']].groupby('code').apply(lambda x: np.sum(x['amount'] <= 40000))

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)


# factor8
def small_order_amount_ratio(data):
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    factor = data[['code','amount']].groupby('code').apply(lambda x: x['amount'][x['amount'] <= 40000].sum()/x['amount'].sum())

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor9
def medium_order(data):
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    factor = data[['code','amount']].groupby('code').apply(lambda x: np.sum((x['amount'] > 40000) & (x['amount'] <= 200000)))

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)


# factor10
def medium_order_amount_ratio(data):
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    factor = data[['code','amount']].groupby('code').apply(lambda x: x['amount'][(x['amount'] > 40000) & (x['amount'] <= 200000)].sum()/x['amount'].sum())

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor11
def large_order(data):
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    factor = data[['code','amount']].groupby('code').apply(lambda x: np.sum((x['amount'] > 200000) & (x['amount'] <= 1000000)))

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)


# factor12
def large_order_amount_ratio(data):
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    factor = data[['code','amount']].groupby('code').apply(lambda x: x['amount'][(x['amount'] > 200000) & (x['amount'] <= 1000000)].sum()/x['amount'].sum())

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor13
def enormous_order(data):
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    factor = data[['code','amount']].groupby('code').apply(lambda x: np.sum((x['amount'] > 1000000)))

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)


# factor14
def enormous_order_amount_ratio(data):
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    factor = data[['code','amount']].groupby('code').apply(lambda x: x['amount'][(x['amount'] > 1000000)].sum()/x['amount'].sum())

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor15
def netinflows(data):
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    f = lambda x: x['amount'][x['flag2'] == 'B'].sum() - x['amount'][x['flag2'] == 'S'].sum()
    factor = data[['code','flag2','amount']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor16
def small_netinflows(data):
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    
    f = lambda x: x['amount'][(x['flag2'] == 'B') & (x['amount'] <= 40000)].sum() - x['amount'][(x['flag2'] == 'S') & (x['amount'] <= 40000)].sum()
    factor = data[['code','flag2','amount']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor17
def medium_netinflows(data):
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()

    f = (
        lambda x: x['amount'][(x['flag2'] == 'B') & (x['amount'] > 40000) & (x['amount'] <= 200000)].sum() 
        - x['amount'][(x['flag2'] == 'S') & (x['amount'] > 40000) & (x['amount'] <= 200000)].sum()
        )
    factor = data[['code','flag2','amount']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor18
def large_netinflows(data):
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()

    f = (
        lambda x: x['amount'][(x['flag2'] == 'B') & (x['amount'] > 200000) & (x['amount'] <= 1000000)].sum() 
        - x['amount'][(x['flag2'] == 'S') & (x['amount'] > 200000) & (x['amount'] <= 1000000)].sum()
        )
    factor = data[['code','flag2','amount']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor19
def enormous_netinflows(data):
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()

    f = (
        lambda x: x['amount'][(x['flag2'] == 'B') & (x['amount'] > 1000000)].sum() 
        - x['amount'][(x['flag2'] == 'S') & (x['amount'] > 1000000)].sum()
        )
    factor = data[['code','flag2','amount']].groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# factor20
def paying_concentration(data):
    delta = np.exp(-6)
    data = data[data['flag1'] == '0'].copy()
    data.loc[:, 'amount'] = (data.loc[:, 'price'] * data.loc[:, 'vol'] / 10000).copy()
    
    f = lambda x:(x[['bidID', 'amount']][x['flag2'] == 'B'].groupby('bidID').sum()).apply(lambda y:(np.sum(np.square(y)) + delta)/ (np.square(np.sum(y))) + delta)
    factor = data[['code', 'flag2', 'bidID', 'amount']].groupby('code').apply(f)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)


# factor27
def open_auction_netinflows(data):
    def f(x):
        x = x[x.loc[:,'flag1'] == '0'].copy()
        x = x[x.loc[:,'time'] == 92500000].copy()
        x.loc[:,'flag2'] = x.loc[:,'flag2'].copy().apply(lambda x: 1 if x == 'B' else -1)
        ans = np.sum(x.loc[:,'flag2'] * x.loc[:,'price'] * x.loc[:,'vol'] / 10000)
        return ans
    factor = data.groupby('code').apply(f)

    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)


def max_sell_order(data):
    def f(x):
        x = x[x.loc[:,'flag1'] == '0'].copy()
        x.loc[:,'amount'] = (x.loc[:,'price'] * x.loc[:,'vol'] / 10000).copy()
        return x.loc[:,['amount','askID']].groupby('askID').sum()['amount'].max()
    factor = data.groupby('code').apply(f)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

def max_buy_order(data):
    def f(x):
        x = x[x.loc[:,'flag1'] == '0'].copy()
        x.loc[:,'amount'] = (x.loc[:,'price'] * x.loc[:,'vol'] / 10000).copy()
        return x.loc[:,['amount','bidID']].groupby('bidID').sum()['amount'].max()
    factor = data.groupby('code').apply(f)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# amount of orders
def amount_order(data):
    factor = data.groupby('code').apply(lambda x:x.shape[0])
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# amount of buy order
def amount_buy_order(data):
    factor = data.groupby('code').apply(lambda x:x[x.loc[:,'flag2'] == 'B'].shape[0]/x.shape[0])
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

# amount of sell order
def amount_sell_order(data):
    factor = data.groupby('code').apply(lambda x:x[x.loc[:,'flag2'] == 'S'].shape[0]/x.shape[0])
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

def maxvol_price(data):
    def f(x):
        x = x[(x.loc[:,'flag1'] == '0')].copy()
        if x.empty:
            return np.nan
        return x.groupby('price').sum()['vol'].idxmax() / 10000
    factor = data.groupby('code').apply(f)
    
    if isinstance(factor, pd.Series):
        factor = factor.to_frame()
    return(factor)

'''
def 
'''

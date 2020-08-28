from joblib import Parallel,delayed
import numpy as np
import os, sys
import pandas as pd
from tqdm import tqdm

sys.path.append('/data/stock/newSystemData/feature_base/structure/yili/db_utilise/')
import trading_date as ts

# save raw data
def save2raw(save_list):
    def save(params):
        factor, date, factor_name = params
        write_path = r'/home/xiaonan/factor_wxn/rawFactor/'
        write_path = os.path.join(write_path, factor_name)

        if not os.path.exists(write_path):
            os.mkdir(write_path)

        factor.insert(0,'history','')
        date = ts.get_nxt_trading_date(date)
        factor.to_csv(write_path + r'/' + str(date) + r'.csv', header = None)
    
    MAX_WORKERS = 12
    Parallel(n_jobs=MAX_WORKERS)(delayed(save)(i) for i in tqdm(save_list))

    


# save normalization data
def save2nor(save_list):
    def save(params):
        factor, date, factor_name = params
        write_path = r'/home/xiaonan/factor_wxn/factor/'
        write_path = os.path.join(write_path, factor_name)

        if not os.path.exists(write_path):
            os.mkdir(write_path)

        factor.insert(0,'history','')
        date = ts.get_nxt_trading_date(date)
        factor.to_csv(write_path + r'/' + str(date) + r'.csv', header = None)
        
    MAX_WORKERS = 12
    Parallel(n_jobs=MAX_WORKERS)(delayed(save)(i) for i in tqdm(save_list))

# Remove the extreme value, the default is five times the median remove the extreme value
# mostly data means data in func[de_extre]
def de_extre(data, n = 5):
    """
    input:
    data -> Series
    output:
    data -> Series
    """
    def three(x,xmin,xmax):
        if x>xmax:
            return(xmax)
        elif x<xmin:
            return(xmin)
        else:
            return(x)
    
    xmedian = data.median()
    mad = (np.abs(data - xmedian)).median()

    if np.sum(np.square(mad)) == 0:
        return(data)
        
    if isinstance(data, pd.Series):
        data = data.apply(three, xmin = xmedian - n * mad, xmax = xmedian + n * mad)
    elif isinstance(data, pd.DataFrame):
        xmin, xmax = xmedian - n * mad, xmedian + n * mad
        data = (data > xmax) * (xmax - data) + (data < xmin) * (xmin - data) + data
    
    return(data)


# Normalized data, the default is zscore method
def normalization(factor, kind = 'z-score'):
    """
    input:
    data -> Series
    output:
    data -> Series
    """
    if isinstance(factor, pd.Series):
        if kind == 'z-score':
            factor = (factor - factor.mean())/factor.std()
            
        elif kind == 'minmax':
            factor = (factor - factor.min())/(factor.max()- factor.min())
            factor = factor - 0.5

    elif isinstance(factor, pd.DataFrame):
        if kind == 'z-score':
            factor = (factor - factor.mean())/factor.std()
            
        elif kind == 'minmax':
            factor = (factor - factor.min())/(factor.max()- factor.min())
            factor = factor - 0.5

    return(factor)

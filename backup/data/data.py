import datetime
from joblib import Parallel,delayed
import numpy as np
import os, sys
import pandas as pd
from tqdm import tqdm

'''
sys.path.append('/home/sharedFold/zhaorui/')
import daily_analysis as da
'''

MAX_WORKERS = 12

# get one csv
def load_one(path,_dtype = {}):
    if _dtype:
        df = pd.read_csv(path,delimiter='\t',header=None,dtype=_dtype)
    else:
        df = pd.read_csv(path,delimiter='\t',header=None)
    return df

# get all csv under the path
def load_all(tick_dir,date,_dtype = {}):
    dir = os.path.join(tick_dir,str(date))
    paths = [os.path.join(dir,i) for i in os.listdir(dir) if i != '.directory' and i[-4:] == r'.csv'] # 剔除可能有的隐藏文件
    res = Parallel(n_jobs=MAX_WORKERS)(delayed(load_one)(path, _dtype) for path in tqdm(paths))
    return pd.concat(res)

# get all snapshot
# data.index -> datetime
# data.code -> int no 'SH'...
def get_snap(date, univ_name = None):
    tick_dir = r'/data/stock/newSystemData/rawdata/tickdata/tick_3s_csv/htsc_tick_3s_1.3.1/'
    with open(r'/data/stock/newSystemData/rawdata/tickdata/tick_3s_csv/' + 'tick_format') as f:
        cols = f.read().split('\n')
    
    res = load_all(tick_dir, date)
    res.columns = cols
    res['code'] = res['code'].apply(lambda x:str(x)[:6])

    f = lambda x: str(x) if len(str(x)) == 9 else '0' + str(x)
    res.index = (res.date.apply(str) + res.time.apply(f)).apply(lambda x: datetime.datetime.strptime(x[:-3],'%Y%m%d%H%M%S'))

    return res

def get_transaction(date):
    tick_dir = r'/data/stock/newSystemData/rawdata/tickdata/transaction_csv/'
    with open(tick_dir + 'transaction_format') as f:
        cols = f.read().split('\n')
    
    res = load_all(tick_dir, date, _dtype={4:str, 5:str})
    res.columns = cols

    f = lambda x: str(x) if len(str(x)) == 9 else '0' + str(x)
    res.index = (res.date.apply(str) + res.time.apply(f)).apply(lambda x: datetime.datetime.strptime(x[:-3],'%Y%m%d%H%M%S'))

    return res

# get one csv and produce the datetime index
def _load_one(path,_dtype = {}):
    if _dtype:
        df = pd.read_csv(path,delimiter='\t',header=None,dtype=_dtype)
    else:
        df = pd.read_csv(path,delimiter='\t',header=None)
    
    f = lambda x: str(x) if len(str(x)) == 9 else '0' + str(x)
    df.index = (df.date.apply(str) + df.time.apply(f)).apply(lambda x: datetime.datetime.strptime(x[:-3],'%Y%m%d%H%M%S'))
    return df

# get all csv under the path
# when the function loads one csv, it uses _load_one instead of load_one 
def _load_all(tick_dir,date,_dtype = {}):
    dir = os.path.join(tick_dir,str(date))
    paths = [os.path.join(dir,i) for i in os.listdir(dir) if i != '.directory' and i[-4:] == r'.csv'] # 剔除可能有的隐藏文件
    res = Parallel(n_jobs=MAX_WORKERS)(delayed(_load_one)(path, _dtype) for path in tqdm(paths))
    return pd.concat(res)

# get all snapshot and it uses _load_all
# data.index -> datetime
# data.code -> int no 'SH'...
def _get_snap(date, univ_name = None):
    def _load_one(path,cols,_dtype = {}):
        if _dtype:
            df = pd.read_csv(path,delimiter='\t',header=None,dtype=_dtype)
        else:
            df = pd.read_csv(path,delimiter='\t',header=None)
        
        df.columns = cols
        f = lambda x: str(x) if len(str(x)) == 9 else '0' * (9 - len(str(x))) + str(x)
        try:
            df.index = (df.date.apply(str) + df.time.apply(f)).apply(lambda x: datetime.datetime.strptime(x[:-3],'%Y%m%d%H%M%S'))
        except:
            print(path)
        df['code'] = df['code'].apply(lambda x:str(x)[:6])
        return df

    tick_dir = r'/data/stock/newSystemData/rawdata/tickdata/tick_3s_csv/htsc_tick_3s_1.3.1/'
    with open(r'/data/stock/newSystemData/rawdata/tickdata/tick_3s_csv/' + 'tick_format') as f:
        cols = f.read().split('\n')
    
    dir = os.path.join(tick_dir,str(date))
    paths = [os.path.join(dir,i) for i in os.listdir(dir) if i != '.directory' and i[-4:] == r'.csv'] # 剔除可能有的隐藏文件
    res = Parallel(n_jobs=MAX_WORKERS)(delayed(_load_one)(path, cols) for path in tqdm(paths))
    return pd.concat(res)

# similarity, _load_all
def _get_transaction(date):
    def _load_one(path, cols, _dtype = None):
        if _dtype:
            df = pd.read_csv(path,delimiter='\t',header=None,dtype=_dtype)
        else:
            df = pd.read_csv(path,delimiter='\t',header=None)
        
        df.columns = cols
        f = lambda x: str(x) if len(str(x)) == 9 else '0' + str(x)
        df.index = (df.date.apply(str) + df.time.apply(f)).apply(lambda x: datetime.datetime.strptime(x[:-3],'%Y%m%d%H%M%S'))
        
        return df
    
    tick_dir = r'/data/stock/newSystemData/rawdata/tickdata/transaction_csv/'
    with open(tick_dir + 'transaction_format') as f:
        cols = f.read().split('\n')
    
    dir = os.path.join(tick_dir,str(date))
    paths = [os.path.join(dir,i) for i in os.listdir(dir) if i != '.directory' and i[-4:] == r'.csv'] # 剔除可能有的隐藏文件
    res = Parallel(n_jobs=MAX_WORKERS)(delayed(_load_one)(path, cols, _dtype={4:str, 5:str}) for path in tqdm(paths))
    return pd.concat(res)

if __name__ == "__main__":
    pass
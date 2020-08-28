# common commands
# db datebase
import datetime
import os, sys
sys.path.append(r'/data/stock/newSystemData/feature_base/structure')
from DB_Control import db
import logging

from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn import tree
from sklearn.svm import SVR
import statsmodels.api as sm
import time
import numpy as np
import pandas as pd

# analysis tools writed by cython
sys.path.append('/home/sharedFold/zhaorui/')
import daily_analysis as da
# obj = da.SignalAnalysis(20200101,20200721,'/home/sharedFold/zhaorui/ret.parquet','close','/home/xiaonan/factor_wxn/factor/bid_ask_spread/')


sys.path.append('/data/stock/newSystemData/feature_base/structure/yili/db_utilise/')
import trading_date as ts
# ts.get_nxt_trading_date(20200721)

sys.path.append('/data/stock/newSystemData/feature_base/structure/yuanyang/alphasystem/dataMgr/')
import dataMgr as dM

def top2000stkpool(start, end):
    pool = dM.load_universe(start,end,univ_name = 'TOP2000')
    return pool


def abcd():
    # make sure the folder log exists.
    if os.path.exists(r'./Log/'):
        pass
    else:
        os.mkdir(r'./Log/')
        
    logger = logging.getLogger('xiaonan')
    logger.setLevel(level = logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler = logging.FileHandler('./Log/log.txt')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    return logger


def allfactortime(logger):
    a = logging.getLogger('xiaonan.allfactorname')
    a.info('start')
    files = os.listdir('./rawFactor/')
    for i in files:
        path = os.path.join('./rawFactor/', i)
        ans = os.listdir(path)

        ans = list(map(lambda x:int(x[:8]), ans))
        
        print('+', '-' * 54, '+')
        print('|',i, ' '* (30-len(i)),'|', min(ans), '|', max(ans),'|')
    a.info('closed')

import tools
def rawFactor_move():
    def move_to(name, date):
        read_path = r'/home/xiaonan/factor_wxn/rawFactor/'
        write_path = r'/home/xiaonan/factor_wxn/factor/'

        path = read_path + name + r'/' + date
        data = pd.read_csv(path, index_col=0, header=None)
        # if 'enormous' not in path:
        data = tools.tools.de_extre(data)
        data = tools.tools.normalization(data)
        if os.path.exists(write_path + r'/' + name):
            pass
        else:
            os.mkdir(write_path + r'/' + name)

        data.to_csv(write_path + name + r'/' + date, header=None)
    
    read_path = r'/home/xiaonan/factor_wxn/rawFactor/'
    task_list = []
    for name in os.listdir(read_path):
        if name == '.directory':
            continue
        for date in os.listdir(os.path.join(read_path, name)):
            if date == 'directory':
                continue
            task_list.append([name, date])

    Parallel(10)(delayed(move_to)(name, date) for name,date in tqdm(task_list))
    print('\n\n Finsihed. \n\n')

def lagFactor():
    read_rawpath = r'/home/xiaonan/factor_wxn/rawFactor/'
    read_path = r'/home/xiaonan/factor_wxn/factor/'

    write_rawpath = r'/home/xiaonan/factor_wxn/lag/rawFactor/'
    write_path = r'/home/xiaonan/factor_wxn/lag/factor/'

    for folder in tqdm(os.listdir(read_rawpath)):
        path = os.path.join(read_rawpath, folder)
        for filename in os.listdir(path):
            data = pd.read_csv(os.path.join(path, filename), index_col = 0, header = None)
            newName = ts.get_nxt_trading_date(int(filename[:8]))
            if os.path.exists(os.path.join(write_rawpath,folder)):
                pass
            else:
                os.mkdir(os.path.join(write_rawpath,folder))
            data.to_csv(os.path.join(write_rawpath,folder) + r'/' + str(newName) + r'.csv', header = None)
    
    for folder in tqdm(os.listdir(read_path)):
        path = os.path.join(read_path, folder)
        for filename in os.listdir(path):
            data = pd.read_csv(os.path.join(path, filename), index_col = 0, header = None)
            newName = ts.get_nxt_trading_date(int(filename[:8]))
            if os.path.exists(os.path.join(write_path,folder)):
                pass
            else:
                os.mkdir(os.path.join(write_path,folder))
            data.to_csv(os.path.join(write_path,folder) + r'/' + str(newName) + r'.csv', header = None)


def corr_heatmap(date, kind = True):
    ret = []

    read_path = r'/home/xiaonan/factor_wxn/factor/'
    factor_list = os.listdir(read_path)
    for factor_name in factor_list:
        if factor_name == '.directory':
            continue
        path = os.path.join(read_path, factor_name)
        path = path + r'/' + str(date) + r'.csv'

        if os.path.exists(path):
            data = pd.read_csv(path, index_col = 0, header = None)
            data.columns = ['nouse',factor_name]
            data.drop(columns = ['nouse'], inplace = True)

            ret.append(data)

    ret = pd.concat(ret, axis = 1)
    ret = ret.fillna(ret.mean())

    heatmap_data = np.corrcoef(ret, rowvar = 0)
    heatmap_data = pd.DataFrame(heatmap_data, columns = ret.columns, index = ret.columns)
    heatmap_data = heatmap_data.applymap(lambda x:round(x,2))
    x_location, y_location = np.where(np.abs(heatmap_data) > 0.75)

    ans = {}
    count = 0
    for i in range(len(x_location)):
        if x_location[i] < y_location[i]:
            t1 = heatmap_data.index[x_location[i]] in ans.keys()
            t2 = heatmap_data.columns[y_location[i]] in ans.keys()
            if t1:
                if t2:
                    pass
                else:
                    ans[heatmap_data.columns[y_location[i]]] = ans[heatmap_data.index[x_location[i]]]
            else:
                if t2:
                    ans[heatmap_data.index[x_location[i]]] = ans[heatmap_data.columns[y_location[i]]]
                else:
                    ans[heatmap_data.index[x_location[i]]], ans[heatmap_data.columns[y_location[i]]] = count, count
                    count+=1
    ans = pd.DataFrame(ans,index = ['group']).T.sort_values('group')
    for i, j in enumerate(ans.groupby('group').apply(lambda x:list(x.index))):
        print(i,' ',str(j)[1:-1])
    print('\n', ans.shape[0], count, '\nthe num of low corr factors', heatmap_data.shape[0] - ans.shape[0] + count, '\n')
    
    if kind:
        heatmap_data = np.abs(heatmap_data)
    
    fig, ax = plt.subplots(figsize=(14,14))
    sns.heatmap(
        heatmap_data,
        xticklabels=heatmap_data.columns,
        yticklabels=heatmap_data.index,
        ax = ax,
        # annot = True
        )
    plt.show(fig)

def test_IC():
    ret_source = '/home/sharedFold/zhaorui/ret.parquet'

    path = r'/home/xiaonan/factor_wxn/factor/'

    for i in os.listdir(path):
        _path = os.path.join(path, i)
        ans = []
        for k in os.listdir(_path):
            if k == '.directory':
                continue
            ans.append(int(k.split('.')[0]))
        start = min(ans)
        end = max(ans)
        try:
            print('factor {} start:'.format(i))
            obj = da.SignalAnalysis(start,end,ret_source,'close',_path)
            obj.show('x0')
            print('factor {} end.\n\n'.format(i))
        except:
            print('***********')
            print('**  ***  **')
            print('***********')
            print('***     ***')
            print('***********\n\n')


def factorTimeInterval():
    ans = {}
    path = r'/home/xiaonan/factor_wxn/factor/'
    for i in os.listdir(path):
        if i == '.directory':
            continue
        ans[i] = {}
        _ = os.path.join(path, i)
        datelist = os.listdir(_)
        datelist = [int(i[:8]) for i in datelist if i != '.directory']
        start = np.min(datelist)
        end = np.max(datelist)
        t = ts.get_trading_date(start, end)
        ans[i]['start'] = start
        ans[i]['end'] = end
        print('****************************************')
        print('factor_name: ', i)
        if len(t) != len(datelist):
            print('{}: the trading date is missing {}!!!'.format(i, len(t) - len(datelist)))
        print('start  time: ', start)
        print('end    time: ', end)
    print('****************************************')
    
    pd.DataFrame(ans).T.to_csv('factorTimeInterval.csv')


# calculate the IC of 5 days of return
def plot_IC(start, end, name, retinterval = 'y_close_5',plot = False):
    plt.rcParams['figure.figsize'] = (18,6)
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    timeline = ts.get_trading_date(start, end)
    ret = pd.read_parquet('/home/sharedFold/zhaorui/ret.parquet')
    # ret = ret[ret['dt'].isin(timeline)].copy()

    ic = {}
    ic['ic_values'] = {}
    read_path = r'/home/xiaonan/factor_wxn/factor/'
    read_path = os.path.join(read_path, name)
    pool = dM.load_universe(start,end,univ_name = 'TOP2000')
    pool = pool.apply(lambda x:x.apply(lambda x:int(x))).copy()
    for date in timeline:
        if os.path.exists(read_path + r'/' + str(date) + '.csv'):
            factor = pd.read_csv(read_path + r'/' + str(date) + '.csv', index_col=0, header=None)
        else:
            continue
        factor.columns = ['nouse','values']
        _ret = ret[ret['dt'] == date].set_index('code').copy()
        comindex = _ret.index & factor.index
        
        _ic = factor.loc[pool.loc[date,'code'].values,['values']].corrwith(_ret.loc[pool.loc[date,'code'].values, retinterval], method = 'spearman').iloc[0]
        ic['ic_values'][str(date)] = _ic

    ans = pd.DataFrame(ic).sort_index()
    print(ans.mean().iloc[0])
    
    fig, ax = plt.subplots()

    ax.bar(x = ans.index,height = ans['ic_values'].apply(lambda x:x if x>=0 else 0).values, color = '#13CCB1',label = 'ic(left +)')
    ax.bar(x = ans.index,height = ans['ic_values'].apply(lambda x:x if x<0 else 0).values, color = '#EACC80', label = 'ic(left -)')

    for i in ['top', 'bottom', 'left', 'right']:
        ax.spines[i].set_visible(True)
    ax.yaxis.grid(linestyle='--',alpha = 0.3)
    ax.set_title('{} {}'.format(name,round(ans.mean().iloc[0],3)))

    ax.legend(loc = 'upper left')
    ax1 = ax.twinx()
    ax1.plot(ans.cumsum(),color = 'darkgrey', label='acc_ic(right)')
    ax1.legend(loc = 'upper right')

    for i,tick in enumerate(ax.get_xticklabels()):
        if i%128 == 0:
            tick.set_visible(True)
            tick.set_rotation(30)
        else:
            tick.set_visible(False)
    for i,tick in enumerate(ax.get_xticklines()):
        if i%128 == 0:
            tick.set_visible(True)
        else:
            tick.set_visible(False)
    
    plt.savefig('./rankIcFig/{}.jpg'.format(name))
    if plot:
        plt.show(fig)
    
def plot_all_ic(start,end,reinterval = 'y_close_5',plot = False):
    for i in tqdm(os.listdir(r'/home/xiaonan/factor_wxn/factor/')):
        try:
            plot_IC(start, end, i, reinterval, plot)
        except:
            print(i, ' failed.')
            print('\n########################\n')


def FMregression(start, end, factor_list):
    def regression(start, end, factor, ret):
        path = r'/home/xiaonan/factor_wxn/factor/'
        path = os.path.join(path, factor)

        x = {}
        y = {}
        for date in ts.get_trading_date(start, end):
            _ans = pd.read_csv(path + r'/' + str(date) + r'.csv', index_col=0,header= None)
            _ans.columns = ['nouse', 'values']
            
            x[date] = _ans.loc[:,'values']

            y[date] = ret[['code','y_close_5']][ret['dt'] == date].set_index('code')['y_close_5']
        
        x = pd.DataFrame(x).T.fillna(0)
        y = pd.DataFrame(y).T.fillna(0)
        code_list = x.columns
        y = y.loc[:,x.columns]
        x = x.to_dict('series')
        y = y.to_dict('series')

        beta = {}
        for code in code_list:
            _x = sm.add_constant(x[code])
            _y = y[code]
            model = sm.OLS(_y,_x)
            r = model.fit()
            beta[code] = r.params.iloc[-1]
        
        beta = pd.DataFrame(beta,index = ['beta']).T
        lambdai = {}
        y = pd.DataFrame(y).T.to_dict('series')
        for date in ts.get_trading_date(start, end):
            model = sm.OLS(y[date], sm.add_constant(beta))
            r = model.fit()
            lambdai[date] = r.params.iloc[-1]

        lambdai = pd.DataFrame(lambdai,index = ['lambda']).T
        lambdai.to_csv('lambda.csv')
        print(lambdai.mean())



    ret = pd.read_parquet('/home/sharedFold/zhaorui/ret.parquet')

    for factor in factor_list:
        regression(start, end, factor, ret)

def count_nan():
    ans = []
    timeline = ts.get_trading_date(start, end)
    path = r'/home/xiaonan/factor_wxn/rawFactor/'
    for factor_name in tqdm(os.listdir(path)):
        if factor_name == '.directory':
            continue
        for date in timeline:
            read_path = path + factor_name + r'/' + str(date) + r'.csv'
            data = pd.read_csv(read_path, index_col=0, header=None)
            data.columns = ['nouse', 'values']
            if data['values'].isnull().sum() >=30:
                ans.append((factor_name,date))
    np.save('list', ans)

def portfolio(start, end, factor_list, inSample = 100, outSample = 10, univ_name = 'TOP2000'):
    print('loading data......')
    NUM = 200
    ret = pd.read_parquet('/home/sharedFold/zhaorui/ret.parquet')
    pool = dM.load_universe(start,end,univ_name = univ_name)
    timeline = ts.get_trading_date(start,end)
    allFactor = pd.DataFrame()
    for factor_name in factor_list:
        read_path = '/home/xiaonan/factor_wxn/factor/' + factor_name +r'/'
        ans = []
        for date in timeline:
            data = pd.read_csv(read_path + str(date) + r'.csv', header = None)
            data.drop([1],inplace = True,axis = 1)
            data.columns = ['code',factor_name]
            data.loc[:,'date'] = date
            ans.append(data)
        ans = pd.concat(ans)
        if allFactor.empty:
            allFactor = ans.copy()
        else:
            allFactor = pd.merge(allFactor, ans, left_on=['code','date'], right_on=['code','date'], how='outer')
    allData = pd.merge(allFactor, ret.loc[:,['y_close_1','code','dt']], left_on=['code','date'], right_on=['code', 'dt'], how = 'left')

    print('loading data end\n')
    print('backtest started')
    allInSample, allOutSample, allNeutral = [],[], []
    predict_days = np.arange(inSample, len(timeline), outSample)
    for i in predict_days:
        inPool = pool.loc[timeline[i-inSample]: timeline[i]]
        if i + outSample < len(timeline):
            outPool = pool.loc[timeline[i]: timeline[i+outSample]]
        else:
            outPool = pool.loc[timeline[i]: ]

        data_inSample = allData[allData.code.isin(np.unique(inPool.values.flatten())) & allData.date.isin(timeline[i-inSample:i])].copy()
        data_outSample = allData[allData.code.isin(np.unique(outPool.values.flatten())) & allData.date.isin(timeline[i:i+outSample])].copy()

        x = data_inSample.loc[:, factor_list].fillna(data_inSample.loc[:, factor_list].mean())
        y = data_inSample.y_close_1.fillna(data_inSample.y_close_1.mean())
        model = sm.OLS(y, sm.add_constant(x))
        r = model.fit()

        x = data_inSample.loc[:, factor_list].fillna(data_inSample.loc[:, factor_list].mean())
        predict_inSample = r.predict(sm.add_constant(x))
        data_inSample.loc[:,'predict'] = predict_inSample

        x = data_outSample.loc[:, factor_list].fillna(data_outSample.loc[:, factor_list].mean())
        predict_outSample = r.predict(sm.add_constant(x))
        data_outSample.loc[:,'predict'] = predict_outSample

        _in = data_inSample.groupby('date').apply(lambda x:x[['predict','y_close_1']].sort_values('predict').iloc[-NUM:,1].mean()).copy()
        _out = data_outSample.groupby('date').apply(lambda x:x[['predict','y_close_1']].sort_values('predict').iloc[-NUM:,1].mean()).copy()
        _out_neutral = _out - data_outSample.groupby('date').apply(lambda x:x['y_close_1'].mean()).copy()

        allInSample.append(_in.loc[timeline[i-inSample]: timeline[i]])
        allOutSample.append(_out)
        allNeutral.append(_out_neutral)

    allInSample = pd.concat(allInSample).sort_index()
    allInSample.index = allInSample.index.map(str)

    allOutSample = pd.concat(allOutSample).sort_index()
    allOutSample.index = allOutSample.index.map(str)

    allNeutral = pd.concat(allNeutral).sort_index()
    allNeutral.index = allNeutral.index.map(str)

    print((0.9997 * (1+allOutSample)).cumprod().iloc[-1])
    fig,ax = plt.subplots(3,1)

    ax[0].plot((0.9997 * (1+allInSample)).cumprod(),color = 'r')
    ax[1].plot((0.9997 * (1+allOutSample)).cumprod(),color = 'b')
    ax[2].plot((0.9997 * (1+allNeutral)).cumprod(),color = 'y')
    
    for t in [0,1,2]:
        for i,tick in enumerate(ax[t].get_xticklabels()):
            if i%128 == 0:
                tick.set_visible(True)
                tick.set_rotation(30)
            else:
                tick.set_visible(False)

        for i,tick in enumerate(ax[t].get_xticklines()):
            if i%128 == 0:
                tick.set_visible(True)
            else:
                tick.set_visible(False)

    fig.savefig('linearmodel.jpg')
    plt.show(fig)


def portfolio_beta(start, end, factor_list, inSample = 100, outSample = 10, univ_name = 'TOP2000'):
    NUM = 200
    print('loading data......')
    ret = pd.read_parquet('/home/sharedFold/zhaorui/ret.parquet')
    pool = dM.load_universe(start,end,univ_name = univ_name)
    timeline = ts.get_trading_date(start,end)
    allFactor = pd.DataFrame()
    for factor_name in factor_list:
        read_path = '/home/xiaonan/factor_wxn/factor/' + factor_name +r'/'
        ans = []
        for date in timeline:
            data = pd.read_csv(read_path + str(date) + r'.csv', header = None)
            data.drop([1],inplace = True,axis = 1)
            data.columns = ['code',factor_name]
            data.loc[:,'date'] = date
            ans.append(data)
        ans = pd.concat(ans)
        if allFactor.empty:
            allFactor = ans.copy()
        else:
            allFactor = pd.merge(allFactor, ans, left_on=['code','date'], right_on=['code','date'], how='outer')
    allData = pd.merge(allFactor, ret.loc[:,['y_close_1','code','dt']], left_on=['code','date'], right_on=['code', 'dt'], how = 'left')

    print('loading data end\n')
    print('backtest started')
    allInSample, allOutSample, allNeutral = [],[],[]
    predict_days = np.arange(inSample, len(timeline), outSample)
    for i in tqdm(predict_days):
        inPool = pool.loc[timeline[i-inSample]: timeline[i]]
        if i + outSample < len(timeline):
            outPool = pool.loc[timeline[i]: timeline[i+outSample]]
        else:
            outPool = pool.loc[timeline[i]: ]

        data_inSample = allData[allData.code.isin(np.unique(inPool.values.flatten())) & allData.date.isin(timeline[i-inSample:i])].copy()
        data_outSample = allData[allData.code.isin(np.unique(outPool.values.flatten())) & allData.date.isin(timeline[i:i+outSample])].copy()

        x = data_inSample.loc[:, factor_list].fillna(data_inSample.loc[:, factor_list].mean())
        y = data_inSample.y_close_1.fillna(data_inSample.y_close_1.mean())
        from sklearn.ensemble import RandomForestRegressor
        model = LinearRegression()
        r = model.fit(y = y.to_frame(), X = x)

        x = data_inSample.loc[:, factor_list].fillna(data_inSample.loc[:, factor_list].mean())
        predict_inSample = r.predict(x)
        data_inSample.loc[:,'predict'] = predict_inSample

        x = data_outSample.loc[:, factor_list].fillna(data_outSample.loc[:, factor_list].mean())
        predict_outSample = r.predict(x)
        data_outSample.loc[:,'predict'] = predict_outSample

        _in = data_inSample.groupby('date').apply(lambda x:x[['predict','y_close_1']].sort_values('predict').iloc[-NUM:,1].mean()).copy()
        _out = data_outSample.groupby('date').apply(lambda x:x[['predict','y_close_1']].sort_values('predict').iloc[-NUM:,1].mean()).copy()
        _out_neutral = _out - data_outSample.groupby('date').apply(lambda x:x['y_close_1'].mean()).copy()

        allInSample.append(_in.loc[timeline[i-inSample]: timeline[i]])
        allOutSample.append(_out)
        allNeutral.append(_out_neutral)

    allInSample = pd.concat(allInSample).sort_index()
    allInSample.index = allInSample.index.map(str)

    allOutSample = pd.concat(allOutSample).sort_index()
    allOutSample.index = allOutSample.index.map(str)

    allNeutral = pd.concat(allNeutral).sort_index()
    allNeutral.index = allNeutral.index.map(str)

    fig,ax = plt.subplots(3,1)

    ax[0].plot((0.9997 * (1+allInSample)).cumprod(),color = 'r')
    ax[1].plot((0.9997 * (1+allOutSample)).cumprod(),color = 'b')
    ax[2].plot((0.9997 * (1+allNeutral)).cumprod(),color = 'y')
    
    for t in [0,1,2]:
        for i,tick in enumerate(ax[t].get_xticklabels()):
            if i%128 == 0:
                tick.set_visible(True)
                tick.set_rotation(30)
            else:
                tick.set_visible(False)

        for i,tick in enumerate(ax[t].get_xticklines()):
            if i%128 == 0:
                tick.set_visible(True)
            else:
                tick.set_visible(False)
                
    fig.savefig('model_beta.jpg')
    # plt.show(fig)

    return (0.9997 * (1+allNeutral)).cumprod()


if __name__ == "__main__":
    a = datetime.datetime.now()
    start = 20150106
    end = 20200820
    
    corr_heatmap(20200827)
    # factorTimeInterval()
    # test_IC()
    # plot_all_ic(start, end, reinterval = 'y_close_5',plot = False)
    # plot_IC(start,end,'corr_skew', plot = True)
    # rawFactor_move()
    # date = 20200804
    # portfolio(date)
    
    # print(len(ts.get_trading_date(20171215, 20200814)))
    '''
    a_factor_list = ['daily_var','daily_skew', 'market_depth_amount','price_vol_corr','corr_mean']
    b_factor_list = ['daily_var','daily_skew', 'market_depth_amount','price_vol_corr','corr_mean','large_order_amount_ratio','vol_mean_price']
    c = ['daily_up_var','daily_down_var','daily_kurt','market_depth_askmount', 'market_depth_bidamount','daily_ret_var', 'daily_ret_up_var', 'daily_ret_down_var',
    'fivemins_ret_down_var', 'bid_ask_spread', 'askid_num', 'bidid_num', 'pareto_buys', 'small_order', 'large_order', 'netinflows', 'fivemins_max_gains', 'trend_strength']
    # factor_list = ['daily_ret_var', 'bid_ask_spread', 'fivemins_max_gains', 'trend_strength', 'open_call_auction', 'daily_ret_skew']
    c_factor_list = ['daily_maxgain','daily_maxdrawdown', 'daily_meangain','daily_meandrawdown', 'break_highpoint', 'break_lowpoint']
    
    factor_list = []
    for i in os.listdir('/home/xiaonan/factor_wxn/factor/'):
        if os.path.exists('/home/xiaonan/factor_wxn/factor/' + i + '/20150106.csv'):
            
            if i in a_factor_list:
                continue
            if i in b_factor_list:
                continue
            if i in c_factor_list:
                continue
            if i in c:
                continue
            
            factor_list.append(i)
    print(len(factor_list))
    ret = portfolio_beta(start, end, factor_list)
    
    # portfolio(start, end, os.listdir('/home/xiaonan/factor_wxn/factor/'))
    # count_nan()

    print('\n{}\n'.format(datetime.datetime.now() - a))
    '''
from _utils import LoadData, ReportFig
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import paraCorr

import matplotlib.pyplot as plt
import sys, os
sys.path.append('/data/stock/newSystemData/feature_base/structure/yili/db_utilise/')
import trading_date as ts

# data will use zhaorui's function.

class Test:
    def group_test(x, y):
        pass


class DailyReport:
    # def __init__(self, start, end, factor_path, factor_name, write_path, trading_settlement = 'close_to_close', delay=0, day=1):
    def __init__(self, factor_path, write_path):
        self.factor_path = factor_path
        self.write_path = write_path
        
        self.univ_name ='TOP2000' 

        self.LoadData = LoadData(factor_path)
        self.ReportFig = ReportFig()

        self.factor = {}
        
    
    def get_factor_data(self, start, end, factor_name, delay):
        return self.LoadData.get_factor(start, end, factor_name, delay)

    def preprocessing(self, start, end, factor_name, delay = 0):
        self.ret = self.LoadData.get_return()
        self.factor[factor_name] = self.get_factor_data(start, end, factor_name, delay)
    
    def distribute(self, start, end, factor_name):
        self.preprocessing(start, end, factor_name)
        _ = self.factor[factor_name]['values'].dropna()
        self.ReportFig.factorDistribute(_, self.write_path + factor_name + r'/')

    
    def group_test(self, start, end, factor_name, n = 5, trading_settlement = 'close2close', delay = 0, day = 1):
        self.preprocessing(start, end, factor_name, delay)

        _factor = self.factor[factor_name].copy()

        allData = pd.merge(_factor, self.ret, how = 'left', left_on=['code','date'], right_on=['code','dt'])
        allData = allData.values

        timeline = ts.get_trading_date(start, end)
        _ret = {}
        _val = {}
        for i in range(allData.shape[0]):
            if allData[i,2] in _ret.keys():
                pass
            else:
                _ret[allData[i,2]] = []
                _val[allData[i,2]] = []

            if np.isnan(allData[i,5]):
                _ret[allData[i,2]].append(0)
            else:
                _ret[allData[i,2]].append(allData[i,5])
            if np.isnan(allData[i,1]):
                _val[allData[i,2]].append(0)
            else:
                _val[allData[i,2]].append(allData[i,1])
        
        for i in range(n):
            locals()['group_{}'.format(i)] = []
        

        for i in tqdm(timeline):
            _ret_i = np.array(_ret[i])

            _val_i = np.array(_val[i])
            percent = []
            for j in range(n+1):
                percent.append(np.percentile(_val_i, 100/n *j))
            percent[-1] += 1

            for j in range(n):
                lay = (_val_i >= percent[j]) & (_val_i < percent[j+1])
                if np.sum(lay) > 0:
                    if np.isnan(np.mean(_ret_i[lay])):
                        locals()['group_{}'.format(j)].append(np.mean(_ret_i))
                    else:
                        locals()['group_{}'.format(j)].append(np.mean(_ret_i[lay]))
                else:
                    locals()['group_{}'.format(j)].append(np.mean(_ret_i))

        if os.path.exists(self.write_path + factor_name + r'/'):
            pass
        else:
            os.mkdir(self.write_path + factor_name + r'/')

        group = {}
        for i in range(n):
            group[str(i)] = locals()['group_{}'.format(i)]
        _ = ((pd.DataFrame(group, index = list(map(str, timeline))) + 1) * 0.997).cumprod()
        pd.DataFrame(group, index = list(map(str, timeline))).to_csv('group.csv')
        self.ReportFig.groupFig(_.apply(lambda x:x - x.mean(), axis = 1), self.write_path + factor_name + r'/')

        return group

    def ic_test(self, start, end, factor_name, trading_settlement = 'close2close', delay = 0, day = 1):
        self.preprocessing(start, end, factor_name, delay)

        _factor = self.factor[factor_name].copy()
        
        allData = pd.merge(_factor, self.ret, how = 'left', left_on=['code','date'], right_on=['code','dt'])
        
        # print(allData.columns)
        allData = allData.values

        timeline = ts.get_trading_date(start, end)
        _ret = {}
        _val = {}
        for i in range(allData.shape[0]):
            if allData[i,2] in _ret.keys():
                pass
            else:
                _ret[allData[i,2]] = []
                _val[allData[i,2]] = []

            if np.isnan(allData[i,5]):
                _ret[allData[i,2]].append(0)
            else:
                _ret[allData[i,2]].append(allData[i,5])
            if np.isnan(allData[i,1]):
                _val[allData[i,2]].append(0)
            else:
                _val[allData[i,2]].append(allData[i,1])
        
        kinds = ['AllIC','xBottom','xTop','yBottom','yTop']
        ic = {}
        for i in kinds:
            ic[i] = []
        
        for i in tqdm(timeline):
            a = paraCorr.one_corr(np.array(_ret[i]), np.array(_val[i]))
            if ic['AllIC']:
                ic['AllIC'].append(ic['AllIC'][-1] + a)
            else:
                ic['AllIC'].append(a)

            lay = (np.array(_ret[i]) > np.median(_ret[i]))
            ytop = paraCorr.one_corr(np.array(_ret[i])[lay], np.array(_val[i])[lay])
            if ic['yTop']:
                ic['yTop'].append(ic['yTop'][-1] + ytop)
            else:
                ic['yTop'].append(ytop)

            lay = (np.array(_ret[i]) < np.median(_ret[i]))
            ybottom = paraCorr.one_corr(np.array(_ret[i])[lay], np.array(_val[i])[lay])
            if ic['yBottom']:
                ic['yBottom'].append(ic['yBottom'][-1] + ybottom)
            else:
                ic['yBottom'].append(ybottom)
            
            lay = (np.array(_val[i]) > np.median(_val[i]))
            xtop = paraCorr.one_corr(np.array(_ret[i])[lay], np.array(_val[i])[lay])
            if ic['xTop']:
                ic['xTop'].append(ic['xTop'][-1] + xtop)
            else:
                ic['xTop'].append(xtop)

            lay = (np.array(_val[i]) < np.median(_val[i]))
            xbottom = paraCorr.one_corr(np.array(_ret[i])[lay], np.array(_val[i])[lay])
            if ic['xBottom']:
                ic['xBottom'].append(ic['xBottom'][-1] + xbottom)
            else:
                ic['xBottom'].append(xbottom)
            
        if os.path.exists(self.write_path + factor_name + r'/'):
            pass
        else:
            os.mkdir(self.write_path + factor_name + r'/')
        
        self.ReportFig.icFig(pd.DataFrame(ic, index = list(map(str,timeline))), self.write_path + factor_name + r'/')
        pd.DataFrame(ic, index = list(map(str,timeline))).to_csv('ic.csv')
        return ic
        
    def ratio(self, start, end, factor_name, freq = '2M', trading_settlement = 'close2close', delay = 0, day = 1):
        group = self.group_test(start, end, factor_name, trading_settlement = 'close2close', delay = 0, day = 1)
        longshort = np.cumprod(np.array(group['0']) * 0.997 + 0.997) - np.cumprod(np.array(group['4']) * 0.997 + 0.997)
        timeline = ts.get_trading_date(start, end)
        time_split = []
        start_point = str(timeline[0])
        print(start_point)
        print(start_point[4:6])
        for i in timeline:
            pass
        # fig, ax = self.ReportFig.plot(pd.DataFrame(longshort, index = list(map(str, timeline))))
        # plt.show(fig)
        
    
if __name__ == '__main__':
    start = 20150106
    end = 20200801
    factor_path = r'/home/xiaonan/factor_wxn/factor/'
    factor_name = 'price_vol_corr'
    write_path = r'/home/xiaonan/factor_wxn/SignalAnalysis/Report/'
    A = DailyReport(factor_path, write_path)
    A.ratio(start,end,factor_name)
    # A.distribute(start, end,factor_name)
    # A.group_test(start, end, factor_name)
    # A.ic_test(start, end, factor_name)

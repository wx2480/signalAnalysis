import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed
import sys, os
sys.path.append('/data/stock/newSystemData/feature_base/structure/yili/db_utilise/')
import trading_date as ts



class LoadData:
    def __init__(self, factor_path):
        self.factor_path = factor_path
        self.ret_path = r'/home/sharedFold/zhaorui/ret.parquet'

    def get_factor(self, start, end, factor_name, delay):
        def get_one(date):
            path = read_path + r'/' + str(date) + '.csv'
            factor = pd.read_csv(path, index_col=None, header=None)
            factor.columns = ['code', 'nouse','values']
            factor.loc[:,'date'] = ts.get_nxt_trading_dates(date, delay + 1)[-1]
            return factor

        timeline = ts.get_trading_date(start, end)
        
        read_path = os.path.join(self.factor_path, factor_name)
        

        ans = Parallel(10)(delayed(get_one)(date) for date in tqdm(timeline))
        '''
        for date in timeline:
            path = read_path + r'/' + str(date) + '.csv'
            factor = pd.read_csv(path, index_col=None, header=None)
            factor.columns = ['code', 'nouse','values']
            factor.loc[:,'date'] = date
            ans.append(factor)
        '''
        factor = pd.concat(ans)
        factor.reset_index(inplace = True)
        factor.drop(['index','nouse'], axis = 1, inplace = True)

        return factor

    def get_return(self):
        ret = pd.read_parquet(self.ret_path)
        return ret


class Fig:
    def __init__(self):
        # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['figure.figsize'] = (15,5)
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['axes.axisbelow'] = True
        self.colors = ['#44A948', '#137CB1', '#EACC80', '#A8D7E2', '#E58061']

    def plot(self, data, accumulate = False, **kwargs):
        if accumulate:
            data = data.cumsum()
        
        fig, ax = plt.subplots()

        for i,label in enumerate(data.columns):
            ax.plot(data.iloc[:,i], color = self.colors[i], label = label)

        for i in ['top', 'bottom', 'left', 'right']:
                ax.spines[i].set_visible(True)
        ax.yaxis.grid(linestyle='--',alpha = 0.3)

        for j,tick in enumerate(ax.get_xticklabels()):
            if j%128 == 0:
                tick.set_visible(True)
                tick.set_rotation(15)
            else:
                tick.set_visible(False)
        
        for j,tick in enumerate(ax.get_xticklines()):
            if j%128 == 0:
                tick.set_visible(True)
            else:
                tick.set_visible(False)
        ax.legend(loc = 'upper left')
        return fig, ax

class ReportFig(Fig):
    def __init__(self):
        super().__init__()

    def groupFig(self, data, write_path = None):
        fig, ax = self.plot(data)
        factor_name = write_path.split(r'/')[-2]
        ax.set_title(factor_name)
        if write_path:
            fig.savefig(write_path + 'group_test.jpg')
        else:
            plt.show(fig)

    def icFig(self, data, write_path = None):
        fig, ax = self.plot(data,accumulate=True)
        factor_name = write_path.split(r'/')[-2]
        ax.set_title(factor_name)
        if write_path:
            fig.savefig(write_path + 'ic_test.jpg')
        else:
            plt.show(fig)

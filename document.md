```python
class ReportFig:
    """
    画图的基类，需要传入符合标准的数据。
    """
    def __init__(self):
        """
        一些画图固定的配置，比如刻度线以及调色盘，统一风格
        """
        pass
    
    def bar(self, ax, data, **kwargs):
        """
        柱状图
        =============parameters==============
        ax：传入axes
        data：需要绘制的数据，index是日期或者是其他的分组之类的，画在横轴上，columns是y轴，columns长度必须为1
        return：axes
        """
        pass
    
    def plot(self, ax, data, accumulate = False, **kwargs):
        """
        折线图
        =============parameters==============
        ax：axes
        data：index是日期或者其他的，画在横轴上，columns是y轴，如果columns长度不是1，那么会画多条曲线
        accumulate：曲线是否是累计值，default不画累计值
        return：axes
        """
        pass
    
    def groupFig(self, data, write_path = None):
        """
        分组检测的图：
        1、分组检测的净值曲线
        2、分组检测的年化收益率柱状图
        =============parameters==============
        data：每天的净值
        write_path：写入路径，如果未给的话会显示出来
        return：规定路径下的jpg
        """
        pass
    
    def icFig(self, data,write_path = None):
        """
        5个IC的折线图
        =============parameters==============
        data：五个IC的DataFrame
        write_path：写入路径，如果未给的话会显示出来
        return：规定路径下的jpg
        """
        pass
    

class LoadData:
    def __init__(self):
        """
        初始化路径
        """
        pass
    
    def get_factor(self, factor_name):
        """
        获取因子值
        =============parameters==============
        factor_name:因子名称
        """
        pass
    
    def get_return(self):
        """
        获取收益率（读取兆瑞哥写的ret文件）
        =============parameters==============
        """
        pass
    
class DailyReport(ReportFig, LoadData):
    """
    一共六张图（都去beta了）
    第一张是因子的分布
    第二张是分组检测的净值
    第三张是分组检测的柱状图
    第四张是rankic的五条折线
    第五张是多空组合的净值以及回撤
    第六张图是一个不同delay的比较，假设delay从0到5，就会有一张图，六个子图，delay0，1，2，3，4，5
    """
    def __init__(self, start, end, factor_path, factor_name, write_path, trading_settlement = 'close_to_close', delay = 0, day = 1):
        """
        载入数据以及初始化后面用到的参数
        =============parameters==============
        start：开始日期    ex：20200101
        end：结束日期    ex：20200804    开始以及结束日期之间必须有交易日存在，而且交易日长度需要大于delay + day
        path：因子存储路径
        factor_name：需要产生报告的因子名称 ex：'/home/xiaonan/market_depth_amount/'
        trading_settlement：close_to_close,open_to_open
        delay：延迟几天
        day：看几日收益率
        """
        pass
    
    def group_test(self):
        """
        分组检测
        =============parameters==============
        return：累计净值曲线的图以及年化收益率的柱状图以及每日收益率的DataFrame，还有从start，到end所有因子值的分布
        """
        pass
    
    def ic_test(self):
        """
        return：IC检验画的图
        """
        pass
    
    def longShort(self):
        """
        return：多空组合（方向由AllIC判断）的折线图以及回撤
        """
        pass
    
    def daily_ratio(self):
        """
        return：计算多空组合相关的指标，覆盖率1，覆盖率2，自相关性，夏普比率，最大回撤，卡玛比率
        """
        pass
    
    def delay_test(self, delays):
        """
        把不同和delay的多空净值曲线画在一起
        """
        pass
    
    def output(self):
        """
        return：整合到word
        """
        pass
```

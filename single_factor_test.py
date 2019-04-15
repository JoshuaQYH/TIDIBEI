"""
-------------------------------------------------------
策略思路：
1. 回测标的：沪深300成分股
2. 回测时间段：2016-01-01 至 2016-12-31
3. 特征选择：待测单因子
4. 单因子回归测试模型思路：
    1. 先获得 50 天以上的 K线数据和因子数据；因子数据必须进行！预处理！
    2. 以25 天为一次训练单位，其中前 20 天的因子作为训练样本特征，
    3. 使用单变量线性模型进行训练。
    4. 回到当前时间点，使用前 20 天的因子数据作为预测样本特征，预测后 5 天的各股票平均收益率的大小。
5. 选股逻辑：
    将符合预测结果的股票按均等分配可用资金进行下单交易。持有 5 天之后，再次进行训练预测。
6. 交易逻辑：
    每次调仓时，若当前有持仓，并且符合选股条件，则仓位不动；
                              若不符合选股条件，则所有仓位平仓；
                若当前无仓，并且符合选股条件，则多开仓；
                            若不符合选股条件，则不开仓，无需操作。

----------
运行方法：
1. 在 main 中定义同一类的因子列表。
2. 逐个因子执行回测。
3. 获取回测报告ID，通过ID获取绩效报告字段。
4. 保留字段到CSV文件中。
"""

from atrader import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
from sklearn import preprocessing

# 作为全局变量进行测试
FactorCode = ["PB"]


# 中位数去极值法
def filter_MAD(df, factor, n=3):
    """
    :param df: 去极值的因子序列
    :param factor: 待去极值的因子
    :param n: 中位数偏差值的上下界倍数
    :return: 经过处理的因子dataframe
    """
    median = df[factor].quantile(0.5)
    new_median = ((df[factor] - median).abs()).quantile(0.5)
    max_range = median + n * new_median
    min_range = median - n * new_median

    for i in range(df.shape[0]):
        if df.loc[i, factor] > max_range:
            df.loc[i, factor] = max_range
        elif df.loc[i, factor] < min_range:
            df.loc[i, factor] = min_range
    return df


def init(context):
    # 账号设置：设置初始资金为 1000000 元
    set_backtest(initial_cash=1000000)
    # 注册数据：日频数据
    reg_kdata('day', 1)
    global FactorCode  # 全局单因子代号
    reg_factor(factor=FactorCode)

    context.FactorCode = FactorCode

    # 超参数设置：
    context.Len = 50  # 时间长度: 当交易日个数小于该事件长度时，跳过该交易日
    # 我们的目的就是使用前20天的因子数据作为样本，未来5天的股票收益率作为标签，进行回归。
    # 在前50天，可以取多组样本和标签，构成回归测试的数据
    context.N1 = 20   # 训练样本的时间跨度，代表训练的天数  // 样本过高过低都不太好，过高可能有噪声，过低数据特征不明显
    context.N2 = 5   # 标签的时间跨度,代表预测未来的天数  // 换手率相应会增加,运行时间也增加。
    context.Num = 0   # 记录当前交易日个数


def on_data(context):
    context.Num = context.Num + 1
    if context.Num < context.Len:  # 如果交易日个数小于Len+1，则进入下一个交易日进行回测
        return
    if bool(context.Num % context.N2):  # 每隔 N2 天进行调仓
        return

    # 获取数据：
    KData = get_reg_kdata(reg_idx=context.reg_kdata[0], length=context.Len, fill_up=True, df=True)
    FData = get_reg_factor(reg_idx=context.reg_factor[0], target_indices=[x for x in range(300)], length=context.Len,
                           df=True)  # 获取因子数据

    # 特征构建：
    Fcode = list()
    for i in range(context.N1):
        for FC in context.FactorCode:
            Fcode.append(FC + str(i))  # 因子 + 序号

    # 数据存储变量：
    FactorData = pd.DataFrame(columns=(['idx', 'signClose'] + Fcode))  # 存储训练特征及标签样本
    FactorDataTest = pd.DataFrame(columns=(['idx'] + Fcode))  # 存储预测特征样本

    # 序号：
    tempIdx = KData[KData['time'] == KData['time'][0]]['target_idx'].reset_index(drop=True)

    # 按标的处理数据：
    for i in range(300):
        # 训练特征集及训练标签构建：
        # 临时数据存储变量:
        FactorData0 = pd.DataFrame(
            columns=(['idx', 'signClose'] + Fcode))
        FactorDataTest0 = pd.DataFrame(np.full([1, len(Fcode) + 1], np.nan), columns=(['idx'] + Fcode))  # 存储预测特征样本

        # 序号对齐
        FData0 = FData[FData['target_idx'] == tempIdx[i]].reset_index(drop=True)

        global FactorCode  # 全局单因子代号

        FData0.dropna(axis=0)  # 删除因子缺失的股票
        FData0 = filter_MAD(FData0, "value", 3)  # 中位数去极值法
        FData0["value"] = preprocessing.scale(FData0["value"])  # 标准化

        # 按特征处理数据：
        for FC in context.FactorCode:
            FCData = FData0[FData0['factor'] == FC]['value'].reset_index(drop=True)
            for k in range(context.N1):
                FactorData0[FC + str(k)] = FCData[k:k + context.Len - context.N1 - context.N2]

        # 按标签处理数据：
        close = np.array(KData[KData['target_idx'] == tempIdx[i]]['close'])
        # 将N2内的天收益率作为标签
        signValue = np.sign((close[context.N1 + context.N2:] - close[context.N1:context.Len - context.N2]) / close[
                                                                                                             context.N1:context.Len - context.N2])  # 标签构建
        FactorData0['signClose'] = signValue
        # idx:
        FactorData0['idx'] = tempIdx[i]
        # 合并数据：
        FactorData = FactorData.append(FactorData0, ignore_index=True)

        # 预测特征集构建：
        FactorDataTest0['idx'] = tempIdx[i]
        # 按特征处理数据：
        for FC in context.FactorCode:
            FCData = FData0[FData0['factor'] == FC]['value'].reset_index(drop=True)
            for k in range(context.N1):
                FactorDataTest0[FC + str(k)] = FCData[context.Len - context.N1 + k]
        FactorDataTest = FactorDataTest.append(FactorDataTest0, ignore_index=True)
        print(i)

    # 数据清洗：
    FactorData = FactorData.dropna(axis=0, how='any').reset_index(drop=True)  # 清洗数据
    FactorDataTest = FactorDataTest.dropna(axis=0, how='any').reset_index(drop=True)  # 清洗数据
    Idx = FactorDataTest['idx']  # 剩余标的序号

    # logistic回归模型：
    # 预测特征构建：
    X = np.ones([FactorData.shape[0], len(Fcode)])
    Xtest = np.ones([FactorDataTest.shape[0], len(Fcode)])
    for i in range(X.shape[1]):
        X[:, i] = FactorData[Fcode[i]]
        Xtest[:, i] = FactorDataTest[Fcode[i]]

    Y = np.array(FactorData['signClose']).astype(int)

    # 构建模型：
    LRModel = LinearRegression()
    print(X)
    print(Y)
    # 模型训练：
    LRModel = LRModel.fit(X, Y)
    # LR分类预测：
    y = LRModel.predict(Xtest)

    # 交易设置：
    positions = context.account().positions['volume_long']  # 多头持仓数量
    valid_cash = context.account(account_idx=0).cash['valid_cash'][0]  # 可用资金

    P = 0.6 / sum(y > 0)  # 设置每只标的可用资金比例

    for i in range(len(Idx)):
        position = positions.iloc[Idx[i]]
        if position == 0 and y[i] > 0:  # 当前无仓，且该股票收益大于0，则开仓，买入
            Num = int(math.floor(valid_cash * P / 100 / KData['close'][Idx[i]]) * 100)  # 开仓数量
            order_volume(account_idx=0, target_idx=int(Idx[i]), volume=Num, side=1, position_effect=1, order_type=2,
                         price=0)  # 指定委托量开仓
        elif position > 0 and y[i] < 0: # 当前持仓，且该股票收益小于0，则平仓，卖出
            order_volume(account_idx=0, target_idx=int(Idx[i]), volume=int(position), side=2, position_effect=2,
                         order_type=2, price=0)  # 指定委托量平仓


if __name__ == '__main__':

    """
    测试因子分类别进行，因子ID请看下列链接
    https://www.digquant.com.cn/documents/23#h2-1-1-revs250--330
    """

    factor_list = ["BIAS20"]  # 将要测试的因子写入这个列表。

    file_path = 'single_factor_test.py'
    block = 'hs300'

    begin_date = '2016-01-01'
    end_date = '2016-06-30'

    for factor in factor_list:
        startegy_name = factor
        FactorCode = [factor]  # 修改全局变量
        print(FactorCode)
        try:
            run_backtest(strategy_name=startegy_name, file_path=file_path,
                        target_list=list(get_code_list('hs300', date=begin_date)['code']),
                        frequency='day', fre_num=1, begin_date=begin_date, end_date=end_date, fq=1)
        except Exception:
            print("该因子回测报告出错,跳过。")
            pass

    """
    ... 此时可能出现报告还未计算结束的情况，得重开另外一个文件获取字段。
    
    atSendCmdGetBackTestPerformance, Code: 2, Reason: strategy backtest is still running
    
    strategy_dicts = get_strategy_id()

    save_dict = {"测试因子": [],
                 '年化收益率': [],
                 '年化夏普率': [],
                 '最大回撤率': [],
                 'alpha': [],
                 'beta': [],
                 '信息比率': []
                 }
    for strategy in strategy_dicts:
        strategy_id = strategy["strategy_id"]
        result = get_performance(strategy_id)
        save_dict['测试因子'].append(result['strategy_name'])
        save_dict['年化收益率'].append(result['annu_return'])
        save_dict['年化夏普率'].append(result['sharpe_ratio'])
        save_dict['最大回撤率'].append(result['max_drawback_rate'])
        save_dict['alpha'].append(result['alpha'])
        save_dict['beta'].append(result['beta'])
        save_dict['信息比率'].append(result['info_ratio'])

    df = pd.DataFrame[save_dict]
    df.to_csv("single_factor_test.csv", sep=',')
    """


"""
----------------------------------------------------------
策略思路：
1. 回测标的：沪深300成分股
2. 回测时间段：2016-01-01 至 2018-09-30
3. 特征选择：每个大类夏普率最高的因子+夏普率高于1.5的因子
    - 质量类：ROIC, CashToCurrentLiability
    - 特色技术指标：STDDEV
    - 收益风险：DDNCR
    - 情绪类：TVMA20
    - 每股指标类：EnterpriseFCFPS
    - 价值类：PS
    - 基础类：AdminExpenseTTM, FinanExpenseTTM, NetIntExpense, GrossProfit
    - 行业分析师：FY12P
    - 动量类：TotalAssetGrowRate
    - 成长类：TotalAssetGrowRate
    - 常用技术类：MA120
... 其余逻辑参照single_factor_test.py

----------------------------------------------------------
"""
from atrader import *
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn import preprocessing
import datetime

# 作为全局变量进行测试
FactorCode = ['ROIC', 'CashToCurrentLiability', 'STDDEV', 'DDNCR', 'PVI', 'EnterpriseFCFPS',
              'PS', 'AdminExpenseTTM', 'FinanExpenseTTM', 'NetIntExpense', 'NIAP', 'FY12P',
              'AD', 'TotalAssetGrowRate', 'MA120']

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

    # context.SVM = svm.SVC(gamma='scale')
    # 账号设置：设置初始资金为 10000000 元
    set_backtest(initial_cash=10000000, future_cost_fee=1.0, stock_cost_fee=30, margin_rate=1.0, slide_price=0.0,
                 price_loc=1, deal_type=0, limit_type=0)
    # 注册数据：日频数据
    reg_kdata('day', 1)
    global FactorCode  # 全局单因子代号
    reg_factor(factor=FactorCode)

    context.FactorCode = FactorCode  #

    # 超参数设置：
    context.Len = 21    # 时间长度: 当交易日个数小于该事件长度时，跳过该交易日，假设平均每个月 21 个交易日左右  250/12
    context.Num = 0   # 记录当前交易日个数

    # 较敏感的超参数，需要调节
    context.upper_pos = 75  # 股票预测收益率的上分位数，高于则买入
    context.down_pos = 10   # 股票预测收益率的下分位数，低于则卖出
    context.cash_rate = 0.6  # 计算可用资金比例的分子，利益大于0的股票越多，比例越小

    # 确保月初调仓
    days = get_trading_days('SSE', '2016-01-01', '2018-09-30')
    months = np.vectorize(lambda x: x.month)(days)
    month_begin = days[pd.Series(months) != pd.Series(months).shift(1)]
    context.month_begin = pd.Series(month_begin).dt.strftime('%Y-%m-%d').tolist()

    # 三均线择时策略
    # 无持仓的情况下，5日和20日均线都大于60日均线，买入，等价于5日和20日均线上穿60日均线，买入；
    # 有持仓的情况下，5日和20日均线都小于60日均线，卖出，等价于5日和20日均线上穿60日均线，买入；
    context.win = 61  # 计算所需总数据长度
    context.win5 = 5  # 5日均线参数
    context.win20 = 20  # 20日均线参数
    context.win60 = 60  # 60日均线参数

def on_data(context):
    context.Num = context.Num + 1
    if context.Num < context.win:  # 如果交易日个数小于win，则进入下一个交易日进行回测
        return
    if datetime.datetime.strftime(context.now, '%Y-%m-%d') not in context.month_begin:  # 调仓频率为月,月初开始调仓
        return

    # 获取数据：
    KData = get_reg_kdata(reg_idx=context.reg_kdata[0], length=context.Len, fill_up=True, df=True)
    FData = get_reg_factor(reg_idx=context.reg_factor[0], target_indices=[x for x in range(300)], length=context.Len,
                           df=True)  # 获取因子数据

    # 特征构建：
    Fcode = context.FactorCode  # 标签不需要代号了

    # 数据存储变量：
    # Close 字段为标签，Fcode 为标签
    FactorData = pd.DataFrame(columns=(['idx', 'benefit'] + Fcode))  # 存储训练特征及标签样本
    FactorDataTest = pd.DataFrame(columns=(['idx'] + Fcode))       # 存储预测特征样本

    # K线数据序号对齐
    tempIdx = KData[KData['time'] == KData['time'][0]]['target_idx'].reset_index(drop=True)

    # 按标的处理数据：
    for i in range(300):
        # 训练特征集及训练标签构建：
        # 临时数据存储变量:
        FactorData0 = pd.DataFrame(np.full([1, len(Fcode) + 2], np.nan),
            columns=(['idx', 'benefit'] + Fcode))
        # 存储预测特征样本
        FactorDataTest0 = pd.DataFrame(np.full([1, len(Fcode) + 1], np.nan), columns=(['idx'] + Fcode))

        # 因子数据 序号对齐, 提取当前标的的因子数据
        FData0 = FData[FData['target_idx'] == tempIdx[i]].reset_index(drop=True)

        # 按特征处理数据：
        for FC in context.FactorCode:
            # 提取当前标的中与当前因子FC相同的部分
            FCData = FData0[FData0['factor'] == FC]['value'].reset_index(drop=True)
            FactorData0[FC] = FCData[0]  # 存储上一个月初的股票因子数据

        # 按标签处理数据：
        # 提取当前标的的前一个月的K线面板数据
        close = np.array(KData[KData['target_idx'] == tempIdx[i]]['close'])
        # 计算当前标的在上一个月的收益率
        benefit = (close[context.Len - 1] - close[0]) / close[0]

        FactorData0['benefit'] = benefit
        # idx: 建立当前标的在训练样本集中的索引
        FactorData0['idx'] = tempIdx[i]
        # 合并数据：组成训练样本
        FactorData = FactorData.append(FactorData0, ignore_index=True)

        # 预测特征集构建：建立标的索引
        FactorDataTest0['idx'] = tempIdx[i]
        # 按特征处理数据，过程同建立训练特征
        for FC in context.FactorCode:
            FCData = FData0[FData0['factor'] == FC]['value'].reset_index(drop=True)
            FactorDataTest0[FC] = FCData[context.Len - 1]

        # 合并测试数据
        FactorDataTest = FactorDataTest.append(FactorDataTest0, ignore_index=True)

    """
    训练集和测试集的表头字段如下
    FactorData DataFrame:
    idx  |  benefit |  Factor 1 | Factor 2| ....
    benefit 作为标签，上月初Factor作为特征，此处是单因子测试，只有一个特征
    FactorDataTest DataFrame: 
    idx | Factor 1 | Factor 2 | ...
    本月初的因子作为预测特征
    """

    # 数据清洗：
    FactorData = FactorData.dropna(axis=0, how='any').reset_index(drop=True)  # 清洗数据
    FactorDataTest = FactorDataTest.dropna(axis=0, how='any').reset_index(drop=True)  # 清洗数据
    Idx = FactorDataTest['idx']  # 剩余标的序号

    # 按特征进行预处理
    for Factor in context.FactorCode:
        FactorData = filter_MAD(FactorData, Factor, 5)  # 中位数去极值法
        FactorData[Factor] = preprocessing.scale(FactorData[Factor])  # 标准化

        FactorDataTest = filter_MAD(FactorDataTest, Factor, 5)  # 中位数去极值法
        FactorDataTest[Factor] = preprocessing.scale(FactorDataTest[Factor])  # 标准化

    # 训练和预测特征构建：# 行（样本数）* 列（特征数）
    X = np.ones([FactorData.shape[0], len(Fcode)])
    Xtest = np.ones([FactorDataTest.shape[0], len(Fcode)])

    # 循环填充特征到numpy数组中
    for i in range(X.shape[1]):
        X[:, i] = FactorData[Fcode[i]]
        Xtest[:, i] = FactorDataTest[Fcode[i]]

    # 训练样本的标签，为浮点数的收益率
    Y = np.array(FactorData['benefit']).astype(float)

    random_forest = RandomForestRegressor(max_depth=5, n_estimators=50)

    # 模型训练：
    random_forest.fit(X, Y)

    # LR分类预测：
    y = random_forest.predict(Xtest)
    # 交易设置：
    positions = context.account().positions['volume_long']  # 多头持仓数量
    valid_cash = context.account(account_idx=0).cash['valid_cash'][0]  # 可用资金

    P = context.cash_rate / (sum(y > 0) + 1)  # 设置每只标的可用资金比例 + 1 防止分母为0

    # 获取收益率的高分位数和低分位数
    low_return, high_return = np.percentile(y, [context.down_pos, context.upper_pos])

    # 进行择时准备
    # 获取前61天的数据
    data = get_reg_kdata(reg_idx=context.reg_kdata[0], length=context.win, fill_up=True,
                         df=True)  # data值为数据帧DataFrame类型，存储所有标的的K线行情数据。
    # 获取收盘价数据
    close = data.close.values.reshape(-1, context.win).astype(float)  # 从data行情数据中获取收盘价，并转为ndarray数据类型
    # 计算均线值：
    ma5 = close[:, -context.win5:].mean(axis=1)    # 5日均线
    ma20 = close[:, -context.win20:].mean(axis=1)  # 20日均线
    ma60 = close[:, -context.win60:].mean(axis=1)  # 60日均线

    # 获取标的序号：从0~299
    target = np.array(range(300))
    positions_val = context.account().positions['volume_long'].values  # 多头持仓数量
    # 计算买入信号：
    buy_signal = np.logical_and(positions_val == 0, ma5 > ma60,
                                ma20 > ma60)  # 无持仓的情况下，5日和20日均线都大于60日均线，买入，等价于5日和20日均线上穿60日均线，买入；
    # 计算卖出信号：
    sell_signal = np.logical_and(positions_val > 0, ma5 < ma60,
                                 ma20 < ma60)  # 有持仓的情况下，5日和20日均线都小于60日均线，卖出，等价于5日和20日均线上穿60日均线，买入；
    # 获取买入信号标的的序号
    target_buy = target[buy_signal].tolist()  # 一个记录了标的是否要买
    # 获取卖出信号标的的序号
    target_sell = target[sell_signal].tolist() # 同上
    for i in range(len(Idx)):
        position = positions.iloc[Idx[i]]

        # 当前无仓，且该股票收益大于高80%分位数，且5日和20日均线都大于或等于60日均线 则开仓，买入
        if position == 0 and y[i] > high_return and valid_cash > 0 and Idx[i] in target_buy:
            Num = int(math.floor(valid_cash * P / 100 / (KData['close'][Idx[i]] + 1)) * 100)
            # 控制委托量，不要过大或过小,需要保证是100的倍数
            if Num < 1000:
                Num *= 10
            if Num > 100000:
                Num = int(Num / 10)
                Num -= Num % 100
            if Num <= 0:  # 不开仓
                continue
            print("开仓数量为：{}".format(Num))
            order_volume(account_idx=0, target_idx=int(Idx[i]), volume=Num, side=1, position_effect=1, order_type=2,
                         price=0)  # 指定委托量开仓

        # 当前持仓，且该股票收益小于低20%分位数，5日和20日均线都小于60日均线 则平仓，卖出
        elif position > 0 and y[i] < low_return and Idx[i] in target_sell:
            print("平仓，数量为: {}".format(position / 10))
            order_volume(account_idx=0, target_idx=int(Idx[i]), volume=int(position),
                         side=2, position_effect=2, order_type=2, price=0)  # 指定委托量平仓


if __name__ == '__main__':
    file_path = 'RF_line3.py'
    block = 'hs300'

    begin_date = '2016-01-01'
    end_date = '2018-09-30'

    strategy_name = 'RF_line3'

    run_backtest(strategy_name=strategy_name, file_path=file_path,
                 target_list=list(get_code_list('hs300', date=begin_date)['code']),
                 frequency='day', fre_num=1, begin_date=begin_date, end_date=end_date, fq=1)

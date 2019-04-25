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
from sklearn import svm
import math
from sklearn import preprocessing
import datetime
from sklearn.neural_network import MLPRegressor

# 作为全局变量进行测试

FactorCode = ['ROIC', 'CashToCurrentLiability', 'STDDEV', 'DDNCR', 'TVMA20', 'EnterpriseFCFPS',
              'PS', 'AdminExpenseTTM', 'FinanExpenseTTM', 'NetIntExpense', 'GrossProfit', 'FY12P',
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
    context.upper_pos = 85  # 股票预测收益率的上分位数，高于则买入
    context.down_pos = 60   # 股票预测收益率的下分位数，低于则卖出
    context.cash_rate = 0.7  # 计算可用资金比例的分子，利益大于0的股票越多，比例越小

    # 确保月初调仓
    days = get_trading_days('SSE', '2016-01-01', '2018-09-30')
    months = np.vectorize(lambda x: x.month)(days)
    month_begin = days[pd.Series(months) != pd.Series(months).shift(1)]
    context.month_begin = pd.Series(month_begin).dt.strftime('%Y-%m-%d').tolist()


def on_data(context):
    context.Num = context.Num + 1
    if context.Num < context.Len:  # 如果交易日个数小于Len+1，则进入下一个交易日进行回测
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
    Y = (np.array(FactorData['benefit']).astype(float) > 0)

    mlp = MLPRegressor(hidden_layer_sizes=6, activation='logistic', solver='adam',
                        max_iter=50)

    # 模型训练：
    mlp.fit(X, Y)

    # LR分类预测：
    y = mlp.predict(Xtest)

    # 交易设置：
    positions = context.account().positions['volume_long']  # 多头持仓数量
    valid_cash = context.account(account_idx=0).cash['valid_cash'][0]  # 可用资金

    P = context.cash_rate / (sum(y > 0) + 1)  # 设置每只标的可用资金比例 + 1 防止分母为0

    # 获取收益率的高分位数和低分位数
    high_return, low_return = np.percentile(y, [context.down_pos, context.upper_pos])

    for i in range(len(Idx)):
        position = positions.iloc[Idx[i]]
        # if position == 0 and y[i] == True and valid_cash > 0:  # 若预测结果为true(收益率>0)，买入
            # print('开仓')
        if position == 0 and y[i] > high_return and valid_cash > 0: # 当前无仓，且该股票收益大于高70%分位数，则开仓，买入
            # 开仓数量 + 1防止分母为0
            # print(valid_cash, P, KData['close'][Idx[i]])  # 这里的数目可考虑减少一点，，有时太多有时太少
            Num = int(math.floor(valid_cash * P / 100 / (KData['close'][Idx[i] * 21 + 20] + 1)) * 100)

            # 控制委托量，不要过大或过小,需要保证是100的倍数
            if Num < 1000:
                Num *= 10
            if Num > 100000:
                Num = int(Num / 10)
                Num -= Num % 100
            if Num <= 0:  # 不开仓
                continue

            print("开仓数量为：{}".format(Num))
            order_id = order_volume(account_idx=0, target_idx=int(Idx[i]), volume=Num, side=1, position_effect=1, order_type=2,
                         price=0)  # 指定委托量开仓
            # 对订单号为order_id的委托单设置止损，止损距离10个整数点，触发时，委托的方式用市价委托
            # stop_loss_by_order(target_order_id=order_id, stop_type=1, stop_gap=10, order_type=2)
        # elif position > 0 and y[i] == False: #预测结果为false(收益率<0)，卖出
        elif position > 0 and y[i] < low_return:  # 当前持仓，且该股票收益小于低30%分位数，则平仓，卖出
            # print("平仓")
            order_volume(account_idx=0, target_idx=int(Idx[i]), volume=int(position), side=2, position_effect=2,
                         order_type=2, price=0)  # 指定委托量平仓


if __name__ == '__main__':

    file_path = 'MLP.py'
    block = 'hs300'

    begin_date = '2016-01-01'
    end_date = '2018-09-30'

    strategy_name = 'MLP'

    run_backtest(strategy_name=strategy_name, file_path=file_path,
                 target_list=list(get_code_list('hs300', date=begin_date)['code']),
                 frequency='day', fre_num=1, begin_date=begin_date, end_date=end_date, fq=1)

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
    - 价值类：PS TA2EV
    - 基础类：AdminExpenseTTM, FinanExpenseTTM, NetIntExpense
    - 行业分析师：FY12P
    - 成长类：TotalAssetGrowRate
    - 常用技术类：MA120
... 其余逻辑参照single_factor_test.py
----------------------------------------------------------

时间窗口滚动模型：
在原来的基础上增加了滚动选项。
原来的时间窗口固定为一个，即前20天为一个时间窗口。
现在支持时间窗口向前滚动获取数据，有：时间窗口第一天的因子值，时间窗口内各股票的平均收益率；

"""
from atrader import *
import pandas as pd
import numpy as np
from sklearn import svm
import math
from sklearn import preprocessing
import datetime
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

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
    FactorCode = ['ROIC', 'CashToCurrentLiability', 'STDDEV', 'DDNCR', 'PVI', 'EnterpriseFCFPS',
                  'PS', 'AdminExpenseTTM', 'FinanExpenseTTM', 'NetIntExpense', 'NIAP', 'FY12P',
                  'AD', 'TotalAssetGrowRate', 'MA120']
    reg_factor(factor=FactorCode)
    context.FactorCode = FactorCode

    # 参数设置：
    context.LEN = 21   # 时间窗口滑动最大范围
    context.N1 = 20    # 时间窗口中的训练/预测特征部分
    context.Num = 0    # 记录当前交易日个数，保证交易日个数需要大于时间窗口滑动的最大范围

    # 较敏感的超参数，需要调节
    context.upper_pos = 80   # 股票预测收益率的上分位数，高于则买入
    context.down_pos = 60    # 股票预测收益率的下分位数，低于则卖出
    context.cash_rate = 0.6  # 计算可用资金比例的分子，

    # 确保月初调仓
    days = get_trading_days('SZSE', '2016-01-01', '2018-09-30')
    months = np.vectorize(lambda x: x.month)(days)
    month_begin = days[pd.Series(months) != pd.Series(months).shift(1)]
    context.month_begin = pd.Series(month_begin).dt.strftime('%Y-%m-%d').tolist()


def on_data(context):
    context.Num = context.Num + 1  # 交易日数目+1
    if context.Num < context.LEN:  # 如果交易日个数小于Len+1，则进入下一个交易日进行回测
        return
    if datetime.datetime.strftime(context.now, '%Y-%m-%d') not in context.month_begin:  # 调仓频率为月,月初开始调仓
        return

    # -------------------------------------------- #
    #  获取 K线数据和因子数据                      #
    # -------------------------------------------- #
    """
    K 线数据 DataFrame结构：
    |  target_idx | time | open   | high   | low   |  close | volume | amount   | open_interest
    | 标的索引号  | 日期 | 开盘价 | 最高价 |最低价 | 收盘价 | 成交量 | 成交金额 | 持仓量
    如果获取了 LEN 天的各股票对应的K线数据，那么行排列是：
    ０ 至 LEN - 1 行先排第一个股票在LEN天内K线数据，
    然后 LEN 至 2 LEN - 1行排第二个股票在LEN天内的K线数据。
    """
    KData = get_reg_kdata(reg_idx=context.reg_kdata[0], length=context.LEN, fill_up=True, df=True)

    """
    因子数据 DataFrame结构：
    | target_idx | date | factor   | value | 
    | 标的序号   | 日期 | 因子名称 | 因子值|
    行排列情况：先排一个股票在LEN天内的某一因子值，然后在排该股票下一个因子值，直到因子值排完，
    然后再轮到下一个股票
    """
    FData = get_reg_factor(reg_idx=context.reg_factor[0], target_indices=[x for x in range(300)], length=context.LEN,
                           df=True)  # 获取因子数据

    # ------------------------------------- #
    #  特征构建                             #
    # ------------------------------------- #
    Fcode = list()
    # 此处构建因子列名，取时间窗的第一天因子作为训练/预测数据样本
    Fcode = context.FactorCode

    FactorData_list = []  # 存储多个时间窗口的训练样本和标签
    """
    用于训练的DataFrame，每一列的含义如下：
    idx  | benefit |  factor1  | factor1 | .... | factorm
    idx 表示沪深300股中股票的序号，范围从 0~299，我们可以通过该序号定位股票
    benefit 表示该股票在某时间窗口后 N2 天内的平均收益率，即涨幅情况
    factorm_n 表示在时间窗口内的第一天的第 m 个因子
    我们使用所有的factor作为训练特征，benefit作为训练标签。
    """
    for i in range(context.LEN - context.N1 + 1):  # 时间窗口个数
        FactorData = pd.DataFrame(columns=(['idx', 'benefit'] + Fcode))  # 存储训练特征及标签样本
        FactorData_list.append(FactorData)   # 将该时间窗的训练数据存入列表

    """
     用于预测的DataFrame，结构如下：
     idx  | factor1_1  | factor2_1 | .... | factorm_n
     idx 表示沪深300股中股票的序号，范围从 0~299，我们可以通过该序号定位股票
     factorm_n 表示第 m 个因子在第 n 天的值
     我们使用所有的factor作为预测特征，预测出未来 N2天的各股票的收益率情况
    """
    FactorDataTest = pd.DataFrame(columns=(['idx'] + Fcode))  # 存储预测特征样本

    # K线数据序号对齐
    tempIdx = KData[KData['time'] == KData['time'][0]]['target_idx'].reset_index(drop=True)

    # ----------------------------------------- #
    #  按标的处理数据，提取训练特征和标签       #
    # ----------------------------------------- #
    for window in range(context.LEN - context.N1 + 1):  # 滚动时间窗
        for i in range(300):  # 按标的处理
            # 训练特征集及训练标签构建：
            FactorData0 = pd.DataFrame(np.full([1, len(Fcode) + 2], np.nan), columns=(['idx', 'benefit'] + Fcode))

            # 因子数据 序号对齐, 提取当前标的的因子数据
            FData0 = FData[FData['target_idx'] == tempIdx[i]].reset_index(drop=True)

            # 按特征处理数据：
            for FC in context.FactorCode:
                # 提取当前标的中与当前因子FC相同的部分
                FCData = FData0[FData0['factor'] == FC]['value'].reset_index(drop=True)
                FactorData0[FC] = FCData[window]

            FactorData0['idx'] = i

            # 按标签处理数据：
            # 提取当前标的的前一个月的K线面板数据
            close = np.array(KData[KData['target_idx'] == tempIdx[i]]['close'])

            # 当前时间窗之后的N2天内的股票收益率情况
            benefit = (close[window + context.N1 - 1] - close[window]) / close[window]

            FactorData0['benefit'] = benefit
            FactorData_list[window] = FactorData_list[window].append(FactorData0, ignore_index=True)
            print("window:{}, stock :{} ".format(window, i))
        print("pass this window: {}".format(window))
    # ----------------------------------- #
    # 提取预测样本特征                    #
    # ----------------------------------- #
    for i in range(300):
        # 存储预测特征样本
        FactorDataTest0 = pd.DataFrame(np.full([1, len(Fcode) + 1], np.nan), columns=(['idx'] + Fcode))

        # 因子数据 序号对齐, 提取当前标的的因子数据
        FData0 = FData[FData['target_idx'] == tempIdx[i]].reset_index(drop=True)

        # 预测特征集构建：建立标的索引
        FactorDataTest0['idx'] = tempIdx[i]

        # 按特征处理数据，过程同建立训练特征
        for FC in context.FactorCode:
            FCData = FData0[FData0['factor'] == FC]['value'].reset_index(drop=True)
            FactorDataTest0[FC] = FCData[context.LEN - 1]

        # 合并测试数据
        FactorDataTest = FactorDataTest.append(FactorDataTest0, ignore_index=True)

    # 数据清洗：
    for i in range(len(FactorData_list)):
        FactorData_list[i] = FactorData_list[i].dropna(axis=0, how='any').reset_index(drop=True)  # 清洗数据
    FactorDataTest = FactorDataTest.dropna(axis=0, how='any').reset_index(drop=True)  # 清洗数据
    Idx = FactorDataTest['idx']  # 剩余标的序号

    # 按特征进行预处理
    for Factor in Fcode:
        # 处理多个时间窗口的训练数据。
        for window in range(len(FactorData_list)):
            FactorData_list[window] = filter_MAD(FactorData_list[window], Factor, 5)  # 中位数去极值法
            FactorData_list[window][Factor] = preprocessing.scale(FactorData_list[window][Factor])  # 标准化

        FactorDataTest = filter_MAD(FactorDataTest, Factor, 5)  # 中位数去极值法
        FactorDataTest[Factor] = preprocessing.scale(FactorDataTest[Factor])  # 标准化

    """
    xgb_params = {'learning_rate': 0.01, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 4, 'seed': 1000,
                  'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1}

    xgb_model = XGBRegressor(**xgb_params)    
    """
    RF = RandomForestRegressor(max_depth=5, n_estimators=50)

    # 训练和预测特征构建：# 行（样本数）* 列（特征数）
    for window in range(len(FactorData_list)):
        X = np.ones([FactorData_list[window].shape[0], len(Fcode)])

        # 循环填充特征到numpy数组中
        for i in range(X.shape[1]):
            X[:, i] = FactorData_list[window][Fcode[i]]

        # 训练样本的标签，为浮点数的收益率
        Y = (np.array(FactorData_list[window]['benefit']).astype(float) > 0)

        # 模型训练：
        print("FITTING!")
        RF.fit(X, Y)

    Xtest = np.ones([FactorDataTest.shape[0], len(Fcode)])
    for i in range(X.shape[1]):
        Xtest[:, i] = FactorDataTest[Fcode[i]]

    # 分类预测：
    y = RF.predict(Xtest)

    # 交易设置：
    positions = context.account().positions['volume_long']  # 多头持仓数量
    valid_cash = context.account(account_idx=0).cash['valid_cash'][0]  # 可用资金

    P = context.cash_rate / (sum(y > 0) + 1)  # 设置每只标的可用资金比例 + 1 防止分母为0

    # 获取收益率的高分位数和低分位数
    high_return, low_return = np.percentile(y, [context.down_pos, context.upper_pos])

    for i in range(len(Idx)):
        position = positions.iloc[Idx[i]]
        if position == 0 and y[i] > high_return and valid_cash > 0 and y[i] > 0: # 当前无仓，且该股票收益大于高70%分位数，则开仓，买入
            # 开仓数量 + 1防止分母为0
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
            order_id = order_volume(account_idx=0, target_idx=int(Idx[i]), volume=Num, side=1, position_effect=1,
                                    order_type=2, price=0)  # 指定委托量开仓
            # 对订单号为order_id的委托单设置止损，止损距离10个整数点，触发时，委托的方式用市价委托
            stop_loss_by_order(target_order_id=order_id, stop_type=1, stop_gap=10, order_type=2)
        # elif position > 0 and y[i] == False: #预测结果为false(收益率<0)，卖出
        elif position > 0 and y[i] < low_return:  # 当前持仓，且该股票收益小于低30%分位数，则平仓，卖出
            # print("平仓")
            order_volume(account_idx=0, target_idx=int(Idx[i]), volume=int(position), side=2, position_effect=2,
                         order_type=2, price=0)  # 指定委托量平仓


if __name__ == '__main__':

    file_path = 'time_roll_model.py'
    block = 'hs300'

    begin_date = '2016-01-01'
    end_date = '2018-09-30'

    strategy_name = 'random_forest'

    run_backtest(strategy_name=strategy_name, file_path=file_path,
                 target_list=list(get_code_list('hs300', date=begin_date)['code']),
                 frequency='day', fre_num=1, begin_date=begin_date, end_date=end_date, fq=1)

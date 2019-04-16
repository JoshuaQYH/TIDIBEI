"""
-------------------------------------------------------
策略思路：
1. 回测标的：沪深300成分股
2. 回测时间段：2016-01-01 至 2018-09-30
3. 特征选择：待测单因子
4. 单因子回归测试模型思路：
    1. 先获得 30 天以上的K线数据和因子数据，预处理
    2. 使用月末因子和本月收益率进行线性回归
    3. 使用单变量线性模型进行训练
    4. 回到当前时间点，使用前 1 天的因子数据作为预测样本特征，预测后 30 天的各股票平均收益率的大小。
5. 选股逻辑：
    将符合预测结果的股票按均等分配可用资金进行下单交易。持有 一个月后 ，再次进行调仓，训练预测。
6. 交易逻辑：
    每次调仓时，若当前有持仓，并且符合选股条件，则仓位不动；
                              若不符合选股条件，则对收益最低的25%标的进行仓位平仓；
                若当前无仓，并且符合选股条件，则多开仓，对收益最高的25%标的进行开仓；
                            若不符合选股条件，则不开仓，无需操作。

---------------------------------------------------------
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
import datetime
import sys

# 作为全局变量进行测试
factor = sys.argv[1]
FactorCode = [factor]
print("传入因子参数为" + factor)


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
    set_backtest(initial_cash=1000000, future_cost_fee=1.0, stock_cost_fee=30, margin_rate=1.0, slide_price=0.0,
                 price_loc=1, deal_type=0, limit_type=0)
    # 注册数据：日频数据
    reg_kdata('day', 1)
    global FactorCode  # 全局单因子代号
    reg_factor(factor=FactorCode)
    print("init 函数, 注册因子为{}".format(FactorCode[0]))
    context.FactorCode = FactorCode  #

    # 超参数设置：
    context.Len = 50  # 时间长度: 当交易日个数小于该事件长度时，跳过该交易日
    # 我们的目的就是使用前30天的因子数据作为样本，未来10天的股票收益率作为标签，进行回归。
    # 在前50天，可以取多组样本和标签，构成回归测试的数据
    context.N1 = 1    # 训练样本的时间跨度，代表训练的天数  // 过高过低都不太好，过高可能有噪声，过低数据特征不明显
    context.N2 = 30   # 标签的时间跨度,代表预测未来的天数  //
    context.Num = 0   # 记录当前交易日个数

    context.upper_pos = 75  # 股票预测收益率的上分位数，高于则买入
    context.down_pos = 25   # 股票预测收益率的下分位数，低于则卖出

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

        # FData0.dropna(axis=0)  # 删除因子缺失的股票
        FData0 = filter_MAD(FData0, "value", 3)  # 中位数去极值法
        # FData0["value"] = preprocessing.scale(FData0["value"])  # 标准化

        # 按特征处理数据：
        for FC in context.FactorCode:
            FCData = FData0[FData0['factor'] == FC]['value'].reset_index(drop=True)
            for k in range(context.N1):
                FactorData0[FC + str(k)] = FCData[k:k + context.Len - context.N1 - context.N2]

        # 按标签处理数据：
        close = np.array(KData[KData['target_idx'] == tempIdx[i]]['close'])
        # 将N2内的天收益率作为标签
        signValue = np.sign((close[context.N1 + context.N2:] -
                             close[context.N1:context.Len - context.N2]) / close[context.N1:context.Len - context.N2])

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

    # 预测特征构建：
    X = np.ones([FactorData.shape[0], len(Fcode)])
    Xtest = np.ones([FactorDataTest.shape[0], len(Fcode)])
    for i in range(X.shape[1]):
        X[:, i] = FactorData[Fcode[i]]
        Xtest[:, i] = FactorDataTest[Fcode[i]]

    Y = np.array(FactorData['signClose']).astype(int)

    # 构建模型： 这一步放在了 init里头，模型成为全局变量
    LRModel = LinearRegression()

    # 模型训练：
    LRModel.fit(X, Y)

    # LR分类预测：
    y = LRModel.predict(Xtest)

    # 交易设置：
    positions = context.account().positions['volume_long']  # 多头持仓数量
    valid_cash = context.account(account_idx=0).cash['valid_cash'][0]  # 可用资金

    P = 0.6 / (sum(y > 0) + 1)  # 设置每只标的可用资金比例 + 1 防止分母为0

    # 获取收益率的高四分位数和低四分位数
    high_return, low_return = np.percentile(y, [30, 70])

    for i in range(len(Idx)):
        position = positions.iloc[Idx[i]]
        if position == 0 and y[i] > high_return and valid_cash > 0: # 当前无仓，且该股票收益大于高70%分位数，则开仓，买入
            # 开仓数量 + 100防止开仓数量为0  + 1防止分母为0
            print(valid_cash, P, KData['close'][Idx[i]])  # 这里的数目可考虑减少一点，，有时太多有时太少
            Num = int(math.floor(valid_cash * P / 100 / (KData['close'][Idx[i]] + 1)) * 100) + 100

            # 控制委托量区间
            if Num < 1000:
                Num *= 10
            if Num > 100000:
                Num = int(Num / 10)

            print("开仓数量为：{}".format(Num))
            order_volume(account_idx=0, target_idx=int(Idx[i]), volume=Num, side=1, position_effect=1, order_type=2,
                         price=0)  # 指定委托量开仓

        elif position > 0 and y[i] < low_return:  # 当前持仓，且该股票收益小于低30%分位数，则平仓，卖出
            #print("平仓")
            order_volume(account_idx=0, target_idx=int(Idx[i]), volume=int(position), side=2, position_effect=2,
                         order_type=2, price=0)  # 指定委托量平仓


if __name__ == '__main__':

    file_path = 'single_factor_test.py'
    block = 'hs300'

    begin_date = '2016-01-01'
    end_date = '2018-09-30'

    strategy_name = factor

    run_backtest(strategy_name=strategy_name, file_path=file_path,
                 target_list=list(get_code_list('hs300', date=begin_date)['code']),
                 frequency='day', fre_num=1, begin_date=begin_date, end_date=end_date, fq=1)


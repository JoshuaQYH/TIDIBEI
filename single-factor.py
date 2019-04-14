"""

-------------------------------------------------------
策略思路：
1. 回测标的：沪深300成分股
2. 回测时间段：2016-01-01 至 2016-12-31
3. 特征选择：待测单因子
4. 单因子回归测试模型思路：
    1. 先获得100天以上的 K线数据和因子数据；因子数据必须进行预处理
    2. 以30天为一次训练单位，其中前25天的因子作为训练样本特征，后5天的各股票平均收益率大小作为标签：> 0 上涨  = 0 不变 < 0 下跌
    3. 使用单变量线性模型进行训练。
    4. 回到当前时间点，使用前25天的因子数据作为预测样本特征，预测后5天的各股票平均收益率的大小。
5. 选股逻辑：
    将收益率大于0且排名前10的股票进行下单交易。持有5天之后，再次进行训练预测。
6. 交易逻辑：
    每次调仓时，若当前有持仓，并且符合选股条件，则仓位不动；
                              若不符合选股条件，则所有仓位平仓；
                若当前无仓，并且符合选股条件，则多开仓；
                            若不符合选股条件，则不开仓，无需操作。

"""

from atrader import *  # 导入atrader工具包
import pandas as pd  # 导入pandas工具包
import numpy as np  # 导入numpy工具包
from sklearn.linear_model import LinearRegression  # 导入LogisticRegression工具包
import math  # 导入math工具包

# 作为全局变量进行测试
FactorCode = ["PB"]

def init(context):
    # 账号设置：设置初始资金为 1000000 元
    set_backtest(initial_cash=1000000)
    # 注册数据：日频数据
    reg_kdata('day', 1)
    global FactorCode  # 全局单因子代号
    reg_factor(factor=FactorCode)
    print(context.reg_factor.describe())
    context.FactorCode = FactorCode
    # 参数设置：
    context.Len = 100  # 时间长度: 当交易日个数小于该事件长度时，跳过该交易日
    context.N1 = 25  #
    context.N2 = 5
    context.Num = 0  # 记录当前交易日个数


def on_data(context):
    context.Num = context.Num + 1
    if context.Num < context.Len:  # 如果交易日个数小于Len+1，则进入下一个交易日进行回测
        return
    if bool(context.Num % context.N2):  # 调仓
        return

    # 获取数据：
    KData = get_reg_kdata(reg_idx=context.reg_kdata[0], length=context.Len, fill_up=True, df=True)
    FData = get_reg_factor(reg_idx=context.reg_factor[0], target_indices=[x for x in range(300)], length=context.Len,
                           df=True)  # 获取因子数据

    # 数据预处理：
    # 数据整理：
    # 特征构建：
    Fcode = list()
    for i in range(context.N1):
        for FC in context.FactorCode:
            Fcode.append(FC + str(i))  # F

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
            columns=(['idx', 'signClose'] + Fcode))  # np.full([context.Len-context.N1,context.N1],np.nan)
        FactorDataTest0 = pd.DataFrame(np.full([1, len(Fcode) + 1], np.nan), columns=(['idx'] + Fcode))  # 存储预测特征样本

        FData0 = FData[FData['target_idx'] == tempIdx[i]].reset_index(drop=True)
        # 按特征处理数据：
        for FC in context.FactorCode:
            FCData = FData0[FData0['factor'] == FC]['value'].reset_index(drop=True)
            for k in range(context.N1):
                FactorData0[FC + str(k)] = FCData[k:k + context.Len - context.N1 - context.N2]

        # 按标签处理数据：
        close = np.array(KData[KData['target_idx'] == tempIdx[i]]['close'])
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
        if position == 0 and y[i] > 0:
            Num = int(math.floor(valid_cash * P / 100 / KData['close'][Idx[i]]) * 100)  # 开仓数量
            order_volume(account_idx=0, target_idx=int(Idx[i]), volume=Num, side=1, position_effect=1, order_type=2,
                         price=0)  # 指定委托量开仓
        elif position > 0 and y[i] < 0:
            order_volume(account_idx=0, target_idx=int(Idx[i]), volume=int(position), side=2, position_effect=2,
                         order_type=2, price=0)  # 指定委托量平仓


if __name__ == '__main__':
    factor_list = ["PE"]
    file_path = 'single-factor.py'
    block = 'hs300'
    begin_date = '2016-01-01'
    end_date = '2016-12-31'
    for factor in factor_list:
        startegy_name = "Test-" + factor
        FactorCode = [factor]  # 修改全局变量
        print(FactorCode)
        run_backtest(strategy_name=startegy_name, file_path=file_path,
                    target_list=list(get_code_list('hs300', date=begin_date)['code']),
                    frequency='day', fre_num=1, begin_date=begin_date, end_date=end_date, fq=1)

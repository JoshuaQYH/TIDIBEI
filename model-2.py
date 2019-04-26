from atrader import *
import numpy as np


# 初始化：注册数据和设置参数
def init(context):
    set_backtest(initial_cash=1e7)  # 初始化设置账户总资金
    reg_kdata('day', 1)  # 注册日频行情数据
    # 设置参数：
    context.win = 61  # 计算所需总数据长度
    context.win5 = 5  # 5日均线参数
    context.win20 = 20  # 20日均线参数
    context.win60 = 60  # 60日均线参数
    context.Tlen = len(context.target_list)  # 标的总数


# 策略实现函数：
def on_data(context):
    # 获取仓位数据：positions=0，表示无持仓
    positions = context.account().positions['volume_long'].values
    # 获取当前时刻交易日的前context.win个交易日长度的行情数据：
    data = get_reg_kdata(reg_idx=context.reg_kdata[0], length=context.win, fill_up=True, df=True)  # data值为数据帧DataFrame类型，存储所有标的的K线行情数据。
    # 判断获取的行情数据是否存在nan值，若存在，因无法计算均线值而停止当前交易日的交易
    if data['close'].isna().any():
        return
    # 获取收盘价数据
    close = data.close.values.reshape(-1, context.win).astype(float) # 从data行情数据中获取收盘价，并转为ndarray数据类型
    # 计算均线值：
    ma5 = close[:, -context.win5:].mean(axis=1) # 5日均线
    ma20 = close[:, -context.win20:].mean(axis=1)  # 20日均线
    ma60 = close[:, -context.win60:].mean(axis=1)  # 60日均线
    # 获取标的序号：
    target = np.array(range(context.Tlen))
    # 计算买入信号：
    buy_signal=np.logical_and(positions == 0, ma5 > ma60, ma20 > ma60 ) # 无持仓的情况下，5日和20日均线都大于60日均线，买入，等价于5日和20日均线上穿60日均线，买入；
    # 计算卖出信号：
    sell_signal=np.logical_and(positions > 0, ma5 < ma60, ma20 < ma60 ) #有持仓的情况下，5日和20日均线都小于60日均线，卖出，等价于5日和20日均线上穿60日均线，买入；
    #获取买入信号标的的序号
    target_buy=target[buy_signal].tolist()
    #获取卖出信号标的的序号
    target_sell = target[sell_signal].tolist()

    #策略下单交易：
    for targets in target_buy:
        order_target_value(account_idx=0, target_idx=targets, target_value=1e6/len(target_buy), side=1,
            order_type=2, price=0) #买入下单

    for targets in target_sell:
        order_target_volume(account_idx=0, target_idx=targets, target_volume=0, side=1,
                        order_type=2, price=0) #卖出平仓


#主函数：回测脚本
if __name__ == '__main__':
    #回测函数：
    run_backtest(strategy_name='ThreeLines', file_path='.', target_list=get_code_list('hs300')['code'],
                 frequency='day', fre_num=1, begin_date='2016-01-01', end_date='2018-09-30', fq=1)


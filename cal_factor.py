"""
因子计算模块
"""

from atrader.calcfactor import *


def init(context:ContextFactor):
    # 注册因子PE
    reg_factor(['PE'])


def calc_factor(context: ContextFactor):
    # 获取注册的因子PE的数值
    result = get_reg_factor(context.reg_factor[0], df=True)
    # 提取因子的数值
    result = result['value'].values
    # 将结果转换为列
    return result.reshape(-1, 1)


if __name__ == "__main__":
    # 投资域为上证50
    run_factor(factor_name='PE', file_path='.', targets='hs300', begin_date='2016-01-01',end_date='2016-12-31', fq=1)
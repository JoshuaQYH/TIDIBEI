"""
author: qiuyihao
date: 2019/04/13 - 04-15
description: 单因子测试
"""
import pandas as pd
import numpy as np
import atrader as at
from sklearn import preprocessing
from sklearn import linear_model
import time
from scipy.stats import pearsonr
import datetime


# 中位数去极值法
def filter_MAD(df, factor, n=5):
    """
    :param df: 去极值的因子序列
    :param factor: 待去极值的因子
    :param n: 中位数偏差值的上下界倍数
    :return: 经过处理的因子dataframe
    """
    # print(df)

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


# 判断某一个日期是否为周末，如果为周末，需要返回一个非周末的字符串。
# 当时间是月末时，时间需要向前，当时间时月初是，时间需要向后
# 采用递归实现,最终返回一个非周末的时间串
# (其实这个函数的作用就是帮助减少几次获取因子而已，，，事后发现还不如直接靠get_factor_by_day判断
def find_day_str(day_str):
    """
    :param day_str: 要求标准的时间串 如 2016-01-01
    :return: 返回一个合适的时间串
    """
    year = int(day_str[0:4])
    month = int(day_str[5:7])
    day = int(day_str[8:10])
    any_day = datetime.datetime(year, month, day).strftime("%w")
    result_str = day_str
    if any_day == '6' or any_day == '0':
        if day < 15:
            day += 1
            if day < 10:
                day = '0' + str(day)
            else:
                day = str(day)
        elif day > 15:
            day -= 1
            day = str(day)
        result_str = find_day_str(day_str[0:8] + day)
    return result_str


# 生成起始日期对
def create_date(begin_date, end_date):
    """
    :param begin_date: 开始日期 指明起始年月  如 '2018-01'
    :param end_date: 结束日期 指明结束年月    如 '2018-10'
    :return: 一个起始年月日列表,一个结束年月日列表
     以一个月的第一天和最后一天作为一对日期 如 ['2018-01-01',..] ['2018-01-31',..]
     注：需要排斥这两天为周末或者法定假期的时候
    """
    # 解析字符串
    begin_year = int(begin_date[0:4])
    begin_month = int(begin_date[5:7])
    end_year = int(end_date[0:4])
    end_month = int(end_date[5:7])

    # 待拼接的年日月
    year = begin_year
    month = begin_month

    begin_date_list = []
    end_date_list = []

    big_month = [1, 3, 5, 7, 8, 10, 12]
    small_month = [4, 6, 9, 11]   # 二月另外判断
    while year <= end_year and month <= end_month:
        start = ''
        end = ''

        if month >= 10:
            start = str(year) + '-' + str(month) + '-' + '01'
            end = str(year) + '-' + str(month) + '-'
        else:
            start = str(year) + '-0' + str(month) + '-' + '01'
            end = str(year) + '-0' + str(month) + '-'

        # 避免出现节假日或者周末,若出现则往后推一天
        while at.get_factor_by_day(factor_list=["PE"], target_list=["SZSE.000001"], date=start) is None:
            start_day = int(start[8:10]) + 1
            if start_day < 10:
                start = start[0:8] + '0' + str(start_day)
            else:
                start = start[0:8] + str(start_day)

        begin_date_list.append(start)  # 插入一个非周末非法定假期的开始时间串

        # 判断月为大，为小
        if month in big_month:
            end = end + '31'
        elif month in small_month:
            end = end + '30'
        elif month == 2:
            if year % 4 == 0 and year % 100 != 0 or year % 400 == 0:
                end = end + '29'
            else:
                end = end + '28'

        while at.get_factor_by_day(factor_list=["PE"], target_list=["SZSE.000001"], date=end) is None:
            end_day = int(end[8:10]) - 1
            end = end[0:8] + str(end_day)

        end_date_list.append(end)  # 插入一个非周末，非法定假期的结束时间串

        month += 1
        if month == 13:
            year += 1
            month = 1
    return begin_date_list, end_date_list


# 计算每一个月的单个股票平均收益率
def cal_yield_rate(code, begin_date, end_date):
    """
    :param code: 股票代码
    :param begin_date: K线起始日期，月初
    :param end_date: K线结束日期，月末
    :return: 在该时间内股票的平均收益率
    """
    day_data = at.get_kdata(target_list=[code], frequency='day', fre_num=1, begin_date=begin_date,
                            end_date=end_date, fill_up=False, df=True, fq=1, sort_by_date=True)
    yield_rate = 0.0
    try:
        yield_rate = (day_data['close'][len(day_data) - 1] - day_data['close'][0])/day_data['close'][0]
    except Exception:
        yield_rate = -1
    return yield_rate


# 股票分层函数: 按流通市值进行划分，分为大，中，小市值。
def stock_layered(code_list, sign = 0):
    """
    :param code_list: 未分层的标的代号
    :param sign: = 0，表示不分层；= 1，返回小市值，= 2，返回中市值； = 3， 返回大市值
    :return: 分层后的标的代码
    """
    if sign == 0:
        return code_list
    pass

# 单因子测试函数
def test_factor(factor, block, begin_date_list, end_date_list, layer_sign = 0):
    """
    :param factor:  待测的单因子
    :param block : 股市指数
    :param begin_date_list: 获取每一期因子的开始时间 （12个月，每月一次，从月初开始和月末结束）
    :param end_date_list: 获取每一期因子的结束时间
    :return: 年化夏普率，IC等等，见函数尾部
    注：使用沪深300股作为测试
    """
    # 记录每一个月的股票池总体收益率
    yield_rate_list = []

    # 记录每一个月股票池各股收益率
    single_yield_rate_list = []

    # 因子每期收益率
    factor_return_list = []

    # 因子每期的IC值
    IC_list = []



    # 遍历每一月，月初调仓
    for i in range(len(begin_date_list)):

        # --------------------------------------------- #
        # 1. 提取 K 线数据 和 股票信息
        # --------------------------------------------- #

        print("{} - {}: 获取K线数据！".format(begin_date_list[i], end_date_list[i]))
        code_list = at.get_code_list(block, date=begin_date_list[i])

        code_list = stock_layered(code_list, layer_sign)  # 分层

        # 若要分层回测，这里需要股票池划分
        target_list = code_list['code'].tolist()  # 本月股票池代码
        weight_list = np.array(code_list['weight'].tolist())  # 本月各股票权重
        # 获取因子月初数据
        print("{} - {}: 获取因子数据！".format(begin_date_list[i], end_date_list[i]))
        factor_data = at.get_factor_by_day(factor_list=[factor], target_list=target_list,
                                           date=begin_date_list[i])

        # ----------------------------------------------- #
        # 2. 数据预处理
        # ----------------------------------------------- #

        # 平均值填充缺失值 中位数去极值 & z-score 规范化
        factor_data = factor_data.fillna(factor_data[factor].mean())
        factor_data = filter_MAD(factor_data, factor, n=5)
        factor_data[factor] = preprocessing.scale(factor_data[factor])

        # 提取因子列，变为np array
        factor_data = np.array(factor_data[factor].tolist())

        # ------------------------------------------------- #
        # 3.从 K 线和股票数据中计算本月的个股收益率和权重
        # 以及IC值
        # ------------------------------------------------- #

        yield_rate = []  # 股票池个股本月平均收益率
        tmp_target_list = target_list
        for j, target in enumerate(target_list):
            rate = cal_yield_rate(target, begin_date_list[i], end_date_list[i])
            if rate != -1:  # 计算标的股票的本月收益率
                yield_rate.append(cal_yield_rate(target, begin_date_list[i], end_date_list[i]))
            else:  # 收益率计算出现错误，从股票池中删除，权重列表中删除，因子列表中删除
                tmp_target_list = np.delete(tmp_target_list, [j])
                weight_list = np.delete(weight_list, [j])
                factor_data = np.delete(factor_data, [j])

        IC = pearsonr(yield_rate, factor_data)[0]  # 获取IC值

        IC_list.append(IC)  # 记录IC值

        weight_list = weight_list / weight_list.sum()  # 权重归一化
        weight_list = weight_list.reshape(-1, 1)
        factor_data = factor_data.reshape(-1, 1)
        yield_rate = np.array(yield_rate).reshape(-1, 1)

        # ----------------------------------------------- #
        # 4. 月初因子和本月收益率进行拟合, 获取因子收益率
        # ----------------------------------------------- #

        print("{} - {}: 开始拟合！".format(begin_date_list[i], end_date_list[i]))
        LR = linear_model.LinearRegression()  # 线性拟合器
        LR.fit(factor_data, yield_rate)  # 拟合月初因子和本月平均收益率

        coef_list = list(LR.coef_)[0]
        coef = coef_list[0]
        factor_return_list.append(coef)  # 记录当期的因子收益率 保留小数点两位

        # -------------------------------------------------- #
        # 5. 预测各股票本月收益率，计算股票池整体收益。
        # -------------------------------------------------- #
        print("{} - {}: 开始预测！".format(begin_date_list[i], end_date_list[i]))
        pred_yield_rate = LR.predict(factor_data)  # 预测的各股票收益率

        rate_list = list(pred_yield_rate)[0]
        rate_list = [round(r, 2) for r in rate_list]
        single_yield_rate_list.append(rate_list)  # 记录当月各股票收益率 小数点两位

        # 利用权重和个股收益计算股票池整体平均收益率
        mean_yield_rate = (pred_yield_rate * weight_list).sum()

        # 记录当月股票整体平均收益率
        yield_rate_list.append(round(float(mean_yield_rate), 2))  # 小数点两位

        print("{} - {}: 股票平均收益率拟合完毕！".format(begin_date_list[i], end_date_list[i]))

    # --------------------------------------------------- #
    # 汇总数据
    # --------------------------------------------------- #

    # 计算超额收益率
    yield_rate_array = np.array(yield_rate_list)
    over_rate = yield_rate_array - 0.004  # 0.004 代表无风险利率
    # 超额收益率均值和标准差
    mean_over_rate = over_rate.mean()
    std_over_rate = over_rate.std()

    # 单位时间夏普率
    sharp_ratio = mean_over_rate / std_over_rate
    # 年化夏普率
    sharp_ratio = np.sqrt(12) * sharp_ratio

    # 计算股票收益率均值方差
    yield_rate_array = np.array(yield_rate_list)
    average_yield_rate = np.mean(yield_rate_array)
    var_yield_rate = np.var(yield_rate_array)

    # 计算因子收益率的均值 标准差
    factor_return_array = np.array(factor_return_list)
    average_factor_return = np.mean(factor_return_array)
    std_factor_return = np.std(factor_return_array)

    # 计算因子收益率大于0的概率
    factor_greater_than_zero = sum([1 for i in factor_return_list if i > 0]) / len(factor_return_list)

    # 计算IC的平均值和标准差
    average_IC = np.mean(np.array(IC_list))
    std_IC = np.std(np.array(IC_list))
    # 计算 IC > 0的概率
    IC_greater_than_zero = sum([1 for i in IC_list if i > 0]) / len(IC_list)

    # 返回夏普率，波动率（收益率方差），因子收益均值，因子收益率，
    test_result_dict = dict()
    test_result_dict["年化夏普率"] = sharp_ratio
    test_result_dict["波动率"] = var_yield_rate
    test_result_dict["因子收益均值"] = average_factor_return
    test_result_dict["因子收益标准差"] = std_factor_return
    test_result_dict["因子收益>0概率"] = factor_greater_than_zero
    test_result_dict["IC均值"] = average_IC
    test_result_dict["IC标准差"] = std_IC
    test_result_dict["IC>0概率"] = IC_greater_than_zero

    return test_result_dict


# 同时多次测试因子，返回一个DataFrame
def test_all_factors(factor_list, block, begin_date, end_date, layer_sign=0):
    """
    :param factor_list: 因子列表
    :param block: 股市指数
    :param begin_date: 开始年月
    :param end_date: 结束年月
    :return: 返回各因子的测试指标结果
    """
    begin_date_list, end_date_list = create_date(begin_date, end_date)
    result_dict_list = list()
    for factor in factor_list:
        result_dict = test_factor(factor, block, begin_date_list, end_date_list, layer_sign)
        result_dict_list.append(result_dict)

    return pd.DataFrame(result_dict_list, index=factor_list)


result = test_all_factors(["NLSIZE", "MktValue", "BIAS10", "NegMktValue", "CurrentAssetsRatio",
                           "MLEV", "Variance20", "ROAEBIT"],
                          'hs300', '2016-01', '2018-09',
                          layer_sign=0)  # 0 不分层  1 低流通市值  2 中流通市值  3 高流通市值

result.to_csv("single_factor_test.csv", sep=',')




"""
author: qiuyihao
date: 2019-04-22
description: 同类因子进行共线性分析,绘制相关系数矩阵
            获取每一类因子中的历史序列，该序列每一个因子由同时期股票的非空因子平均求得。
            计算相关序列的相关系数，绘制相关系数矩阵
"""
import numpy as np
import pandas as pd
import atrader as at
import seaborn as sns
import matplotlib.pyplot as plt


def draw_heatmap(df, filename):
    dfData = df.corr()
    plt.subplots(figsize=(13, 13))
    sns.heatmap(dfData, annot=True, vmax=1, vmin=0, square=True, cmap='Blues')
    plt.savefig(filename)
    plt.show()


def analysis_factor(factor_list, code_list, filename):
    print(factor_list, code_list[0])
    factor_data = at.get_factor_by_code(factor_list=factor_list, target=code_list[0],
                                        begin_date='2016-01-01', end_date='2018-09-30')

    factor_data = factor_data.drop(['date'], axis=1)

    not_full_num = len(code_list)

    for tf in factor_data.isnull().any():
        if tf == True:
            factor_data = pd.DataFrame(np.full([factor_data.shape[0], factor_data.shape[1]], 0.0),
                                       columns=[factor_list])
            not_full_num -= 1
            break

    factor_data.columns = factor_list

    for i in range(len(code_list) - 1):
        tmp_data = at.get_factor_by_code(factor_list, target=code_list[i+1],
                                         begin_date='2016-01-01', end_date='2018-09-30')
        tmp_data = tmp_data.drop(['date'], axis=1)
        null_flag = False
        for tf in tmp_data.isnull().any():
            if tf == True:
                null_flag = True
                not_full_num -= 1
                print("NAN... pass ")
                break
        if not null_flag:
            if tmp_data.iloc[:, 0].mean() >= 10000000:
                tmp_data /= 100000  # 某些因子数据过于庞大，需要缩小
            factor_data = factor_data + tmp_data
            print("add ... ")
    factor_data /= not_full_num
    draw_heatmap(factor_data, filename)


if __name__ == '__main__':
    A = at.get_code_list('hs300', date='2016-01-01')
    code_list = A['code'].tolist()

    file_name_list = ["Q1_基础类", "Q1_质量类"]
        #, "情绪类", "价值类", "每股指标类",
        #              "行业分析师类", "特色技术指标类"]

    factor_list = [['AdminExpenseTTM', 'NIAP', 'FinanExpenseTTM', 'NetIntExpense'],  # 基础类
                   ['DebtEquityRatio', 'SuperQuickRatio']  # 质量类
                  ]
                 # ['TVMA20', 'VOL20', 'OBV20', 'JDQS20'],  # 情绪类
                 #  ['PE', 'PB', 'PS', 'NLSIZE', 'TA2EV', 'CTOP'],  # 成长因子类
                 #  ['BasicEPS', 'EPS', 'EnterpriseFCFPS'],  # 每股指标类
                 #  ['RSTR24', 'FY12P', 'SFY12P', 'PEIndu', 'EPIBS'],  # 行业分析师类
                 #  ['AVGPRICE', 'BOP', 'KAMA', 'LINEARREG', 'STDDEV']  # 特色技术指标类

    for i, factor in enumerate(factor_list):
        #if i != 1:
        #    continue
        print(file_name_list[i])
        analysis_factor(factor, code_list, file_name_list[i])  # 最终得到因子相关系数矩阵


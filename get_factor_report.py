import atrader as at
import pandas as pd
import numpy as np
import sys
"""
运行之前！！！！！！！！！！！
修改输出的csv文件名！！！！！！
注意不要重复，可能会覆盖原来的文件！！！
"""

class_name = sys.argv[1]
csv_file = class_name + ".csv"
strategy_dicts = at.get_strategy_id()
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
    result = at.get_performance(strategy_id)
    save_dict['测试因子'].append(result['strategy_name'])
    save_dict['年化收益率'].append(result['annu_return'])
    save_dict['年化夏普率'].append(result['sharpe_ratio']*np.sqrt(12))
    save_dict['最大回撤率'].append(result['max_drawback_rate'])
    save_dict['alpha'].append(result['alpha'])
    save_dict['beta'].append(result['beta'])
    save_dict['信息比率'].append(result['info_ratio'])

df = pd.DataFrame(save_dict)
df.to_csv(csv_file, sep=',')
print(df)
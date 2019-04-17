:: 添加因子ID，进行单因子回测

:: 基础类 ------------------------------- 该类结束测试
:: 净运行资本
::python single_factor_test.py NetWorkingCapital
:: 毛利  !
::python single_factor_test.py GrossProfit
:: 企业自由现金流量
::python single_factor_test.py FCFF
:: 带息债务
::python single_factor_test.py IntDebt
:: 运营资本 !
::python single_factor_test.py WorkingCapital
:: 固定资产合计  !
::python single_factor_test.py TotalFixedAssets
:: 净利息费用 !
::python single_factor_test.py NetIntExpense
:: 价值变动净收益
python single_factor_test.py ValueChgProfit
:: 留存收益
python single_factor_test.py RetainedEarnings
:: 财务费用 !
python single_factor_test.py FinanExpenseTTM
:: 管理费用 !
python single_factor_test.py AdminExpenseTTM
:: 净利润
python single_factor_test.py NetProfitTTM

:: 质量类 -------------------------------------- 该类可结束测试
:: 产权比率 !
::python single_factor_test.py DebtEquityRatio
:: 超速动比率 !
::python single_factor_test.py SuperQuickRatio
:: 现金比率  !
::python single_factor_test.py CashToCurrentLiability
:: 流动资产周转率  !
::python single_factor_test.py CurrentAssetsRate
:: 投入资本回报率 !
::python single_factor_test.py ROIC
:: 利息保障倍数 !
::python single_factor_test.py InterestCover
:: 净资产收益率平均
::python single_factor_test.py ROEWeighted
:: 总资产报酬  !
::python single_factor_test.py ROAEBIT
:: 股利支付率
::python single_factor_test.py DividendPaidRatio

:: 收益风险类---------------------------------- 该类效果极差
:: 60日方差 !
::python single_factor_test.py Variance60
:: 股价偏度 !
::python single_factor_test.py Skewness20
:: 历史波动 !
::python single_factor_test.py HSIGMA
:: 20日信息比率 !
::python single_factor_test.py InformationRatio20
:: 20日夏普率 !
::python single_factor_test.py Sharperatio20
:: 历史贝塔 !
::python single_factor_test.py HBETA
:: 下跌贝塔 !
::python single_factor_test.py DDNBT
:: 20日收益方差
::python single_factor_test.py GainVariance20
:: 20日损失方差
::python single_factor_test.py LossVariance20
:: 20日收益损失方差比
::python single_factor_test.py GainLossVarianceratio20
:: 超额收益标准差
::python single_factor_test.py DASTD
:: 股权向后复权因子
::python single_factor_test.py BackwardADJ
:: 个股20日 alpha
::python single_factor_test.py CAPMAlpha20






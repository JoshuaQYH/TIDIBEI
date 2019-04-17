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
::python single_factor_test.py ValueChgProfit
:: 留存收益
::python single_factor_test.py RetainedEarnings
:: 财务费用 !
::python single_factor_test.py FinanExpenseTTM
:: 管理费用 !
::python single_factor_test.py AdminExpenseTTM
:: 净利润
::python single_factor_test.py NetProfitTTM

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
:: 60日方差
::python single_factor_test.py Variance60
:: 股价偏度
::python single_factor_test.py Skewness20
:: 历史波动
::python single_factor_test.py HSIGMA
:: 20日信息比率
::python single_factor_test.py InformationRatio20
:: 20日夏普率
::python single_factor_test.py Sharperatio20
:: 历史贝塔
::python single_factor_test.py HBETA
:: 下跌贝塔
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

:: 个股收益的120日峰度
::python single_factor_test.py Kurtosis120
:: 个股收益的60日峰度
::python single_factor_test.py Kurtosis60
:: 个股收益的20日峰度
::python single_factor_test.py Kurtosis20
:: 下跌贝塔
::python single_factor_test.py DDNBT
:: 下跌相关系数
:: python single_factor_test.py DDNCR
:: 下跌波动
::python single_factor_test.py DDNSR
:: 60日信息比率
::python single_factor_test.py InformationRatio60
:: 60日夏普率
::python single_factor_test.py Sharperatio60

:: 120日信息比率
::python single_factor_test.py InformationRatio120
:: 120日夏普率
::python single_factor_test.py Sharperatio120
:: 120 日特诺雷比
::python single_factor_test.py Treynorratio120
:: 120日 beta
::python single_factor_test.py Beta120
:: 12 个月累计收益范围的对数
::python single_factor_test.py CMRA12
:: 60日收益方差
::python single_factor_test.py GainVariance60
:: 120日损失方差
::python single_factor_test.py LossVariance120
:: 120日收益损失方差比
::python single_factor_test.py GainLossVarianceratio120

:: 情绪类 -------------------
:: 20日成交金额的移动平均值
::python single_factor_test.py TVMA20
:: 20日平均换手率
::python single_factor_test.py VOL20
:: 20日成交量标准差
::python single_factor_test.py VSTD20
:: 正成交量指标
::python single_factor_test.py PVI
:: 成交量比率
::python single_factor_test.py VR
:: 20日资金流量
::python single_factor_test.py MONEYFLOW20
:: 20日收集派发指标
::python single_factor_test.py ACD20
:: 人气指标
::python single_factor_test.py AR
:: 20日能量潮指标
::python single_factor_test.py OBV20
:: 阶段强势指标
::python single_factor_test.py JDQS20
:: 资本利得突出量
::python single_factor_test.py CGO_10
:: 显著性因子 20
::python single_factor_test.py ST_20
:: 综合效用因子 20
::python single_factor_test.py TK_20
:: 抢跑因子
::python single_factor_test.py FR_pure



:: 模式识别类 -----------------------------------
:: 藏婴吞没（CDLCONCEALBABYSWALL）
python single_factor_test.py CDLCONCEALBABYSWALL
:: 射击之星（CDLSHOOTINGSTAR）
python single_factor_test.py CDLSHOOTINGSTAR
:: 十字暮星（CDLEVENINGDOJISTAR）
python single_factor_test.py CDLEVENINGDOJISTAR
:: 吞噬模式（CDLENGULFING）
python single_factor_test.py CDLENGULFING
:: 刺透形态（CDLPIERCING）
python single_factor_test.py CDLPIERCING
:: 倒锤头（CDLINVERTEDHAMMER）
python single_factor_test.py CDLINVERTEDHAMMER

::python get_factor_report.py 模式识别类

:: 每股指标类 ----------------------------------
:: 基本每股收益（BasicEPS）
python single_factor_test.py BasicEPS
:: 每股收益TTM值（EPS）
python single_factor_test.py EPS
:: 每股净资产（NetAssetPS）
python single_factor_test.py NetAssetPS
:: 每股营业总收入（TORPS）
python single_factor_test.py TORPS
:: 每股营业利润（OperatingProfitPS）
python single_factor_test.py OperatingProfitPS
:: 每股息税前利润（EBITPS）
python single_factor_test.py EBIPTS
:: 每股现金流量净额（CashFlowPS）
python single_factor_test.py CashFlowPS
:: 每股企业自由现金流量（EnterpriseFCFPS）
python single_factor_test.py EnterpriseFCFPS

::python get_factor_report.py 每股指标类

:: 行业与分析师类 ------------------------------
:: 12月相对强势(RSTR12)
python single_factor_test.py RSTR12
:: 24月相对强势(RSTR24)
python single_factor_test.py RSTR24
:: 分析师盈利预测（FY12P）
python single_factor_test.py FY12P
:: 分析师营收预测（SFY12P）
python single_factor_test.py SFY12P
:: （PB–PB的行业均值）/PB的行业标准差（PBIndu）
python single_factor_test.py PBIndu
:: PCF–PCF的行业均值）/PCF的行业标准差（PCFIndu）
python single_factor_test.py PCFIndu
::（PE–PE的行业均值）/PE的行业标准差（PEIndu）
python single_factor_test.py PEIndu
:: （PS–PS的行业均值）/PS的行业标准差（PSIndu）
python single_factor_test.py PSIndu
:: 投资回报率预测（EPIBS）
python single_factor_test.py EPIBS
:: 未来预期盈利增长（FEARNG）
python single_factor_test.py FEARNG
:: 未来预期盈收增长（FSALESG）
python single_factor_test.py FSALESG
:: 长期盈利增长预测（EgibsLong）
python single_factor_test.py EgibsLong

::python get_factor_report.py 行业与分析师类

:: 特色技术指标 --------------------------------
:: 绝对价格振荡器(APO)
python single_factor_test.py APO
:: 平均价格(AVGPRICE)
python single_factor_test.py AVGPRICE
:: 均势指标(BOP)
python single_factor_test.py BOP
:: 考夫曼自适应移动平均线（KAMA）
python single_factor_test.py KAMA
:: 线性回归（LINEARREG）
python single_factor_test.py LINEARREG
:: 标准差(STDDEV)
python single_factor_test.py STDDEV
:: 时间序列预测（TSF）
python single_factor_test.py TSF

::python get_factor_report.py 特色技术指标

::python get_factor_report.py last4
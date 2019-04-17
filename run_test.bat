:: 添加因子ID，进行单因子回测

:: 基础类 -------------------------------
:: 净运行资本
::python single_factor_test.py NetWorkingCapital
:: 毛利
::python single_factor_test.py GrossProfit
:: 企业自由现金流量
::python single_factor_test.py FCFF
:: 带息债务
::python single_factor_test.py IntDebt
:: 运营资本
::python single_factor_test.py WorkingCapital
:: 固定资产合计
::python single_factor_test.py TotalFixedAssets

:: 质量类 --------------------------------------
:: 产权比率
::python single_factor_test.py DebtEquityRatio
:: 超速动比率
::python single_factor_test.py SuperQuickRatio
:: 现金比率
::python single_factor_test.py CashToCurrentLiability
:: 流动资产周转率
::python single_factor_test.py CurrentAssetsRate
:: 投入资本回报率
::python single_factor_test.py ROIC
:: 利息保障倍数
::python single_factor_test.py InterestCover
:: 净资产收益率平均
::python single_factor_test.py ROEWeighted
:: 总资产报酬
::python single_factor_test.py ROAEBIT
:: 股利支付率
::python single_factor_test.py DividendPaidRatio

:: 收益风险类----------------------------------
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


:: #run by lxl BEGIN#
:: 模式识别类 -----------------------------------
:: 藏婴吞没（CDLCONCEALBABYSWALL）
::python single_factor_test.py CDLCONCEALBABYSWALL
:: 射击之星（CDLSHOOTINGSTAR）
:: python single_factor_test.py CDLSHOOTINGSTAR

:: 每股指标类 ----------------------------------
:: 基本每股收益（BasicEPS）
::python single_factor_test.py BasicEPS
:: 每股收益TTM值（EPS）
::python single_factor_test.py EPS
:: 每股净资产（NetAssetPS）
::python single_factor_test.py NetAssetPS
:: 每股营业总收入（TORPS）
::python single_factor_test.py TORPS
:: 每股营业利润（OperatingProfitPS）
::python single_factor_test.py OperatingProfitPS
:: 每股息税前利润（EBITPS）
::python single_factor_test.py EBIPTS
:: 每股现金流量净额（CashFlowPS）
::python single_factor_test.py CashFlowPS
:: 每股企业自由现金流量（EnterpriseFCFPS）
::python single_factor_test.py EnterpriseFCFPS

:: 行业与分析师类 ------------------------------
:: 12月相对强势(RSTR12)
::python single_factor_test.py RSTR12
:: 24月相对强势(RSTR24)
::python single_factor_test.py RSTR24
:: 分析师盈利预测（FY12P）
::python single_factor_test.py FY12P
:: 分析师营收预测（SFY12P）
::python single_factor_test.py SFY12P
:: （PB–PB的行业均值）/PB的行业标准差（PBIndu）
::python single_factor_test.py PBIndu
:: PCF–PCF的行业均值）/PCF的行业标准差（PCFIndu）
::python single_factor_test.py PCFIndu
::（PE–PE的行业均值）/PE的行业标准差（PEIndu）
::python single_factor_test.py PEIndu
:: （PS–PS的行业均值）/PS的行业标准差（PSIndu）
::python single_factor_test.py PSIndu
:: 投资回报率预测（EPIBS）
::python single_factor_test.py EPIBS
:: 未来预期盈利增长（FEARNG）
::python single_factor_test.py FEARNG
:: 未来预期盈收增长（FSALESG）
::python single_factor_test.py FSALESG
:: 长期盈利增长预测（EgibsLong）
::python single_factor_test.py EgibsLong

:: 特色技术指标 --------------------------------
:: 绝对价格振荡器(APO)
::python single_factor_test.py APO
:: 平均价格(AVGPRICE)
::python single_factor_test.py AVGPRICE
:: 均势指标(BOP)
::python single_factor_test.py BOP
:: 考夫曼自适应移动平均线（KAMA）
::python single_factor_test.py KAMA
:: 线性回归（LINEARREG）
::python single_factor_test.py LINEARREG
:: 标准差(STDDEV)
::python single_factor_test.py STDDEV
:: 时间序列预测（TSF）
::python single_factor_test.py TSF

:: #run by lxl END#

python get_factor_report.py




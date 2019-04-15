# Tidy-QuantTrading
泰迪杯数据挖掘比赛协作仓库。——基于机器学习方法构建多因子选股模型。
> Group Members：XiaoRu Chen，Xiaoling Ling，Yihao Qiu

## TODO

### 单因子测试

确定一个单因子测试文件，定义待测因子列表，执行多次单因子runtest。
- 保留回测报告，获取字段，保存在CSV文件。
- 结果可视化。
- 筛选得到最优因子。
- 因子做共线性分析，获取最终因子。

### 选用机器学习模型回测
- 特征和标签构建。
- 等权重线性模型。
- 建立baseline models，尝试使用多种模型。SVR，RNN(LSTM)，xgboost, random_forest...
- 交易逻辑确定。
- 回测结果记录，分析。

### 风险控制
...


## LINK 
- [AutoTrader 官方文档](https://www.digquant.com.cn/documents/17#h1-u5FEBu901Fu5F00u59CB-0)
- [股票交易名词解释: 多头，空头，平仓，持仓，调仓....](http://stock.hexun.com/menu/stepbystep/step3.html)
- [头寸解释](https://wiki.mbalib.com/wiki/%E5%A4%B4%E5%AF%B8)

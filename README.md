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

> _**官方提示：**_ 单因子问题，可以手动实现一下，或者可以弱化一下，通过间接的方法去实现这个问题，并不一定要实现一个完整的回测框架才能解决单因子分析的问题

### 选用机器学习模型回测
- 特征和标签构建。
- 等权重线性模型。
- 建立baseline models，尝试使用多种模型。SVR，RNN(LSTM)，xgboost, random_forest...
- 交易逻辑确定。
- 回测结果记录，分析。

> _**官方提示：**_ random forest, SVM, Adaboost, etc.
> 
#### 关于模型的一些设想
- 可参考论文[GBDT提取特征 + SVM二分类的方法](https://github.com/JoshuaQYH/TIDIBEI/blob/master/references/SA20190100000_36930159.pdf)
- [LSTM进行选股](https://qiniu-images.datayes.com/huatai9.pdf)（在月频数据较少，可能效果不好）
- [Adaboost](http://pg.jrj.com.cn/acc/Res/CN_RES/INVEST/2016/5/31/ed36ae43-0f6e-4051-bb9c-2e9a67632d74.pdf), randomforest, svm([启发式](http://or.nsfc.gov.cn/bitstream/00001903-5/353458/1/1000008947591.pdf)），[xgboost](https://cloud.tencent.com/developer/article/1137060)等等进行集成。如[Stacking](https://cloud.tencent.com/developer/article/1137060),bagging. 

### 风险控制
- 风险模型
- 择时模型


## 文件说明
- `data_exploration.ipynb`: atrader API调用测试文件。
- `get_factor_report.py`: 当单因子回测结束之后，执行文件，得到策略字段。
- `single_factor_test.py`: 单因子测试文件。
- `find_factor.py`: 自实现的因子绩效分析文件(**已弃用**)
- `run_test.bat`: 脚本自动化运行python程序，实现多次执行策略。
- 'factor_analysis': 类内因子共线性分析文件，绘制相关系数矩阵。

## LINK 
- [AutoTrader 官方API文档](https://www.digquant.com.cn/documents/17#h1-u5FEBu901Fu5F00u59CB-0)
- [股票交易名词解释: 多头，空头，平仓，持仓，调仓....](http://stock.hexun.com/menu/stepbystep/step3.html)
- [头寸解释](https://wiki.mbalib.com/wiki/%E5%A4%B4%E5%AF%B8)
- [阮一峰常用git命令清单](http://www.ruanyifeng.com/blog/2015/12/git-cheat-sheet.html)
- [点宽因子数据字典](https://www.digquant.com.cn/documents/23)
- [名词解释：IC/IR](https://xueqiu.com/1652627245/108835836)
- [名词解释：alpha值/beta值](https://blog.csdn.net/yezi113yezi/article/details/81078128)
- [A题华师现场解读](https://edu.tipdm.org/)
- [人工智能阿尔法策略框架-对282个因子的分析](https://www.jiqizhixin.com/articles/2019-01-26-5)

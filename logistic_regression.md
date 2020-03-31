[TOC]



# 1、逻辑回归概述

## 1.1 名为“回归”的分类器

名为回归的线性分类器，由线性回归变化而来，已知线性回归的方程如下：
$$
\begin{equation}\begin{split}
z&=\theta_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n\\
 &=[\theta_0,\theta_1,\theta_2,\cdots,\theta_n]*\begin{bmatrix}x_0\\x_1\\x_2\\\vdots \\x_n\end{bmatrix}\\
 &=\theta^Tx(x_0=1)\end{split}\end{equation}
$$
线性回归的任务：构造一个预测函数$z$映射输入的特征矩阵$x$和标签值$y$的线性关系，**构建预测函数的核心就是找到模型的参数：$\theta^T$和$\theta_0$（最小二乘法）**

线性回归，连续型的输入$x$导出连续型的标签$y$，若连续型的输入$x$如何导出离散型的标签$y$呢？

Sigmoid函数（联系函数）：$g(z)$接近1时样本标签为1，接近0时样本标签0，实现分类
$$
g(z)=\frac {1}{1+e^{-z}}
$$
<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200328171052583.png" alt="image-20200328171052583" style="zoom:50%;" />

| Sigmoid函数的公式和性质                                      |
| ------------------------------------------------------------ |
| Sigmoid函数：<br> 一个S型函数，可将任何实数映射到（0,1）之间，自变量趋近负无穷$g(z)$趋于0，自变量趋近正无穷$g(z)$趋于1，故可将任意值函数转换为更适合二分类的函数<br>也被当做归一化的一种方法 ，但只能无限趋近0、1，而MinMaxScalar归一化之后，可将数据压缩在[0,1]之间，可以取到0、1 |

二元逻辑回归的一般形式：***y(x)有着概率的性质，但不是真正的概率***
$$
g(z)=y(x)=\frac{1}{1+e^{-\theta^Tx}}
$$
通过推导发现：
$$
ln\frac{y(x)}{1-y(x)}=\theta^Tx
$$
左侧即为对数几率，

[为什么要对对数几率进行建模？](https://zhuanlan.zhihu.com/p/42656051)

## 1.2 为什么需要逻辑回归

决策树、随机森林不需要对数据进行任何预处理，**逻辑回归是一个返回对数几率的、在线性数据上表现优异的分类器，主要用在金融领域**

逻辑回归仍然受到工业商业喜爱，使用广泛，因为：

1. 逻辑回归对线性关系的拟合效果好

   - 若数据是非线性的，不要用逻辑回归；
   - 梯度提升树GBDT对线性数据的效果比逻辑回归好

2. 计算快

   对于线性数据，计算效率优于SVM、随机森林

3. 分类结果不是固定的0、1，而是类概率数字

   可以把逻辑回归返回的结果当做连续数据使用，如评分卡制作时，除了判断是否违约外，还需要给出“信用分”

4. 抗噪能力强、小数据集上表现好等（大数据集，树模型表现好）

## 1.3 sklearn中的逻辑回归

| 逻辑回归相关的类                      | 说明                                               |
| ------------------------------------- | -------------------------------------------------- |
| linear_model.LogisticRegression       | 逻辑回归分类器(又叫logit回归，最大熵分类器)        |
| linear_model.LogisticRegressionCV     | 带交叉验证的逻辑回归分类器                         |
| linear_model.logistic_regression_path | 计算Logistic回归模型以获得正则化参数的列表         |
| linear_model.SGDClassifier            | 利用梯度下降求解的线性分类器(SVM，逻辑回归等等)    |
| linear_model.SGDRegressor             | 利用梯度下降最小化正则化后的损失函数的线性回归模型 |
| metrics.log_loss                      | 对数损失，又称逻辑损失或交叉熵损失                 |
|                                       |                                                    |
| **其他会涉及的类**                    | **说明**                                           |
| metrics.confusion_matrix              | 混淆矩阵，模型评估指标之一                         |
| metrics.roc_auc_score                 | ROC曲线，模型评估指标之一                          |
| metrics.accuracy_score                | 精确性，模型评估指标之一                           |

# 2、linear_model.LogisticRegression

## 2.1 二元逻辑回归的损失函数

class `sklearn.linear_model.LogisticRegression` (penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, verbose=0, warm_start=False, n_jobs=None)

逻辑回归基于`训练数据`求解$\theta$的需求，并且希望训练出来的模型尽可能拟合训练数据，因此出现损失函数：

| 损失函数                                                     |
| ------------------------------------------------------------ |
| 衡量参数为$\theta$的模型拟合训练集时产生的信息损失的大小，并以此衡量参数$\theta$的优劣<br> 损失函数小，模型在**训练集**上表现优异，拟合充分，参数优秀；<br> 损失函数大，模型在**训练集**上表现差劲，拟合不足，参数糟糕<br> **我们追求，能够让损失函数最小化的参数组合**<br> <br> 注意：没有“求解参数”需求的模型没有损失函数，比如KNN、决策树 |

逻辑回归的损失函数，由极大似然估计推导：
$$
J(\theta)=-\sum_{i=1}^m(y_i*log(y_\theta(x_i))+(1-y_i)*log(1-y_\theta(x_i)))
$$
其中，$\theta$表示求解出来的一组参数，$m$是样本个数，$y_i$是样本$i$在样本上的真实标签，$y_\theta(x_i)$是样本$i$基于参数$\theta$计算出来的逻辑回归返回值，$x_i$是样本$i$各个特征的取值

<font color=#0000FF size=2 face="黑体">***推导过程如下：***  </font>

二元逻辑回归的标签服从伯努利分布（0-1分布），对于一个样本$i$的预测情况如下：

预测为1的概率：$P_1=P(\hat y_i=1|x_i,\theta)=y_\theta(x_i)$

预测为0的概率：$P_0=P(\hat y_i=0|x_i,\theta)=1-y_\theta(x_i)$

整合如下：$P(\hat y_i|x_i,\theta)=P_1^{y_i}*P_0^{1-y_i}$

为了保证模型拟合效果最好，损失最小，希望$P(\hat y_i|x_i,\theta)=1$，或者说追求$P(\hat y_i|x_i,\theta)$最大值

对于所有样本：
$$
\begin{equation}\begin{split}
P&=\prod_{i=1}^m P(\hat y_i|x_i,\theta)\\
 &=\prod_{i=1}^m (P_1^{y_i}*P_0^{1-y_i})\\
 &=\prod_{i=1}^m (y_\theta(x_i)^{y_i}*(1-y_\theta(x_i))^{1-y_i})
\end{split}\end{equation}
\\
$$
取对数后：
$$
\begin{equation}\begin{split}
logP&=log\prod_{i=1}^m (y_\theta(x_i)^{y_i}*(1-y_\theta(x_i))^{1-y_i})\\
    &=\sum_{i=1}^m (log\ y_\theta(x_i)^{y_i}+ log(1-y_\theta(x_i))^{1-y_i})\\
    &=\sum_{i=1}^m (y_i*log\ y_\theta(x_i)+(1-y_i)*log(1-y_\theta(x_i)))
\end{split}\end{equation}\\
$$

这就是交叉熵函数，对$logP$取负值，即我们的损失函数$J(\theta)$				

| 关键概念：似然与概率                                         |
| ------------------------------------------------------------ |
| 二者概念相似，以$P(\hat y_i|x_i,\theta)$为例：<br> 若参数$\theta$已知，特征向量$x_i$未知，$P$是在探索不同特征值下所有可能$\hat y$的可能性，即概率，研究自变量和因变量的关系<br> 若参数$\theta$未知，特征向量$x_i$已知，$P$是在探索不同参数下素有可能$\hat y$的可能性，即似然，研究参数与因变量的关系<br> 在逻辑回归中，特征矩阵已知，参数未知，且追求$P(\hat y_i,\theta)$的最大值，即追求“极大似然” |


在追求损失函数的最小值，即让模型在训练集上表现最优，可能会导致模型过拟合，**通过正则化控制逻辑回归过拟合问题**

## 2.2 正则化

正则化用来防止模型过拟合，常用的L1、L2正则化，在sklearn中，正则化后的公式如下：
$$
J(\theta)_{L1}=C*J(\theta)+\sum_{j=1}^n|\theta_j|\ \ (j≥1) \\
J(\theta)_{L2}=C*J(\theta)+\sqrt{\sum_{j=1}^n(\theta_j)^2}\ \ (j≥1) \\
$$
其中，$J(\theta)$是损失函数，$C$是控制正则化程度的超参数（$C$越大，正则化越弱，$C$越小，正则化越强），$n$代表方程中特征总数即参数总数，$j$代表每个参数，$j≥1$是因为参数向量$\theta$中第一个参数为$\theta_0$，即截距，通常不参与正则化

在sklearn中：

| 参数    | 说明                                                         |
| ------- | ------------------------------------------------------------ |
| penalty | 可以输入"l1"或"l2"来指定使用哪一种正则化方式，不填写默认"l2"。 注意，若选择"l1"正则化，参数solver仅能够使用求解方式”liblinear"和"saga“，若使用“l2”正则 化，参数solver中所有的求解方式都可以使用。 |
| C       | C正则化强度的倒数，必须是一个大于0的浮点数，不填写默认1.0，即默认正则项与损失函数的 比值是1:1。C越小，损失函数会越小，模型对损失函数的惩罚越重，正则化的效力越强，参数$\theta$会逐渐被压缩得越来越小。 |

<font color=#0000FF size=2 face="黑体">***L1正则化和L2正则化的异同点：***</font>

- L1正则化

  可将参数压缩为0；

  当$C$逐渐变小、正则化逐渐加强的过程中，携带信息量小、对模型贡献不大的特征的参数会更快变成0，因此L1正则化本质上也是一个特征选择的过程，掌管了参数的**"稀疏性"**；

  若特征量很大、数据维度很高，倾向于使用L1正则化

- L2正则化

  可将参数尽量小，不会为0；

  当$C$逐渐变小、正则化逐渐加强的过程中，会尽量让每个特征都有一些小的贡献，携带信息量小、对模型贡献不大的特征的参数会接近0；

  若只是为了防止过拟合，L2正则化即可

**两种正则化下$C$的取值，可以通过学习曲线调整**

```python
from sklearn.linear_model import LogisticRegression as LR 
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt 

data=load_breast_cancer()
x=data.data
y=data.target
data.data.shape

## 实例化+训练，solver不需要写
lrl1=LR(penalty="l1",solver="liblinear",C=0.5,max_iter=1000)
lrl2=LR(penalty="l2",solver="liblinear",C=0.5,max_iter=1000)
lrl1=lrl1.fit(x,y)
lrl2=lrl2.fit(x,y)

## 逻辑回归的重要属性coef_,查看每个特征对应的参数
lrl1.coef_
lrl2.coef_

## 哪个正则化效果更好？
l1=[]
l2=[]
l1test=[]
l2test=[]
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=420)
for i in np.linspace(0.05,1,19): ## 0.05到1取19个数
    lrl1=LR(penalty="l1",solver="liblinear",C=i,max_iter=1000)
    lrl2=LR(penalty="l2",solver="liblinear",C=i,max_iter=1000)  
    lrl1=lrl1.fit(train_x,train_y)
    lrl2=lrl2.fit(train_x,train_y)
    
    # 准确率两种方式
#     l1.append(accuracy_score(lrl1.predict(train_x),train_y))
#     l1test.append(accuracy_score(lrl1.predict(test_x),test_y))
#     l2.append(accuracy_score(lrl2.predict(train_x),train_y))
#     l2test.append(accuracy_score(lrl2.predict(test_x),test_y))
    
    l1.append(lrl1.score(train_x,train_y))
    l1test.append(lrl1.score(test_x,test_y))
    l2.append(lrl2.score(train_x,train_y))
    l2test.append(lrl2.score(test_x,test_y))
    
graph=[l1,l2,l1test,l2test]
color=['green','black','lightgreen','gray']
label=["L1","L2","L1test","L2test"]

plt.figure(figsize=(6,6))
for i in range(len(graph)):
    plt.plot(np.linspace(0.05,1,19),graph[i],color[i],label=label[i])
plt.legend(loc=4) ## 图例的位置在哪里？4表示在右下角，具体可以用shift+tab的解释
plt.show()
```

<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200330103436540.png" alt="image-20200330103436540" style="zoom:50%;" />

L1、L2正则化效果差不多，随着C增加，正则化越来越弱，模型在训练集和测试集上的准确率都在上升，当C在0.8左右的时候，训练集的准确率继续升高但测试集开始下跌，出现了过拟合，故C设定在0.8或者0.9比较合适

## 2.3 特征工程

逻辑回归往往需要降维，方法如下：

- 业务选择

  先通过算法降维，再人工选择

- PCA和SVD一般不用

  PCA和SVD的降维结果不可解释，一旦通过此方法降维后，无法解释特征和标签的关系；

  若不需要探究特征和标签之间的关系，可用PCA和SVD

- 统计方法可以使用，但不是必要的

  降维方法不可用，特征选择方法可用
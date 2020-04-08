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

**class `sklearn.linear_model.LogisticRegression` (penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, verbose=0, warm_start=False, n_jobs=None)**

| 参数         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| penalty      | 可以输入"l1"或"l2"来指定使用哪一种正则化方式，不填写默认"l2"。 注意，若选择"l1"正则化，参数solver仅能够使用求解方式”liblinear"和"saga“，若使用“l2”正则 化，参数solver中所有的求解方式都可以使用。 |
| C            | C正则化强度的倒数，必须是一个大于0的浮点数，不填写默认1.0，即默认正则项与损失函数的 比值是1:1。C越小，损失函数会越小，模型对损失函数的惩罚越重，正则化的效力越强，参数$\theta$会逐渐被压缩得越来越小。 |
| max_iter     | 最大迭代次数，能走的最大步数                                 |
| solver       | 求解器，默认是“liblinear"                                    |
| multi_class  | 告知求解器处理的分类问题的类型，默认”ovr"(one-vs-rest)<br>'ovr': 二分类问题，或让模型使用“1对多”的形式处理多分类问题<br>"multinomial": ”Many-vs_many"，**多分类问题**，这种输入下，solver不能玩诶‘liblinear"<br>"auto": 根据数据的分类情况和其他参数自动确定模型要处理的分类问题的类型，如果数据是二分类，或者solver=’liblinear'，auto会默认选择‘ovr’，否则“multinomial"<br />注意：默认值在0.22版本从”ovr"更改为“auto" |
| class_weight | 对不均衡样本进行处理，给少量的标签更多的权重，让模型更加偏向少数类，默认None，即所有标签权重相同<br />若“balanced"，即对样本进行均衡处理 |
|              |                                                              |
| **属性**     | **说明**                                                     |
| .coef_       | 查看每个特征对应的参数                                       |
| .n_iter_     | 调用本次求解真正实现的迭代次数                               |
| .intercept_  | 返回截距                                                     |
| .classes_    | 返回分类器的标签                                             |
| **接口**     | **说明**                                                     |
|              |                                                              |

<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200408095905212.png" alt="image-20200408095905212" style="zoom:50%;" />



## 2.1 二元逻辑回归的损失函数

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

**逻辑回归往往需要通过特征选择，很少用降维算法，因为降维算法选出来的结果有不可解释性**

1、业务选择：先进行特征选择（如过滤法、嵌入法、包装法），再人工选择

2、降维算法如PCA和SVD一般不用：

PCA和SVD的降维结果不可解释，一旦通过此方法降维后，无法解释特征和标签的关系；若不需要探究特征和标签之间的关系，可用PCA和SVD

**备注：**多重共线性会影响线性模型的效果，对于线性回归而言，确实如此，需要用方差过滤和方差膨胀因子VIF（variance inflation factor）来消除共线性；但对于逻辑回归，不是很必要，有时还需要多一些相互关联的特征来增加模型的表现，但若训练过程中你感觉共线性影响了模型效果，可以试试用VIF消除共线性，但sklearn没有提供这个功能

<font color=#0000FF size=2 face="黑体">***特征选择方法的应用：***  </font>

1、所有过滤法如方差过滤、卡方顾虑、F值检验、互信息法都可以用

2、嵌入法使用

```python
#===========================高效嵌入法=======================
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

data=load_breast_cancer()
data.data.shape

LR_=LR(solver='liblinear',C=0.9,random_state=420)
cross_val_score(LR_,data.data,data.target,cv=10).mean() #95.09%

x_embedded=SelectFromModel(LR_,norm_order=1).fit_transform(data.data,data.target)
# threshold未设置，则只根据L1正则化的结果选择特征，即选择了L1正则化后参数不为0的特征
# 一旦调整threshold的值，就不再使用L1正则化选择特征，而是使用.coef_生成的各个特征的系数选择
x_embedded.shape #569*9

cross_val_score(LR_,x_embedded,data.target,cv=10).mean() #93.68%

#===========================调节SelectFromModel中的threshold=======================
fullx=[]
fsx=[]
threshold=np.linspace(0,abs((LR_.fit(data.data,data.target).coef_)).max(),20)
k=0
for i in threshold:
    x_embedded=SelectFromModel(LR_,threshold=i,norm_order=1).fit_transform(data.data,data.target)
    fullx.append(cross_val_score(LR_,data.data,data.target,cv=10).mean())
    fsx.append(cross_val_score(LR_,x_embedded,data.target,cv=10).mean())
    print(threshold[k],x_embedded.shape[1])
    k+=1
plt.figure(figsize=(20,5))
plt.plot(threshold,fullx,label='full')
plt.plot(threshold,fsx,label='feature selection')
plt.xticks(threshold)
plt.legend()
plt.show()

## 为保证较好的效果，需保证有17+个特征

## 进一步细化
fullx=[]
fsx=[]
threshold=np.linspace(0,0.106,20)
k=0
for i in threshold:
    x_embedded=SelectFromModel(LR_,threshold=i,norm_order=1).fit_transform(data.data,data.target)
    fullx.append(cross_val_score(LR_,data.data,data.target,cv=10).mean())
    fsx.append(cross_val_score(LR_,x_embedded,data.target,cv=10).mean())
    print(threshold[k],x_embedded.shape[1])
    k+=1
plt.figure(figsize=(20,5))
plt.plot(threshold,fullx,label='full')
plt.plot(threshold,fsx,label='feature selection')
plt.xticks(threshold)
plt.legend()
plt.show()

## 为保证较好的效果，需保证有27个特征，仅去除3个特征

#===========================调整LR的C=======================
fullx=[]
fsx=[]
c=np.arange(0.01,10.01,0.5)
for i in c:
    LR_=LR(solver='liblinear',C=i,random_state=420)
    x_embedded=SelectFromModel(LR_,norm_order=1).fit_transform(data.data,data.target)
    fullx.append(cross_val_score(LR_,data.data,data.target,cv=10).mean())
    fsx.append(cross_val_score(LR_,x_embedded,data.target,cv=10).mean())
    
print(max(fsx),c[fsx.index(max(fsx))])
plt.figure(figsize=(20,5))
plt.plot(c,fullx,label='full')
plt.plot(c,fsx,label='feature selection')
plt.xticks(c)
plt.legend()
plt.show()

# 进一步细化
fullx=[]
fsx=[]
c=np.linspace(6.01,8.01,40)
for i in c:
    LR_=LR(solver='liblinear',C=i,random_state=420)
    x_embedded=SelectFromModel(LR_,norm_order=1).fit_transform(data.data,data.target)
    fullx.append(cross_val_score(LR_,data.data,data.target,cv=10).mean())
    fsx.append(cross_val_score(LR_,x_embedded,data.target,cv=10).mean())
    
print(max(fsx),c[fsx.index(max(fsx))])
plt.figure(figsize=(20,5))
plt.plot(c,fullx,label='full')
plt.plot(c,fsx,label='feature selection')
plt.xticks(c)
plt.legend()
plt.show()

## 结果：0.9563164376458386 6.984358974358974

## 验证模型效果，降维前
LR_=LR(solver='liblinear',C=6.984358974358974,random_state=420)
cross_val_score(LR_,data.data,data.target,cv=10).mean()
# 0.9491454930429521

## 验证模型效果，降维后
LR_=LR(solver='liblinear',C=6.984358974358974,random_state=420)
x_embedded=SelectFromModel(LR_,norm_order=1).fit_transform(data.data,data.target)
cross_val_score(LR_,x_embedded,data.target,cv=10).mean()
# 0.9563164376458386
```

3、系数累加法

可以使用.coef_画出曲线，选择曲线的折点，转折点之前被累加的特征都是我们需要的

需要先对特征系数进行从大到小的排序，并确保排序后每个系数对应的原始特征的位置，从而找到重要的特征，比较麻烦

```python

```

4、包装法使用

当需要直接设定特征个数时

```python
#===========================包装法=======================
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE  ## 包装法

data=load_breast_cancer()
data.data.shape

LR_=LR(solver='liblinear',random_state=420)
selector=RFE(LR_,n_features_to_select=15,step=5).fit(data.data,data.target) # RFE实例化
selector.support_ # 返回所有特征是否被选中的不二矩阵

x_rfe=selector.transform(data.data)
cross_val_score(LR_,x_rfe,data.target,cv=10).mean()

#=============学习曲线选合适的特征个数
score=[]
LR_=LR(solver='liblinear',random_state=420)
for i in range(1,29):
    selector=RFE(LR_,n_features_to_select=i,step=1).fit(data.data,data.target) # RFE实例化
    x_rfe=selector.transform(data.data)
    score.append(cross_val_score(LR_,x_rfe,data.target,cv=10).mean())
print(max(score),score.index(max(score)))
plt.figure(figsize=(20,5))
plt.plot(range(1,29),score)
plt.xticks(range(1,29))
plt.legend()
plt.show()

## 仍然是27个特征，只能去除3个，没啥效果，需要调整LR的C
```

## 2.4 梯度下降法求解逻辑回归

**梯度**：在多元函数上，对各个自变量求$\partial$偏导数，把求得的各个自变量的偏导数以向量的形式写出来，逻辑回归的自变量是$[\theta_1,\theta_2,...,\theta_n]$，梯度向量$d=[\frac {\partial J}{\partial \theta_1},\frac {\partial J}{\partial \theta_2},...,\frac {\partial J}{\partial \theta_n}]^T$，简称$grad J(\theta_1,\theta_2,...,\theta_n)$或者$\nabla J(\theta_1,\theta_2,...,\theta_n)$，梯度向量$d$的方向是损失函数$J(\theta)$的值增加最快的方向，即只要沿着梯度向量的<font color=#0000FF>*反方向*</font>移动坐标，损失函数$J(\theta)$的取值就会减少的最快，也就最容易找到损失函数的最小值

**梯度下降：**在众多的$[\theta_1,\theta_2,...,\theta_n]$可能的值中遍历，一次次求解坐标点的梯度向量，不断让损失函数$J$逐渐逼近最小值，在返回这个最小值对应的参数取值$[\theta_1^*,\theta_2^*,...,\theta_n^*]$

**步长$\alpha$：**不是任何物理距离，不是梯度下降过程中任何距离的直接变化，它是梯度向量的大小$d$的一个比例，影响着参数向量$\theta$每次迭代后改变的部分，调节梯度下降的速度；若步长太大，需要的迭代次数就很少，但梯度下降过程可能跳过损失函数的最低点，无法获得最低值；若步长过小，虽然函数会逐渐逼近我们需要的最低点，但速度缓慢，迭代次数需要很多；在sklearn中通过max_iter控制步长大小

<font color=#0000FF size=2>***具体推导：***  </font>

逻辑回归的损失函数如下：
$$
J(\theta)=-\sum_{i=1}^m(y_i*log(y_\theta(x_i))+(1-y_i)*log(1-y_\theta(x_i)))
$$


对自变量$\theta$求偏导：					
$$
\frac {\partial}{\partial\theta_j}J(\theta)=d_j=\sum_{i=1}^m(y_\theta(x_i)-y_i)x_{ij}
$$
遍历$\theta$的过程可以描述为：
$$
\theta_{j+1}=\theta_j-\alpha*d_j=\theta_j-\alpha*\sum_{i=1}^m(y_\theta(x_i)-y_i)x_{ij}
$$
$\alpha$被称为步长，控制着每迭代一次后$\theta$的变化，并一次来影响每次迭代后的梯度向量的大小和方向，

```python
#=============max_iter的学习曲线====================================================
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data=load_breast_cancer()
data.data.shape

l2=[]
l2test=[]

train_x,test_x,train_y,test_y=train_test_split(data.data,data.target,test_size=0.3,random_state=420)

for i in np.arange(1,201,10):
    LR_=LR(penalty="l2",solver='liblinear',C=6.984358974358974,max_iter=i).fit(train_x,train_y)
    l2.append(accuracy_score(LR_.predict(train_x),train_y))
    l2test.append(accuracy_score(LR_.predict(test_x),test_y))
    
graph=[l2,l2test]
color=["black","red"]
label=["L2","L2test"]

plt.figure(figsize=(20,5))
for i in range(len(graph)):
    plt.plot(np.arange(1,201,10),graph[i],color[i],label=label[i])
plt.legend()
plt.xticks(np.arange(1,201,10))
plt.show()

## max_iter=21时效果比较好，红色的警告（没有收敛）可以忽略

lr=LR(penalty="l2",solver="liblinear",C=6.984358974358974,max_iter=300).fit(train_x,train_y)
lr.n_iter_
## 只迭代了29次，虽然最大次数设置为300
```

红色警告可忽略<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200407214311586.png" alt="image-20200407214311586" style="zoom:50%;" />

## 如何处理 二元回归与多元回归问题？

sklearn中通过solver和multi_class来处理逻辑回归多分类问题

”OvR"：当把某种分类类型看作1，其他看作0，“一对多” One-vs-rest，在sklearn中未“ovr”

“MvM"：把好几个分类类型划为1，其他看作0，”多对多“ Many-vs-Many，简称”MvM"，在sklearn中看作“Multinominal"

**逻辑回归还是把多元回归转化成了二元回归处理**

```python
#=================multinomial和ovr的差别===========================
from sklearn.datasets import load_iris  ## 三分类数据集
from sklearn.linear_model import LogisticRegression as LR
iris=load_iris()  
for multi_class in ('multinomial','ovr'):
    clf=LR(solver='sag',max_iter=1000,random_state=420,multi_class=multi_class).fit(iris.data,iris.target)
    print("training score: %.3f (%s)" %(clf.score(iris.data,iris.target),multi_class))
    
## 分别为0.980，0.953，多元表现更好
LR(solver='sag',max_iter=1000,random_state=420,multi_class="multinomial").fit(iris.data,iris.target).classes_
```

## 如何处理 样本不平衡问题？

**样本不平衡：**在一组数据集中，标签的一类天生占有很大的比例或者误分类的代价很高（即我们想要捕捉出某种特定的分类的时候）

**什么时候误分类代价很高？**

- 对潜在犯罪者和普通人进行分类 
- 银行中判断新客户是否违约

<font color=#0000FF size=2>**类别分布不均衡的方法论：**</font>

![image-20200408182408283](/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200408182408283.png)



**过采样示例：SMOTE算法**

```python
import pandas as pd
from imblearn.over_sampling import SMOTE       #过度抽样处理库SMOTE
df=pd.read_table('data2.txt',sep=' ',names=['col1','col2','col3','col4','col5','label'])    
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
groupby_data_orginal=df.groupby('label').count()      #根据标签label分类汇总

model_smote=SMOTE()    #建立smote模型对象
x_smote_resampled,y_smote_resampled=model_smote.fit_sample(x,y)
x_smote_resampled=pd.DataFrame(x_smote_resampled,columns=['col1','col2','col3','col4','col5'])
y_smote_resampled=pd.DataFrame(y_smote_resampled,columns=['label'])
smote_resampled=pd.concat([x_smote_resampled,y_smote_resampled],axis=1)
groupby_data_smote=smote_resampled.groupby('label').count()
```

**欠采样示例：**

```python
from imblearn.under_sampling import RandomUnderSampler 
model_RandomUnderSampler=RandomUnderSampler()                #建立RandomUnderSample模型对象
x_RandomUnderSample_resampled,y_RandomUnderSample_resampled=model_RandomUnderSampler.fit_sample(x,y)         #输入数据并进行欠抽样处理
x_RandomUnderSample_resampled=pd.DataFrame(x_RandomUnderSample_resampled,columns=['col1','col2','col3','col4','col5'])
y_RandomUnderSample_resampled=pd.DataFrame(y_RandomUnderSample_resampled,columns=['label'])
RandomUnderSampler_resampled=pd.concat([x_RandomUnderSample_resampled,y_RandomUnderSample_resampled],axis=1)
groupby_data_RandomUnderSampler=RandomUnderSampler_resampled.groupby('label').count()
```

**在逻辑回归中，遇到不平衡样本怎么办？**

- 可以设置class_weight='balanced'，给少数类更多的权重，给多数类更少的权重，默认是None，即少数类和多数类的权重一样，不建议
- 过采样是最好的办法

# 3、案例：评分卡模型


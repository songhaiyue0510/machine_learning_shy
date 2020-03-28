[TOC]

# **1、朴素贝叶斯**

属性条件独立性假设：假设所有属性相互独立，换言之，假设每个属性独立地对分类结果发生影响
$$
P(c|x)=\frac {P(c)P(x|c)}{P(x)}=\frac {P(c)}{P(x)}\prod_{i=1}^{d}P(x_{i}|c)
$$

对于所有类别，$P(x)$都相同，所以贝叶斯判定准则有:
$$
h_{nb}(x)=\mathop {argmax}_{c\in y}P(c)\prod_{i=1}^{d}P(x_{i}|c) \\且 \ P(c)=\frac{|D_{c}|}{|D|}
$$

对离散属性而言，
$$
P(x_{i}|c)=\frac{|D_{c,x_{i}}|}{|D_{c}|}
$$

对连续属性而言，
$$
P(x_{i}|c)=\frac {1}{\sqrt{2\pi}\sigma_{c,i}}exp(-\frac{(x_{i}-\mu_{c,i})^2}{2\sigma_{c,i}^2})
$$



----

# **2、sklearn中的贝叶斯算法**

## **2.1 高斯朴素贝叶斯算法**

假设：连续属性$P(x_{i}|c)$服从高斯分布$P(x_{i}|c)\sim N(\mu_{c,i},\sigma_{c,i}^2)$
$$
P(x_{i}|c)=\frac {1}{\sqrt{2\pi}\sigma_{c,i}}exp(-\frac{(x_{i}-\mu_{c,i})^2}{2\sigma_{c,i}^2})
$$

数据集：月亮型数据、二分数据表现好，在环形数据集表现不太好

高斯朴素贝叶斯的决策边界是弧形的

Naive Bayes的分类效果不好，样本少训练准确度比较好

过拟合，即训练集比测试集效果好很多，

1. 当样本量少时，都有过拟合；样本量大时，过拟合减弱

2. 决策树是天生过拟合的模型，可以通过剪枝或者增大样本量来进一步减少过拟合

3. Naive Bayes的测试集、训练集的准确率低，速度快（高维数据可以考虑），过拟合不是主要问题

----

## **2.2 多项式朴素贝叶斯算法**

假设：概率服从简单的多项式分布

擅长分类型变量，$P(x_{i}|c)$的概率是离散的，并且不同$x_{i}$下的$P(x_{i}|c)$相互独立，互不影响，sklearn不接受负值输入，得到的矩阵长为稀疏矩阵，常用于文本分类
$$
P(x_{i}|c)=\frac{|D_{c,x_{i}}|}{|D_{c}|}
$$
对于一个在标签类别$Y=c$下，结构为（m,n）的特征矩阵来说，我们有：
$$
X_{y}=\left[\begin{array}\\
x_{11}&x_{12}&x_{13}\cdots&x_{1n}\\
x_{21}&x_{22}&x_{23}\cdots&x_{2n}\\
\ &\ &\cdots&\ \\
x_{m1}&x_{m2}&x_{m3}\cdots&x_{mn}\\
\end{array}\right]
$$
其中每个特征$x_{ji}$都是特征$X_i$发生的次数，因此，平滑后的最大似然估计为
$$
D_{c,xi}=\frac {\sum_{{y_j}=c}x_{ji}+\alpha}{\sum_{i=1}^n\sum_{{y_j}=c}x_{ji}+\alpha n}
$$
**$\alpha$为平滑系数**，防止训练数据中出现过的一些词未出现在测试集中导致的0概率，当$\alpha=1$即拉普拉斯平滑，当$\alpha<1$即利德斯同平滑，是自然语言处理中常用的平滑分类数据的统计手段

在sklearn中，sklearn.naive_bayes.MultinomialNB(alpha=1.0,fit_prior=True,class prior=None)

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss

import numpy as np
import pandas as pd

# 准备数据
class_1=500
class_2=500
centers=[[0.0,0.0],[2.0,2.0]]
cluster_std=[0.5,0.5]
x,y=make_blobs(n_samples=[class_1,class_2],
              centers=centers,
              cluster_std=cluster_std,
              random_state=0,
              shuffle=False)

# 训练数据
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=420)

# 归一化，train_x_仍然是连续型变量
mms=MinMaxScaler().fit(train_x) # 特征矩阵归一化
train_x_=mms.transform(train_x)
test_x_=mms.transform(test_x)

# 重要属性1：调用根据数据获取的，每个标签类的对数先验概率log(P(Y))，由于概率在[0,1]之间，因此对数先验概率永远是负值
# 先验概率接近，说明不存在样本不均衡问题
mnb.class_log_prior_          ## 先验概率的log值
mnb.class_log_prior_.shape    ## 等于标签中的类别数量
np.exp(mnb.class_log_prior_)  ## 真正的先验概率值

# 重要属性2：返回一个固定标签类别下的每个特征的对数概率log(P(Xi|y))
mnb.feature_log_prob_          ## 2个特征，2个标签
mnb.feature_log_prob_.shape    ## 
np.exp(mnb.feature_log_prob_)  ## 

# 重要属性3：在fit时每个标签类别下的样本数，当fit接口的sample_weight被设置时，该接口返回的值也会受到加权的影响
mnb.class_count_          ## 
mnb.class_count_.shape    ## 返回和标签类别一样的结构

# 验证效果，效果不理想，因为输入变量是连续型变量，多项式贝叶斯适用于分类型变量
mnb.predict(test_x_)
mnb.predict_proba(test_x_) # 每个样本在每个标签下的概率
mnb.score(test_x_,test_y)
brier_score_loss(test_y,mnb.predict_proba(test_x_)[:,1],pos_label=1)
```

分类效果不理想，由于输入变量是连续型变量，将其转化成分类型变量，效果惊人

```python
# 将连续型输入改成分类型数据(哑变量），无需归一化
from sklearn.preprocessing import KBinsDiscretizer # 对连续性变量进行分箱
kbs=KBinsDiscretizer(n_bins=10,encode='onehot').fit(train_x)

train_x_=kbs.transform(train_x)
test_x_=kbs.transform(test_x)
mnb=MultinomialNB().fit(train_x_,train_y)
print(mnb.score(test_x_,test_y))
print(brier_score_loss(test_y,mnb.predict_proba(test_x_)[:,1],pos_label=1))
```

---

## **2.3 伯努利朴素贝叶斯**

   是多项式朴素贝叶斯的一种特殊形式， 用来处理二项分布，假设数据服从多元伯努利分布，即存在多个特征，**但每个特征都是二分类的**，可以用不二变量表示为{0,1}{-1,1}等二分类组合，**若数据不是二分类，可以用二值化参数binarize改变数据**

​    此算法在意的是事件***“是否发生”***，常用来处理文本分类数据

sklearn.naive_bayes.BernoulliNB(alpha=1.0,binarize=0.0,fit_prior=True,class_prior=None)

| 参数      | 含义                                                         |
| --------- | ------------------------------------------------------------ |
| alpha     | 浮点数，可不填(默认1.0），alpha=0则无平滑选项，alpha越大，精确性月底 |
| binarize  | 浮点数，可不填（默认0），设定二值化的阈值（应用于所有特征），若设定为None，则假定特征已经二值化 |
| fit_prior | 布尔值，可不填（默认True），是否学习先验概率P(Y=c)，若为false，则不用先验概率，而使用统一的先验概率（uniform prior），即每个标签的概率1/n_classes |
| n_classes | 类似数组结构（n_classes, )，可不填（默认None），类的先验概率P(Y=c)，若未给出可根据数据进行计算 |

```python
## 模型
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

## 数据集处理
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 数据集
from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss

## 基础包
import numpy as np
import pandas as pd
## 准备数据
class_1=500
class_2=500
centers=[[0.0,0.0],[2.0,2.0]]
cluster_std=[0.5,0.5]
x,y=make_blobs(n_samples=[class_1,class_2],
              centers=centers,
              cluster_std=cluster_std,
              random_state=0,
              shuffle=False)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=420)

# 普通来说，应该用二值化的类sklearn.preprocessing.Binarizer来将特征一个个二值化，
# 这样效率低，故这里采用归一化后直接设置阈值的方式
mms=MinMaxScaler().fit(train_x) # 特征矩阵归一化
train_x_=mms.transform(train_x)
test_x_=mms.transform(test_x)
# 不设置二值化
bn1_=BernoulliNB().fit(train_x_,train_y)
print(bn1_.score(test_x_,test_y))
print(brier_score_loss(test_y,bn1_.predict_proba(test_x_)[:,1],pos_label=1))
# 设置二值化
bn1_=BernoulliNB(binarize=0.5).fit(train_x_,train_y)
print(bn1_.score(test_x_,test_y))
print(brier_score_loss(test_y,bn1_.predict_proba(test_x_)[:,1],pos_label=1))
```

---

## **2.4 补集朴素贝叶斯**

标准多项式朴素贝叶斯算法的改进，**能够解决样本不均衡问题、不在乎特征之间是否条件独立**，同样擅长处理**分类型数据**

[详细数学原理和证明过程](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)

CNB是来自每个标签类别的补集的概率，并以此来计算每个特征的权重
$$
D_{x_i,y≠c}=\frac{\alpha_i+\sum_{y_i≠c}x_{ij}}{\alpha_in+\sum_{i,y≠c}\sum_{i=1}^{n}x_{ij}}
$$
其中j表示每个样本，$x_{ij}$表示在样本$j$上对于特征$i$的取值，在文本中通常是计数或者TF-IDF值，$\alpha$是平滑系数

其实，$\sum_{y_j≠c}x_{ij}$是一个特征$i$下所有标签类别不等于$c$的样本的特征值之和，$\sum_{i,y≠c}\sum_{i=1}^{n}x_{ij}$所有特征下所有标签类别不等于$c$的样本的特征取值之和，特征值为：
$$
w_{ci}=logD_{x_i,y≠c}\\
或者可以选择\\
w_{ci}=\frac{logD_{x_i,y≠c}}{\sum_j|logD_{x_i,y≠c}|}
$$
对数后得到权重，或者除以L2范式，解决多项式分布中，特征取值比较多的样本支配参数估计的情况，即一个样本可能在很多个特征下都有取值，基于这个权重，朴素贝叶斯的一个样本的预测规则未：
$$
p(Y≠c|X)=arg\,\min_{c}\sum_{i}x_iw_{ci}
$$
*即最小补集概率对应的标签就是样本的标签*

Sklearn.naive_bayes.ComplementNB(alpha=1.0,fit_prior=True,class_prior=None,nomr=False)

| 参数      | 含义                                                         |
| --------- | ------------------------------------------------------------ |
| alpha     | 浮点数，可不填(默认1.0），alpha=0则无平滑选项，alpha越大，精确性月底 |
| norm      | 布尔值数，可不填（默认false）<br> 计算权重时是否使用L2范式进行规范，默认不规范，希望规范设置为True |
| fit_prior | 布尔值，可不填（默认True），是否学习先验概率P(Y=c)，若为false，则不用先验概率，而使用统一的先验概率（uniform prior），即每个标签的概率1/n_classes |
| n_classes | 类似数组结构（n_classes, )，可不填（默认None），类的先验概率P(Y=c)，若未给出可根据数据进行计算 |

---

## **2.5 探索贝叶斯：贝叶斯的样本不均衡问题**

探索贝叶斯算法在不均衡样本上的表现

```python
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB,ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer ## 分箱
from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss,recall_score,roc_auc_score
from time import time
import datetime
## 准备数据, 
class_1=50000 # 多数类为500000个样本，负例
class_2=500 # 少数类500个样本,正例
centers=[[0.0,0.0],[5.0,5.0]] # 设定两个类别的中心
cluster_std=[3,1] # 设定两个类别的方差，通常样本多则方差大，样本少则方差小
x,y=make_blobs(n_samples=[class_1,class_2], # 分别有class_1个负例，class_2个正例
              centers=centers,
              cluster_std=cluster_std,
              random_state=0,
              shuffle=False)

# 查看贝叶斯在样本不均衡上的表现
name=["Multinomial","Gaussian","Bernoulli","Complement"]
models=[MultinomialNB(),GaussianNB(),BernoulliNB(),ComplementNB()]  # 统一做成二分类

for name,clf in zip(name,models):
    times=time()
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=420)
    # 预处理，2个特征分10箱，且onehot处理后则20个特征都是二分类
    if name!="Gaussian":
        kbs=KBinsDiscretizer(n_bins=10,encode='onehot').fit(train_x)
        train_x=kbs.transform(train_x)
        test_x=kbs.transform(test_x)
    # 拟合   
    clf.fit(train_x,train_y)
    pred_y=clf.predict(test_x)
    proba=clf.predict_proba(test_x)[:,1]
    score=clf.score(test_x,test_y)  # 准确率：预测正确的样本比例，在不均衡样本中使用有问题
    brier=brier_score_loss(test_y,proba,pos_label=1)
    recall=recall_score(test_y,pred_y) # 召回率：正样本中有多少被预测正确
    auc=roc_auc_score(test_y,proba) # 对样本是否均衡不敏感
    print(name,":")
    print("Brier loss score:{:.3f}".format(brier))
    print("Accuracy:{:.3f}".format(score))
    print("Recall:{:.3f}".format(recall))
    print("Auc:{:.3f}".format(auc))
    print("time cost:{}\n".format(datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f")))
```

<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200327113839120.png" alt="image-20200327113839120" style="zoom:40%;" />

---

# **3、概率分类器的评估指标**

## **3.1 布里尔分数Brier Score**

**"校准程度"**：概率预测的准确程度，衡量算法预测出来的概率与真是结果的差异
$$
Brier Score=\frac {1}{N}\sum_{i=1}^n(p_i-o_i)^2
$$
其中$N$是样本数量，$p_i$是朴素贝叶斯预测出来的概率，$o_i$是样本真实的结果（只能为0或1）

Brier Score分数的范围0~1，**分数越高预测结果越差，校准程度越差**

```python
from sklearn.metrics import brier_score_loss
# 第一个参数：真实标签，第二个参数：预测出的概率值
# 二分类情况下，predict_proba 会返回两列，分别为2个类别(0,1)的概率
# SVC的接口decision_function只有一列
# pos_label=1表示正样本，与prob的索引一致
# test_y只能为0、1变量
brier_score_loss(test_y,mnb.predict_proba(test_x_)[:,1],pos_label=1)
```

## **3.2 对数似然函数Log Loss**

***只能用于评估分类型模型***，取值越小，概率估计越准确，模型越理想

假设样本的真实标签$y_{trure}$在{0,1}中取值，在类别1下的概率估计为$y_{pred}$，log是以e为底的自然对数，对数损失为：
$$
-logP(y_{ture}|y_{pred})=-(y_{true}*log(y_{pred})+(1-y_{true})*log(1-y_{pred}))
$$
在sklearn中，

```python
from sklearn.metrics import log_loss
log_loss(Ytest,prob) ## prob是预测出的概率，不是样本分类
```

**什么时候使用对数似然，什么时候使用布里尔分数？**

对数似然是概率类模型的黄金指标，优先选择，缺点为：

- 没有界，不像布里尔分数有上限
- 解释性不如布里尔分数
- 在以最优化为目标的模型上表现更好
- 不能接受为0、1的概率，否则会取到极值

按照如下规则进行选择：

| 需求           | 优先使用对数似然                           | 优先使用布里尔分数                 |
| -------------- | :----------------------------------------- | ---------------------------------- |
| **衡量模型**   | 对比多个模型，或衡量模型的不同变化         | 衡量单一模型的表现                 |
| **可解释性**   | 机器学习和深度学习之间的行家交流，学术论文 | 商业报告、老板开会，业务模型的衡量 |
| **最优化指向** | 逻辑回归，SVC                              | 朴素贝叶斯                         |
| **数学问题**   | 概率只能无限接近0、1，无法取到0、1         | 概率可以取到0、1，比如树、随机森林 |

## **3.3 可靠性曲线（reliability curve)**

用来调节概率的校准程度，以预测概率为横坐标，真实标签为纵坐标，***结果越靠近y=x越好***，用于二分类情况比较多

利用**calibration_curve**获取横纵坐标

| 参数      | 含义                                                         |
| --------- | ------------------------------------------------------------ |
| y_true    | 真实标签                                                     |
| y_prob    | 预测返回，正类别的概率值                                     |
| normalize | 布尔值，默认False<br> 将y_prob归一化到[0,1]之间，若y_prob本身在[0,1]之间不用设置此参数 |
| n_bins    | 整数值，表示分箱的个数                                       |
| **返回**  | **含义**                                                     |
| trueproba | 可靠性曲线的纵坐标，结构为（n_bins, )，是每个箱子中少数类（Y=1)的占比 |
| predproba | 可靠性曲线的纵坐标，结构为（n_bins, )，是每个箱子中概率的均值 |

**1、建立数据集**

```python
# 基础包
import numpy as np
import pandas as pd

## sklearn相关模型包
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR

# 数据集
from sklearn.datasets import load_digits
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification as mc

# 数据集处理
from sklearn.model_selection import train_test_split

## 评估指标
from sklearn.metrics import brier_score_loss  ## 布里尔分数
from sklearn.metrics import log_loss  ## 对数似然函数

# 画图
import matplotlib.pyplot as plt
import seaborn as sns

x,y=mc(n_samples=100000,
       n_features=20, # 20个特征
       n_classes=2, # 二分类
       n_informative=2, # 其中有2个代表较多信息
       n_redundant=10, # 10个都是冗余特征
       random_state=42)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.99,random_state=42)
```

**2、利用calibration_curve**

```python
from sklearn.calibration import calibration_curve
trueproba,predproba=calibration_curve(test_y,prob_pos,n_bins=10)
fig=plt.figure()
ax1=plt.subplot()
ax1.plot([0,1],[0,1],'k:',label='perfectly calibrated') ## 对角线作为对比
ax1.plot(predproba,trueproba,'s-',label='%s (%1.3f)' %("Bayes",clf_score))
ax1.set_ylabel("True label")
ax1.set_xlabel("predicted probability")
ax1.set_ylim([-0.05,1.05])
ax1.legend()
plt.show()
```

<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200326094354723.png" alt="image-20200326094354723" style="zoom:30%;" />

**3、不同模型的可靠性曲线对比：**

```python
## 建立更多模型
name=["GaussianBayes","Logistic","SVC"]
gnb=GaussianNB()
logi=LR(C=1,solver='lbfgs',max_iter=30000,multi_class="auto")
svc=SVC(kernel="linear",gamma=1)

fig,ax1=plt.subplots(figsize=(8,6))
ax1.plot([0,1],[0,1],'k:',label='perfectly calibrated') ## 对角线作为对比
for clf,name_ in zip([gnb,logi,svc],name):
    clf.fit(train_x,train_y)
    pred_y=clf.predict(test_x)
    # hasattr(obj,name):查看一个类obj中是否存在名字为name的接口，存在则返回true
    if hasattr(clf,"predict_proba"):
        prob_pos=clf.predict_proba(test_x)[:,1]
    else:
        prob_pos=clf.decision_function(test_x)
        prob_pos=(prob_pos-prob_pos.min())/(prob_pos.max()-prob_pos.min())
    # 返回布里尔分数
    clf_score=brier_score_loss(test_y,prob_pos,pos_label=y.max())
    trueproba,predproba=calibration_curve(test_y,prob_pos,n_bins=10)
    ax1.plot(predproba,trueproba,'s-',label='%s (%1.3f)' %(name_,clf_score))
ax1.set_ylabel("True probability for class1")
ax1.set_xlabel("Mean predicted probability")
ax1.set_ylim([-0.05,1.05])
ax1.legend()
ax1.set_title("Calibration plots(reliablity curve)")
plt.show()
```

<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200326102027110.png" alt="image-20200326102027110" style="zoom:30%;" />

- 逻辑回归效果最好
- 贝叶斯呈现和sigmoid函数相反的形状，***说明数据集中的特征不是相互独立的***，违背了贝叶斯的“朴素”原则（样本中有10个冗余特征）
- 支持向量机的曲线效果，***说明分类器置信度不足***，大量的样本在决策边界的附近，即便决策边界能够将样本点判断正确，模型本身对这个结果也不是非常确信的（支持向量机在面对混合度较高的数据时，有着天生的置信度不足的缺点）

***利用直方图查看预测概率的分布，是否如上所说***

```python
## 建立更多模型
name=["GaussianBayes","Logistic","SVC"]
gnb=GaussianNB()
logi=LR(C=1,solver='lbfgs',max_iter=30000,multi_class="auto")
svc=SVC(kernel="linear",gamma=1)

fig,ax2=plt.subplots(figsize=(8,6))

for clf,name_ in zip([gnb,logi,svc],name):
    clf.fit(train_x,train_y)
    pred_y=clf.predict(test_x)
    # hasattr(obj,name):查看一个类obj中是否存在名字为name的接口，存在则返回true
    if hasattr(clf,"predict_proba"):
        prob_pos=clf.predict_proba(test_x)[:,1]
    else:
        prob_pos=clf.decision_function(test_x)
        prob_pos=(prob_pos-prob_pos.min())/(prob_pos.max()-prob_pos.min())
    ax2.hist(prob_pos
            ,bins=10
            ,label=name_
            ,histtype="step" # 设置直方图为透明
            ,lw=2) # 设置直方图每个柱子描边的粗细
ax2.set_ylabel("Distribution of probability")
ax2.set_xlabel("Mean predicted probability")
ax2.set_xlim([-0.05,1.05])
ax2.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
ax2.legend()
plt.show()
```

<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200326111300233.png" alt="image-20200326111300233" style="zoom:30%;" />

- SVM: 概率集中在0.5附近，中间高两边低，证明了大部分样本都在决策边界附近，置信度在0.5%左右
- NB：两边高中间低，集中在0、1附近，置信度过高
- LR：位于二者中间

**使用近似回归来矫正算法**即使可靠性曲线尽可能接近对角线，校准可靠性曲线

1. 基于Platt的sigmoid模型的校准方法
2. 基于等渗回归（isotonic calibration)的非参数校准方法

两种方法都发生在训练集上，CalibratedClassifierCV(base_estimator=None,method='sigmoid',cv='warn')

** CalibratedClassifierCV没有借口decision_function，查看这个类校准果果的模型生成概率用predict_proba接口

| 参数           | 含义                                                         |
| -------------- | ------------------------------------------------------------ |
| base_estimator | 需要校准器输出决策功能的分类器，必须存在predict_proba或decision_function接口<br> 若cv=prefit，分类器必须已经拟合数据完毕（很少用） |
| cv             | 整数，确定交叉验证的策略，可能输入是：<br> * None，默认3折交叉验证<br> * 任意整数，若为2分类，自动使用sklearn.model_selection.StratifiedFold进行折数分割；若y是连续变量，则使用sklearn.model_selection.KFold进行分割<br> * 使用其他类建好的交叉验证模式或生成器cv<br> * 可跌打的，已经分割完毕的测试集和训练集索引数组<br> * 输入“prefit”，则假设已经在分类器上拟合完毕数据 |
| method         | 校准方法，“sigmoid"或者”isotonic"，样本量少时建议sigmoid，isotonic倾向于过拟合 |

```python
## 校准模型
def plot_calib(models,name,train_x,test_x,train_y,test_y,n_bins=10):
    import matplotlib.pyplot as plt
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve
    
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,6))
    ax1.plot([0,1],[0,1],'k:',label='Perfectly calibrated')
    
    for clf,name_ in zip(models,name):
        clf.fit(train_x,train_y)
        pred_y=clf.predict(test_x)
        # hasattr(obj,name):查看一个类obj中是否存在名字为name的接口，存在则返回true
        if hasattr(clf,"predict_proba"):
            prob_pos=clf.predict_proba(test_x)[:,1]
        else:
            prob_pos=clf.decision_function(test_x)
            prob_pos=(prob_pos-prob_pos.min())/(prob_pos.max()-prob_pos.min())
        clf_score=brier_score_loss(test_y,prob_pos,pos_label=y.max())
        score=clf.score(test_x,test_y)
        trueproba,predproba=calibration_curve(test_y,prob_pos,n_bins=10)
        ax1.plot(predproba,trueproba,'s-',label='%s (brier=%1.3f)(accuracy=%1.3f)' %(name_,clf_score,score))
        ax2.hist(prob_pos,bins=10,label=name_,histtype="step",lw=2) 
    ax1.set_ylabel("True probability for class1")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylim([-0.05,1.05])
    ax1.legend()
    ax1.set_title("Calibration plots(reliablity curve)")

    ax2.set_ylabel("Distribution of probability")
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_xlim([-0.05,1.05])
    ax2.set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    ax2.legend()

    plt.show()
# gnb校准
from sklearn.calibration import CalibratedClassifierCV
name=['GaussianBayes',"logistic",'Bayes+isotonic','bayes+sigmoid']

gnb=GaussianNB()
models=[gnb
       ,LR(C=1,solver='lbfgs',max_iter=30000,multi_class="auto")
       ,CalibratedClassifierCV(gnb,cv=2,method='isotonic')
       ,CalibratedClassifierCV(gnb,cv=2,method='sigmoid')]

plot_calib(models,name,train_x,test_x,train_y,test_y)

# svm校准
from sklearn.calibration import CalibratedClassifierCV
name=['SVC',"logistic",'SVC+isotonic','SVC+sigmoid']

svc=SVC(kernel="linear",gamma=1)
models=[svc
       ,LR(C=1,solver='lbfgs',max_iter=30000,multi_class="auto")
       ,CalibratedClassifierCV(svc,cv=2,method='isotonic')
       ,CalibratedClassifierCV(svc,cv=2,method='sigmoid')]

plot_calib(models,name,train_x,test_x,train_y,test_y)
```

<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200326163909506.png" alt="image-20200326163909506" style="zoom:50%;" />

<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200326164618267.png" alt="image-20200326164618267" style="zoom:50%;" />

对于支持向量机，校准模型效果较好，**概率校正对于原本的可靠性曲线是类似sigmoid形状的曲线的算法比较有效**

对于朴素贝叶斯，isotonic的校准方式使可靠性曲线效果更好，但校准后模型的准确度降低了，布里尔分数明显变小，二者趋势相反，为什么？

<span style='color:red'>**二者相悖时，以准确率为标准**</span>

1. 第一种可能性：

   对于SVC、决策树等模型，概率不是真正的概率，更偏向于置信度，且分类不是依据概率预测（决策树依据树杈，SVC依据边界）

   因此会出现类别1的概率为0.4但仍然被分类为1，这种情况代表着模型很没有信息认为这是1

   这种时候，概率校准可能会向着更加错误的方向调整（如把概率0.4的点调节的更接近0，导致模型判断错误）

2. 第二种可能性：

    对于朴素贝叶斯，由于各种各样的假设存在，使概率估计其实是有偏估计，校准后，预测概率更贴近真实概率，本质是在统计学上让算法更加贴近整体样本状况的估计

   测试集对估计的真是样本的贴近程度，会导致校准后可能准确率上升后者下降

3. 其他可能性：

   - 概率校准过程中的数学细节应县了校准，calibration_curve中如何分箱，真实标签和预测值如何生成校准曲线的横纵坐标等

   **现实中，根据需求追求最高的准确率(accuracy) 或者概率拟合最好（brier score loss)，通常是追求最高的准确率和recall，则可以考虑逻辑回归**

   

# 4、案例：贝叶斯做文本分类

## 3.1 文本编码简介：单词计数向量 和 TF-IDF算法

**单词计数向量：**将文本编码成数字，每个特征下的数字代表单词在样本中出现了几次，**是离散的、代表次数、正整数**

**TF-IDF**(term frequency-inverse document frequency)，词频你文档频率，通过单词在文档中出现的频率来衡量其权重，IDF的大小与一个词的常见程度成反比，越常见，编码后设置的权重越小，以此压制频繁出现的无意义的词

```python
from sklearn.feature_extraction.text import CountVectorizer as CV # 单词计数向量
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF # TF-IDF算法
import pandas as pd

sample=["Machine learning is fascinatin, it is wonderful",
       "Machine learning is a sensational techonology",
       "Elsa is a popular character"]

## 单词计数向量
vec=CountVectorizer()
x=vec.fit_transform(sample) # 得到 3X11的稀疏矩阵
x
vec.get_feature_names() # 按照字母的顺序排列，x.get_feature_names()不对，一个矩阵肯定没有接口
# 稀疏矩阵无法输入pandas
CVresult=pd.DataFrame(x.toarray(),columns=vec.get_feature_names())
CVresult
TFIDFresult.sum(axis=0)/TFIDFresult.sum(axis=0).sum()

## TF-IDF算法
vec=TFIDF()
x=vec.fit_transform(sample) # 得到 3X11的稀疏矩阵
x
vec.get_feature_names()# 按照字母的顺序排列，x.get_feature_names()不对，一个矩阵肯定没有接口
TFIDFresult=pd.DataFrame(x.toarray(),columns=vec.get_feature_names())
TFIDFresult

TFIDFresult.sum(axis=0)/TFIDFresult.sum(axis=0).sum()
```

<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200327183139414.png" alt="image-20200327183139414" style="zoom:50%;" />

<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200327183017283.png" alt="image-20200327183017283" style="zoom:40%;" />



**单词计数向量法出现两个问题：**

1. 样本不均衡问题

   根据多项式朴素贝叶斯的计算公式
   $$
   D_{c,xi}=\frac {\sum_{{y_j}=c}x_{ji}+\alpha}{\sum_{i=1}^n\sum_{{y_j}=c}x_{ji}+\alpha n}
   $$
   将每一列嘉禾，除以整个特征矩阵的和，就是每一列对应的概率$D_{c,x_i}$，由于是对$x_{ji}$进行加和，对于在很多特征都有值的样本而言，其对$D_{c,x_i}$贡献更大，因此补集朴素贝叶斯引入L2范式的权重

2. 常用词频繁出现占有很大权重

   比如“is"虽然出现了4次，但对语义的并没有什么影响，会误导算法，采用TF-IDF方法

## 3.2 案例

```python
from sklearn.datasets import fetch_20newsgroups #20个新闻组的语料库，英文
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF # TF-IDF算法
from sklearn.naive_bayes import MultinomialNB,ComplementNB,BernoulliNB
from sklearn.metrics import brier_score_loss,recall_score,roc_auc_score 
from sklearn.calibration import CalibratedClassifierCV # 概率校准

# 数据，通过参数提取数据
categories=["sci.space","rec.sport.hockey","talk.politics.guns","talk.politics.mideast"]
train=fetch_20newsgroups(subset='train',categories=categories) 
test=fetch_20newsgroups(subset='test',categories=categories) 
# 初次运行，需要下载
# fetcg_20gnewsgroups是一个类，有接口可以调用

train # 类字典结果，键值对结构
# 查看共有多少文章
len(train.data)
# 查看一篇文章，比较乱
train.data[0]
# 打印一下，比较好看
print(train.data[0])
np.unique(train.target)

# 是否存在样本不均衡问题
for i in [0,1,2,3]:
    print(i,(train.target==i).sum()/len(train.target))
    
 # 使用TF-IDF编码
train_x,test_x,train_y,test_y=train.data,test.data,train.target,test.target
tfidf=TFIDF().fit(train_x)
train_x_=tfidf.transform(train_x)  ## 特征按字母排序
test_x_=tfidf.transform(test_x)
train_x_

## 看看结果
tosee=pd.DataFrame(train_x_.toarray(),columns=tfidf.get_feature_names())
tosee.head()

## 建模
name=["Multinomial"
      ,"Multinomial+isotonic"
      ,"Multinomial+sigmoid"
      ,"Complement"
      ,"Complement+isotonic"
      ,"Complement+sigmoid"
      ,"Bournulli"
      ,"Bournulli+isotonic"
      ,"Bournulli+sigmoid"]
models=[MultinomialNB()
        ,CalibratedClassifierCV(MultinomialNB(),cv=2,method='isotonic')
        ,CalibratedClassifierCV(MultinomialNB(),cv=2,method='sigmoid')
        ,ComplementNB()
        ,CalibratedClassifierCV(ComplementNB(),cv=2,method='isotonic')
        ,CalibratedClassifierCV(ComplementNB(),cv=2,method='sigmoid')
        ,BernoulliNB()
        ,CalibratedClassifierCV(BernoulliNB(),cv=2,method='isotonic')
        ,CalibratedClassifierCV(BernoulliNB(),cv=2,method='sigmoid')]
for name,clf in zip(name,models):
    clf.fit(train_x_,train_y)
    pred_y=clf.predict(test_x_)
    proba=clf.predict_proba(test_x_)
    score=clf.score(test_x_,test_y)
    print(name)
    print("Accuracy:{:.3f}".format(score))
    print("\n")
    
## 补集朴素贝叶斯效果更好，sigmoid校准后效果略有提升
## 伯努利朴素贝叶斯在isotonic校准后精确度提升明显
```

<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200327203209900.png" alt="image-20200327203209900" style="zoom:50%;" />
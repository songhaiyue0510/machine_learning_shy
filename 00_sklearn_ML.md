<img src="/Users/songhaiyue/Library/Application Support/typora-user-images/image-20200330110035961.png" alt="image-20200330110035961" style="zoom:50%;" />

### Machine learning 常用模块

```python
# 基础模块
import pandas as pd
import numpy as np

# 数据集
from sklearn.datasets import fetch_20newsgroups #20个新闻组的语料库，英文

# 数据集处理
from sklearn.model_selection import train_test_split ##分割测试、训练集

# 机器学习算法
from sklearn.naive_bayes import MultinomialNB,ComplementNB,BernoulliNB # 朴素贝叶斯
from sklearn.linear_model import LogisticRegression as LR  # 逻辑回归算法
from sklearn.svm import SVC # 支持向量机

# 辅助算法
from sklearn.feature_extraction.text import CountVectorizer as CV # 单词计数向量
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF # TF-IDF算法
from sklearn.calibration import CalibratedClassifierCV # 概率校准

# 评估指标
from sklearn.metrics import brier_score_loss,recall_score,roc_auc_score 
from sklearn.calibration import calibration_curve ## 可靠性曲线
from sklearn.model_selection import cross_val_score # 交叉验证

## 数据预处理和特征选择
from sklearn.preprocessing import MinMaxScaler,StandardScaler ## 数据的无量纲化
from sklearn.impute import SimpleImputer           ## 缺失值处理模块
from sklearn.preprocessing import LabelEncoder     ## 标签专用，能够将分类转换为分类数值
from sklearn.preprocessing import OrdinalEncoder   ## 特征专用，将分类特征转换为分类数值，不能导入一维数组
from sklearn.preprocessing import OneHotEncoder    ## 独热编码，哑变量
from sklearn.preprocessing import Binarizer        ## 连续变量二值化
from sklearn.preprocessing import KBinsDiscretizer ## 连续变量分箱

```



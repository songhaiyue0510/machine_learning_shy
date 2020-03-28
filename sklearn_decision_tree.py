# ! /user/bin/env python
# # -*- coding: utf-8 -*-

"""
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
后剪枝算法+实例：
https://blog.csdn.net/weixin_43216017/article/details/87534496
树的可视化：
进入到上面保存好的dot所在目录，打开cmd运行dot out.dot -T pdf -o out.pdf 命令，pdf 图片
"""

##--------import everything--------------------------------
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
##---------------------------------------------------- 数据读入和处理----------------------------------------------------
filename='/Users/songhaiyue/Desktop/B01_python/Machine_Learning_in_Action/machinelearninginaction/Ch03/lenses.txt'
fr=open(filename)
# print(fr.read())
data=pd.DataFrame([inst.strip().split('\t') for inst in fr.readlines()])
data.columns=['age','prescript','astigmatic','tearRate','y']
print('\n数据\n',data)
x=data.iloc[:,0:3]
print('\n数据中的x\n',x)
#sparse=False意思是不产生稀疏矩阵
vec=DictVectorizer(sparse=False)
#先用 pandas 对每行生成字典，然后进行向量化
X_train = vec.fit_transform(x.to_dict(orient='record'))
print('\n向量化后的矩阵\n',X_train)
print('\n向量化后的数据类型\n',type(X_train))
print('\n向量化后的列名\n',vec.get_feature_names())

Y_train=data['y']
print('\n原始特征数据\n',Y_train)
feature_names=vec.get_feature_names()
# class_names=['hard','no lenses','soft']
class_names=list(set(Y_train))
class_names.sort() # 必须升序排列
print(class_names)
##---------------------------------------------------- 鸢尾花数据----------------------------------------------------
# from sklearn.datasets import load_iris
# iris = load_iris()
# # print(iris)
# X_train=iris.data
# Y_train=iris.target
# feature_names=iris.feature_names
# class_names=iris.target_names
# class_names.sort() # 必须升序排列
# print(class_names)
##---------------------------------------------------- 决策树训练----------------------------------------------------
# 划分成训练集，交叉集，验证集
test_size=0.0
train_x, test_x, train_y, test_y = train_test_split(X_train, Y_train, test_size=test_size)

# 训练决策树
clf = tree.DecisionTreeClassifier(criterion='gini'
                                  ,min_samples_leaf=6
                                  # ,min_impurity_decrease=0.01
                                  )
clf.fit(X_train, Y_train)

# 决策树可视化
with open("out.dot", 'w') as f:
    f = tree.export_graphviz(clf
                             , out_file=f
                             , feature_names=feature_names
                             , filled=True
                             , rounded=True
                             , class_names=class_names ## 类型为list，必须升序排列
                             , node_ids=True)

## 从sklearn中提取决策规则
from sklearn.tree import _tree
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print ("def tree({}):".format(", ".join(feature_names)))
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print ("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {}".format(indent, tree_.value[node]))
            print ("{}return node为{}".format(indent, node))
    recurse(0, 1)
tree_to_code(clf,feature_names)


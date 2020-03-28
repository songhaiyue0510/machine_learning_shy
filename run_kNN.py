# ! /user/bin/env python
# # -*- coding: utf-8 -*-
# import kNN

# group,labels= kNN.createDataSet()
# print(group)
# print(labels)
# kNN.classfy0([0,0],group,labels,3)

## 例子1：约会网站数据R
# filename='/Users/shy/Desktop/machine_learning_in_action/machinelearninginaction/Ch02/datingTestSet2.txt'

## 测试样本量不同时，错误率
# for r in range(1,10,1):1
#   test_ratio=float(r/10)
#   print('the test ratio is {}'.format(test_ratio))
#   kNN.datingClassTest(test_ratio,filename)
# datingDatMat,datingLabels=kNN.file2matrix(filename)
# print(datingDatMat[:10])
# normMat,ranges,minValue=kNN.autoNorm(datingDatMat)
# print(normMat[:10])


# # 可视化
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
#
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(datingDatMat[:,0],datingDatMat[:,1],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
# plt.legend(loc='upper left')
# plt.show()

# kNN.classifyPerson(filename)


## 例子2：在二进制存储的图像上使用kNN
# test_dir ="/Users/shy/Desktop/machine_learning_in_action/machinelearninginaction/Ch02/digits/testDigits"
# train_dir ="/Users/shy/Desktop/machine_learning_in_action/machinelearninginaction/Ch02/digits/trainingDigits"
# kNN.handwritingClassTest(train_dir,test_dir)

## 使用sklearn

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection  import cross_val_score

# 交叉验证的函数
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


#读取鸢尾花数据集
iris = load_iris()
x = iris.data
y = iris.target
## target 共有3个类别
print(set(y))

#循环，取k=1到k=31，查看误差效果，确定最佳的K值
k_range = range(1, 31)
k_error = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
    scores = cross_val_score(knn, x, y, cv=6, scoring='accuracy')
    k_error.append(1 - scores.mean())

#画图，x轴为k值，y值为误差值
# plt.plot(k_range, k_error)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Error')
# plt.show()

n_neighbors=k_error.index(min(k_error))+1
print(n_neighbors)

# 确定步长
h=0.02
# 创建彩色的图
cmap_light=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold=ListedColormap(['#FF0000','#00FF00','#0000FF'])

# 二维
for weights in ['uniform','distance']:
    # 创建knn分类实例，并拟合数据
    clf=KNeighborsClassifier(n_neighbors,weights=weights)
    ## 针对前2个特征进行训练
    clf.fit(x[:,:2],y)
    # 绘制决策边界
    x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
    y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
    # z_min,z_max=x[:,2].min()-1,x[:,2].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    # numpy.c_[]和np.r_[]可视为兄弟函数，两者的功能为np.r_[]添加行，np.c_[]添加列。
    # x.ravel()ravel函数将多维数组降为一维，仍返回array数组，元素以列排列
    z=clf.predict(np.c_[xx.ravel(),yy.ravel()])

    # 将结果放入彩色图中
    z=z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx,yy,z,cmap=cmap_light)

    # 绘制训练点
    plt.scatter(x[:,0],x[:,1],c=y,cmap=cmap_bold)
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.title("3-Class classification (k={},weights={})".format(n_neighbors,weights))
plt.show()

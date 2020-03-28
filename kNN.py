# ! /user/bin/env python
# # -*- coding: utf-8 -*-

import numpy as np
import operator
import os

## 建立一个初始数据集
def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B',"B"]
    return group,labels

## 计算某待测样本的类别
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndices=distances.argsort()
    classCount={}
    for i in range(k):
        voteLabel=labels[sortedDistIndices[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    # dict.items() 以列表形式返回可遍历的(键, 值) 元组数组
    # operator.itemgetter(n1,n2) operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 将文本记录转换为Numpy的解析程序
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    # print(arrayOLines[:10])
    numberOfLines=len(arrayOLines)
    # print(numberOfLines)
    # x
    returnMat=np.zeros((numberOfLines,3))
    # y
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

## 归一化 newValue = (oldValue - min)/(max-min)
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=np.zeros(np.shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minVals,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

## 测试代码
def datingClassTest(ratio,filename):
    hoRatio=ratio
    datingDataMat,datingLabels=file2matrix(filename)
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        # print("the classifier came back with: {}, the real answer is: {}".format(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount+=1
    print("the total error rate is {}%".format(100*errorCount/float(numTestVecs)))

## 约会网站预测函数
def classifyPerson(filename):
    resultList=['not at all','in small doses','in large doses']
    perCentTats=float(input("percentage of time spent playing video games >> "))
    ffMiles=float(input("frequent flier miles earned per year >> "))
    iceCream=float(input("liters of ice cream consumed per year >>"))
    datingDataMat,datingLabels=file2matrix(filename)
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=np.array([ffMiles,perCentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person {}".format(resultList[classifierResult-1]))

#### 二进制图像使用kNN算法
# 图像转换成向量
def img2vector(filename):
    returnVect=np.zeros((1,1024)) ## 建立1✖️1024的array
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

# 手写数字识别系统的测试代码
def handwritingClassTest(train_directory,test_directory):
    hwLabels=[]

    traingFileList=os.listdir(train_directory)
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。它不包括 . 和 .. 即使它在文件夹中
    m=len(traingFileList)
    trainingMat=np.zeros((m,1024))
    for i in range(m):
        fileNameStr=traingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector(train_directory+"/{}".format(fileNameStr))

    testFileList=os.listdir(test_directory)
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector(test_directory+"/{}".format(fileNameStr))
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        # print("the classifier came back with {}, the real answer is {}".format(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            errorCount+=1.0
    print("\nthe total number of errors is: {}".format(errorCount))
    print("\nthe total error rate is: {:.2f}%".format(errorCount*100/float(mTest)))

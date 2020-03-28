# ! /user/bin/env python
# # -*- coding: utf-8 -*-

"""
决策树
优点：计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不想关特征数据
缺点：可能产生过度匹配问题
适用数据类型：数值型和标称型
"""

from math import log
import operator
## 度量数据的无序程度


def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={} ## 新建一个字典
    for featVec in dataSet:
        currentLabel=featVec[-1] #默认最后一列是y
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    # print(labelCounts)
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2) ## 2是底数
    return shannonEnt

def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels
## 按照给定特征划分数据集
# dataset为待划分的数据集，axis为划分数据集的特征，value为特征的返回值
def splitDataSet(dataSet,axis,value):
    retDataSet=[] #建立一个list
    for featVec in dataSet:
        # print("1---",featVec[axis])
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis] ## 取出特征前的数据
            # print("2---",reducedFeatVec[:axis])
            ## append(x)将x作为一个整体加到list的末尾；extend(x)将x中所有元素加到list末尾
            ## a=[1,2,3],b=[4,5,6]
            # a.append(b)得到[1,2,3,[4,5,6]]，a.extend(b)得到[1,2,3,4,5,6]
            reducedFeatVec.extend(featVec[axis+1:])
            # print("3---",reducedFeatVec)
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1 ## 最后一列为Y,feature为数据集的列数-1
    # print(numFeatures)
    baseEntropy=calcShannonEnt(dataSet)
    # print(baseEntropy)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        # print(i)
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
            # print(newEntropy)
        infoGain=baseEntropy-newEntropy
        # print(infoGain)
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

# 多数表决的方式决定叶子的分类
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount=0
            classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


## 创建树
def createTree(dataSet,labels):
    # print("dataset is {}".format(dataSet))
    classList=[example[-1] for example in dataSet]
    # print("class list is {}".format(classList))
    # print("class list count is {}".format(classList.count(classList[0])))
    # print("dataSet size is {}".format(dataSet[0]))
    ## 若类别完全相同，则停止划分
    if classList.count(classList[0])==len(classList):
        # print("class list is {}".format(classList[0]))
        # print("class list is {}".format(classList))
        # print("class is the same, stop splitting")
        # print("-"*20)
        return classList[0]
    ## 遍历完所有特征时返回出现次数最多的
    if len(dataSet[0])==1:
        # print("majority cnt is {}".format(majorityCnt(classList)))
        # print("-"*20)
        # print("all features are done, define the leaf class")
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    # print("best Feat is {}".format(bestFeat))
    bestFeatLabel=labels[bestFeat]
    print("分割特征是{}".format(bestFeatLabel))
    ## 建立字典
    myTree={bestFeatLabel:{}}
    # print("my tree is {}".format(myTree))
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    # print("best feat values is {}".format(featValues))
    # print("*"*20)
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        # print("subLabel is {}".format(subLabels))
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

## 使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    print(firstStr)
    secondDict=inputTree[firstStr]
    print(secondDict)
    featIndex=featLabels.index(firstStr)
    print(featIndex)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel
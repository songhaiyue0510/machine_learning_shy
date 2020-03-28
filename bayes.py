# ! /user/bin/env python
# # -*- coding: utf-8 -*-

"""

"""
import numpy as np
# 创建实验样本
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1] # 1代表侮辱性文字，0代表正常言论
    return postingList,classVec

# 获取词汇表
def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

# 将文本转换成数字，
# 朴素贝叶斯词级模型，将每个词的出现与否作为一个特征
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word: {} is not in my vocabulary!".format(word))
    return returnVec

# 朴素贝叶斯词袋模型，将每个词出现与否+出现次数作为一个特征
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

# 朴素贝叶斯分类器训练函数
# trainMatrix文档矩阵，trainCategory文档标签所构成的向量
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    # print(numTrainDocs)
    numWords=len(trainMatrix[0])
    # print(numWords)
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=np.ones(numWords)
    p1Num=np.ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            # print(p1Num)
            p1Denom+=sum(trainMatrix[i])
            # print(p1Denom)
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=np.log(p1Num/p1Denom)
    p0Vect=np.log(p0Num/p0Denom)  #避免下溢出
    return p0Vect,p1Vect,pAbusive

# 朴素贝叶斯份分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0


# 使用朴素贝叶斯进行交叉验证
# 使用朴素贝叶斯进行垃圾邮件分类
def textParse(bigString):
    import re
    listOfTokens=re.split(r'\W+',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest(filepath):
    docList=[]
    classList=[]
    fullText=[]
    # 导入并解析文本文件
    for i in range(1,26):
        wordList=textParse(open(filepath+'email/spam/%d.txt' %i,encoding ='unicode_escape').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open(filepath+'email/ham/%d.txt' %i,encoding ='unicode_escape').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    # 随机构建训练集
    vocabList=createVocabList(docList)
    trainingSet=list(range(50))
    testSet=[]
    for i in range(10):
        randIndex=int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    # 对测试集分类
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is:{}'.format(float(errorCount)/len(testSet)))

## 使用朴素贝叶斯发现地域相关的用词
import feedparser
ny=feedparser.parse('http://newyork.craigslist.org/stp/index/rss')
ny['entries']
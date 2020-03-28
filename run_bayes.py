# ! /user/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import bayes
import numpy as np
import re

##----------------------------基于给定数据分类----------------------------------------
# listOPosts,listClasses=bayes.loadDataSet()
# myVocabList=bayes.createVocabList(listOPosts)
# print(myVocabList)
# # print(len(myVocabList))
# # print(bayes.setOfWords2Vec(myVocabList,listOPosts[0]))
# # print(bayes.setOfWords2Vec(myVocabList,['time']))
#
# trainMat=[]
# for postinDoc in listOPosts:
#     trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))
#
# print(trainMat)
# p0V,p1V,pAb=bayes.trainNB0(trainMat,listClasses)
# # print(p0C,p1V,pAb)
# testEntry=['love','my','dalmation']
# thisDoc=np.array(bayes.setOfWords2Vec(myVocabList,testEntry))
# print(testEntry,'classified as:',bayes.classifyNB(thisDoc,p0V,p1V,pAb))
# testEntry=['stupid','garbage']
# thisDoc=np.array(bayes.setOfWords2Vec(myVocabList,testEntry))
# print(testEntry,'classified as:',bayes.classifyNB(thisDoc,p0V,p1V,pAb))

##----------------------------使用朴素贝叶斯过滤垃圾邮件----------------------------------------
filepath='/Users/songhaiyue/Desktop/B01_python/Machine_Learning_in_Action/machinelearninginaction/Ch04/'

# filename=filepath+'email/ham/6.txt'
# emailText=open(filename,encoding ='unicode_escape').read()
# # print(emailText)
# listOfTokens=re.split('\s',emailText)
# print(listOfTokens)

bayes.spamTest(filepath)


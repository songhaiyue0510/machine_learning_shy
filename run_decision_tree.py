# ! /user/bin/env python
# # -*- coding: utf-8 -*-

"""

"""
# import decision_tree as trees
# import tree_plotter as treePlotter
#
# myDat,labels=trees.createDataSet()
# # myDat[0][-1]='maybe'
# print("数据集是：","\n",myDat,"\n")
# print("labels是：","\n",labels,"\n")
# ent=trees.calcShannonEnt(myDat)
# # print(ent)
# # print(trees.splitDataSet(myDat,1,1))
# # print(trees.chooseBestFeatureToSplit(myDat))
#
# myTree=trees.createTree(myDat,labels)
# print("tree 是：","\n",myTree,"\n")

# myDat,labels=trees.createDataSet()
# # myTree['no surfacing'][3]='maybe'
# # print(treePlotter.getNumLeafs(myTree))
# # print(treePlotter.getTreeDepth(myTree))
# # treePlotter.createPlot(myTree)

##--------使用决策树预测隐形眼镜类型--------------------------------
# import decision_tree as trees
# import tree_plotter as treePlotter
# filename='/Users/songhaiyue/Desktop/B01_python/Machine_Learning_in_Action/machinelearninginaction/Ch03/lenses.txt'
# fr=open(filename)
# # print(fr.read())
# lenses=[inst.strip().split('\t') for inst in fr.readlines()]
# # print(lenses[:1])
# # print(lenses)
# lensesLabels=['age','prescript','astigmatic','tearRate']
# lensesTree=trees.createTree(lenses,lensesLabels)
# print("lenses Tree is:","\n",lensesTree,'\n')
# # treePlotter.createPlot(lensesTree)


# -*- coding: cp936 -*-
from numpy import *
import matplotlib.pyplot as plt
import time
from logisticRegression import trainLogRegress, testLogRegress, showLogRegress 

def loadData():
    train_x=[]
    train_y=[]
    fileIn = open('K:/����ѵ��/testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return mat(train_x), mat(train_y).transpose()

## step 1: load data
print "step 1: load data..."
train_x, train_y = loadData()
test_x = train_x
test_y = train_y

## step 2: training
print "step 2: training..."
##opts = {'alpha':0.01, 'maxIter':20, 'optimizeType': 'smoothStocGradDescent'}
opts = {'alpha':0.01, 'maxIter':20, 'optimizeType': 'stocGradDescent'}
##opts = {'alpha':0.01, 'maxIter':20, 'optimizeType': 'gradDescent'}
optimalWeights = trainLogRegress(train_x, train_y, opts)

## step 3: testing
print "step 3: testing..."
accuracy = testLogRegress(optimalWeights, test_x, test_y)

## step 4: show the result
print "step 4: show the result..."
print 'The classify accuracy is : %.3f%%' % (accuracy * 100)
showLogRegress(optimalWeights, train_x, train_y)

##�߼��ع�ѵ��ģ�ͣ����ȳ�ʼ��Ȩ�ز���Ϊ1��Ȼ��ָ����������������Ȩ�ز�����Ȼ�����sigmoid�����������yֵ����0.5�Ƚϣ�������׼ȷ�ʡ�

import csv
import random
import operator
import math
def loadDataset(filename,split,trainset=[],testset=[]):
    with open(filename,'rt')as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y]=float(dataset[x][y])
            if random.random()<split:
                trainset.append(dataset[x])
            else:
                testset.append(dataset[x])
trainset=[]#训练集
testset=[]#测试集
csvfile = open(r'iris.data.txt','rt')
lines=csv.reader(csvfile)
dataset=list(lines)
for x in range(len(dataset)-1):
    for y in range(4):
        dataset[x][y] = float(dataset[x][y])
        if random.random()<0.44:
            trainset.append(dataset[x])
        else:
            testset.append(dataset[x])
print(trainset)
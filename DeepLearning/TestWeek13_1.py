"""
《神经网络与深度学习》部分的第三周浅层神经网络
"""
# Package imports
import numpy as np
import matplotlib.pyplot as plt
from DeepLearning.testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from DeepLearning.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1) # set a seed so that the results are consistent

"""
First, let’s get the dataset you will work on. The following code will load a “flower”
2-class dataset into variables X and Y  简单的逻辑回归的例子
"""
X,Y = load_planar_dataset()
shape_x = X.shape
shape_y = Y.shape
m = X.shape[1]     #训练集大小
print(shape_x,shape_y,m)

# 2、简单的逻辑回归
# 训练逻辑回归分类器
clf = sklearn.linear_model.LogisticRegressionCV()  #创建分类器对象
clf.fit(X.T,Y.T)

plot_decision_boundary(lambda x: clf.predict(x), X, Y)   #用训练好的分类器去预测新数据
plt.title("Logistic Regression")
plt.show()

# 预测正确率，数据集不是线性可分的，所以逻辑回归表现不好
LR_predictions = clf.predict(X.T)
print(Y.shape,LR_predictions.shape)
LR_accuracy = (np.dot(Y,LR_predictions)+np.dot(1-Y,1-LR_predictions))/Y.size*100
print ('Accuracy of logistic regression: %d' % LR_accuracy +'%')



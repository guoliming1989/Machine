'''
Created on 2018年3月4日

@author: guoliming
'''
import numpy as np 
from sklearn.datasets import load_digits 
from sklearn.metrics import confusion_matrix, classification_report #对结果的衡量函数
from sklearn.preprocessing import LabelBinarizer #转化为二维的数字类型
from NetWork.NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split  #交差验证，测试集与训练集的验证

digits = load_digits()  #装载数据
X = digits.data  #数据
y = digits.target  #标签
X -= X.min() # normalize the values to bring them into the range 0-1  
X /= X.max()

nn = NeuralNetwork([64,100,10],'logistic')  #64个像素点8x8的，输出层为10因为输出为10个数字，隐藏层为100
X_train, X_test, y_train, y_test = train_test_split(X, y)  
labels_train = LabelBinarizer().fit_transform(y_train)  
labels_test = LabelBinarizer().fit_transform(y_test)
print ("start fitting")
nn.fit(X_train,labels_train,epochs=3000)  
predictions = []  
for i in range(X_test.shape[0]):  
    o = nn.predict(X_test[i] )  
    predictions.append(np.argmax(o))  
print (confusion_matrix(y_test,predictions))#预测的数字为10*10，预测正确的为对角线
print (classification_report(y_test,predictions))#预测每一个数正确的概率
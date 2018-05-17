import numpy as np
import pylab as pl
#主要对Python进行画图的操作
from sklearn import svm
np.random.seed(100)
X=np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
print(X)
Y=[0]*20+[1]*20
print(Y)
clf=svm.SVC(kernel='linear')
clf.fit(X,Y)
w=clf.coef_[0]#参数

a=-w[0]/w[1]#斜率
print(w)
print(a)
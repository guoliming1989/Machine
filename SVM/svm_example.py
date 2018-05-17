import numpy as np
import pylab as pl
#主要对Python进行画图的操作
from sklearn import svm


#产生数据，随机数
np.random.seed(100)
X=np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]]
#产生20个点的2维的，均值和方差都为2的二维数组，可以使用一条直线分开
Y=[0]*20+[1]*20

#使用svm，产生模型
clf=svm.SVC(kernel='linear')
clf.fit(X,Y)

#得到超平面get the separating hyperplane
w=clf.coef_[0]#参数
a=-w[0]/w[1]  #斜率
xx=np.linspace(-5,5)#xx的值
yy=a*xx-(clf.intercept_[0])/w[1]  #画出斜线

#取出二个支持向量，求出二个向量的直线，最大化边界
b=clf.support_vectors_[0]
yy_down=a*xx+(b[1]-a*b[0])
b=clf.support_vectors_[-1]
yy_up=a*xx+(b[1]-a*b[0])

print ("w:",w)
print ("a:",a)

print ("support_vectors_",clf.support_vectors_)
print ("clf.coef_",clf.coef_)

pl.plot(xx,yy,'k-')
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')
pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,-1],s=80,facecolors='none')
pl.scatter(X[:,0],X[:,-1],c=Y,cmap=pl.cm.Paired)    #scatter显示出离散的点
pl.axis('tight')
pl.show()
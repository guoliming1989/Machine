from sklearn import neighbors
#临近算法包含在这个类中
from sklearn import datasets
 #导入数据集
knn=neighbors.KNeighborsClassifier()
#调用knn的分类器
iris=datasets.load_iris()   #返回一个数据集复制到iris上面
print(iris)
knn.fit(iris.data, iris.target)
#建立模型，传入特征值和目标值
predictedLabel=knn.predict([[0.1,0.2,0.3,0.4]])
print (predictedLabel)
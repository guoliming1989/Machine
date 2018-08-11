import numpy
import xgboost
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
# load数据集
dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=",")
# 把 X Y 分开
X = dataset[:,0:8]
Y = dataset[:,8]
# 现在我们分开训练集和测试集
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = cross_validation.train_test_split \
(X, Y, test_size=test_size, random_state=seed)
# 训练模型
model = xgboost.XGBClassifier()
# 这⾥里里参数的设置可以⻅见：http://xgboost.readthedocs.io/en/latest/python/
#python_api.html#module-xgboost.sklearn
model.fit(X_train, y_train)
# 做预测
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# 显示准确率
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO  #格式的预处理
#读取文件
allElectronicsData=open(r'test.csv','rt')#读取csv文件
reader=csv.reader(allElectronicsData)
headers=next(reader)#定义的第一行
print(headers)

featureList=[]   #数据的输入特性，使用的是数值型的值
labelList=[]
for row in reader:
    labelList.append(row[len(row)-1])#添加标签，取每行的最后一个数字
    #print(labelList)
    rowDict={}#取每行的特征值
    for i in range(1,len(row)-1):
        rowDict[headers[i]]= row[i]#每一行设置字典项
    featureList.append(rowDict)
print(featureList)
vec=DictVectorizer()   #直接在vec对象上调用方法，对于feature值的转化,转化为0或1的属性
dummyX=vec.fit_transform(featureList).toarray()
print("dummyX:"+str(dummyX))
print(vec.get_feature_names())
print("labelList:"+str(labelList))

lb=preprocessing.LabelBinarizer()#对于label的转化
dummyY=lb.fit_transform(labelList)
print("dummyY:"+str(dummyY))

clf=tree.DecisionTreeClassifier(criterion='entropy')  #分类树的参数的选取，基于信息熵来度量标准
clf=clf.fit(dummyX,dummyY)   #填入参数，调用fit，构建决策树
print("clf:"+str(clf))

with open("test.dot",'w')as f:
    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)
    #将原0、1的数值转化为属性值

# predict
oneRowX = dummyX[0, :]
print("oneRowX:" + str(oneRowX))
# 修改这一行的数据，然后执行
newRowX = oneRowX
newRowX[0] = 1
newRowX[1] = 0
print("newRowX:" + str(newRowX))
# 添加一个中括号
predictedY = clf.predict([newRowX])
# print(help(clf.predict))
print("predictedY:" + str(predictedY))

# 将dot文件转化为pdf文件，dot -Tpdf iris.dot -o outpu.pdf
# 决策树转化
# dot -Tpdf D:\PythonWork\TeachingPython\src\DecsionTree\test.dot -o D:\PythonWork\TeachingPython\src\DecsionTree\outpu.pdf


import numpy as np
def kmeans(X,k,maxIt):        #数据集、maxIt迭代的次数
    numPoints,numDim=X.shape  #返回多少行和列
    dataSet=np.zeros((numPoints,numDim+1))#初始化，加上一列
    dataSet[:,:-1]=X
    centroids=dataSet[np.random.randint(numPoints,size=k),:]   #随机选取k个中心点
    centroids[:,-1]=range(1,k+1)   #给中心点加上标记
    
    iterations=0
    oldCentroids=None  #旧的中心点
    
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print ("iterations:\n",iterations)
        print ("dataSet:\n",dataSet)
        print ("centroids:\n",centroids)
        
        oldCentroids=np.copy(centroids)   #旧的中心点和新的中心点
        iterations+=1
        updateLabels(dataSet, centroids)  #更新label
        centroids=getCentroids(dataSet, k)
    return dataSet
    
    
    
def shouldStop(oldCentroids,centroids,iterations,maxIt):
    if iterations>maxIt:
        return True
    return np.array_equal(oldCentroids, centroids)    #中心点的位置是否有变化，比较是否为同一个对象还是值相等

def updateLabels(dataSet,centroids):#中心点和数据集重新分类
    numPoints,numDim=dataSet.shape
    for i in range(0,numPoints):
        dataSet[i,-1]=getLabelFromClosestCentroid(dataSet[i,:-1], centroids)
        
        
        
def getLabelFromClosestCentroid(dataSetRow,centroids):
    label=centroids[0,-1]
    
    minDist=np.linalg.norm(dataSetRow-centroids[0,:-1])
    
    for i in range(1,centroids.shape[0]):
        dist=np.linalg.norm(dataSetRow-centroids[i,:-1])
        if dist<minDist:
            minDist=dist
            label=centroids[i,-1]
    print ("minDist",minDist)
    return (label)

def getCentroids(dataSet,k):
    
    result=np.zeros((k,dataSet.shape[1]))
    for i in range(1,k+1):
        oneCluster=dataSet[dataSet[:,-1]==i,:-1]
        result[i-1,:-1]=np.mean(oneCluster,axis=0)
        result[i-1,-1]=i
    return result

x1=np.array([1,1])
x2=np.array([2,1])
x3=np.array([4,3])
x4=np.array([5,4])
testX=np.vstack((x1,x2,x3,x4))
result=kmeans(testX, 2, 10)
print ("final result:\n",result)
    
    
    
    
    
    
    
    
    
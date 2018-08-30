import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Week113.testCases_v2 import *
from Week113.planar_utils import sigmoid,plot_decision_boundary,load_planar_dataset,load_extra_datasets

# 下载额外的数据集
np.random.seed(1)   # 设定一个种子，保证结果的一致性
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

dataset = "blobs"
X,Y = datasets[dataset]
X,Y = X.T,Y.reshape(1,Y.shape[0])

if dataset == "blobs":
    Y = Y % 2
print(X.shape, Y.shape, X.shape[1])

def setcolor(Y):
    color=[]
    for i in range(Y.shape[1]):
        if Y[:,i]==1:
            color.append('b')
        else:
            color.append('r')
    return color

#显示数据
plt.scatter(X[0,:], X[1:], s=30, c=setcolor(Y), cmap=plt.cm.Spectral)
plt.show() #加上才显示


# 3、神经网络模型
# 3-1 定义三层网络结构
def layer_sizes(X,Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x,n_h,n_y)

# 3-2 初始化模型参数
def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h,n_x)
    W2 = np.random.randn(n_y,n_h)
    b1 = np.zeros((n_h,1))
    b2 = np.zeros((n_y,1))
    assert(W1.shape == (n_h,n_x))
    assert(W2.shape == (n_y,n_h))
    assert(b1.shape == (n_h,1))
    assert(b2.shape == (n_y,1))
    parameters = {"W1":W1, "W2":W2, "b1":b1, "b2":b2}
    return parameters

# 3-3 计算前向传播
def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)

    #assert(A2.shape == (1,X.shape[1]))
    cache = {"Z1":Z1, "Z2":Z2, "A1":A1, "A2":A2}

    return A2,cache

# 3-4 计算损失函数
def compute_cost(A2,Y,parameters):
    m = Y.shape[1]

    #计算交叉熵损失函数
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
    cost = -np.sum(logprobs)/m
    cost = np.squeeze(cost)
    assert(isinstance(cost,float))

    return cost

# 3-5 计算反向传播过程
def backward_propagation(parameters,cache,X,Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2-Y
    dW2 = 1/m * np.dot(dZ2,A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T,dZ2) * (1-np.power(A1,2))
    dW1 = 1/m * np.dot(dZ1,X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1":dW1, "dW2":dW2, "db1":db1, "db2":db2}

    return grads

# 3-6 更新参数
def update_parameters(parameters,grads,learning_rate=1.2):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]

    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    b1 -= learning_rate * db1
    b2 -= learning_rate * db2

    parameters = {"W1":W1, "W2":W2, "b1":b1, "b2":b2}

    return parameters

# 3-7 模型
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    for i in range(0,num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=0.3)

        if print_cost and i%100==0:
            print("cost after iteration %i: %f" % (i,cost))

    return parameters

# 3-8 预测
def predict(parameters, X):
    A2,cache = forward_propagation(X,parameters)
    predictions = np.around(A2)
    return predictions

# 3-9 运行测试代码
# 下载数据，训练模型
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# 预测正确率
predictions = predict(parameters, X)
print(Y.shape,predictions.shape)
accuracy = float(np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T))/Y.size*100
print('Accuracy : %d' % accuracy +'%')

# 画边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
import numpy as np
def tanh(x):  #双曲线函数
    return np.tanh(x)
def tanh_deriv(x): #双曲函数的导数 
    return 1.0 - np.tanh(x)*np.tanh(x)
def logistic(x):  #逻辑回归函数
    return 1/(1 + np.exp(-x))
def logistic_derivative(x):  #逻辑回归函数导数
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:   
    def __init__(self, layers, activation='tanh'):  
#构造函数，当实例化一个对象，首先要调用构造函数，self相当于java中的this，layers相当于神经网络中的层数，activation指定函数，指定默认的函数，如果用户可以另外指明
        """  
        :param layers: A list containing the number of units in each layer.
        Should be at least two values  
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"  
        """  
        if activation == 'logistic':  
            self.activation = logistic  
            self.activation_deriv = logistic_derivative  
        elif activation == 'tanh':  
            self.activation = tanh  
            self.activation_deriv = tanh_deriv
    
        self.weights = []  #初始化权重的值，取得随机数
        for i in range(1, len(layers) - 1):  #神经网络的层次,每一层网络之中存在左右二个权重向量，从1开始表示第2层开始
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)  #产生随机的权重，每一层的左右二层
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)
        print("weight")    
        print(self.weights)   
        
    def fit(self, X, y, learning_rate=0.2, epochs=10000):   
        #X表示二维的矩阵，y为label，learning_rate学习率0.2， epochs：抽取其中的某一行对神经网络更新，最多循环一万次    
        X = np.atleast_2d(X)   #确定x最少为2维的数组      
        temp = np.ones([X.shape[0], X.shape[1]+1])    #初始化矩阵     
        temp[:, 0:-1] = X  # adding the bias unit to the input layer        取第一列到除了最后的一列的所有行 
        X = temp         
        y = np.array(y)
    
        for k in range(epochs):  
            i = np.random.randint(X.shape[0])  #随机取一行
            a = [X[i]]#抽取的一行实例来更新
    
            for l in range(len(self.weights)):  #going forward network, for each layer
                a.append(self.activation(np.dot(a[l], self.weights[l])))  #内积Computer the node value for each layer (O_i) using activation function
            error = y[i] - a[-1]  #Computer the error at the top layer
            deltas = [error * self.activation_deriv(a[-1])] #For output layer, Err calculation (delta is updated error)
            
            #Staring backprobagation，更新权重
            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer 
                #Compute the updated error (i,e, deltas) for each node going from top layer to input layer 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))  
            deltas.reverse()  #颠倒顺序
            for i in range(len(self.weights)):  
                layer = np.atleast_2d(a[i])  
                delta = np.atleast_2d(deltas[i])  
                self.weights[i] += learning_rate * layer.T.dot(delta)
                
    def predict(self, x):         
        x = np.array(x)         
        temp = np.ones(x.shape[0]+1)         
        temp[0:-1] = x         
        a = temp         
        for l in range(0, len(self.weights)):             
            a = self.activation(np.dot(a, self.weights[l]))         
        return a



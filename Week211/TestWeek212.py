import numpy as np
import sklearn
import matplotlib.pyplot as plt
import sklearn.datasets
from Week211.reg_utils import load_2D_dataset,compute_cost,forward_propagation,predict,predict_dec,relu,sigmoid
from Week211.reg_utils import initialize_parameters,backward_propagation,update_parameters,plot_decision_boundary
import scipy.io

train_x,train_y,test_x,test_y=load_2D_dataset()

def model(X,Y,learning_rate=0.3,num_iterations=30000,print_cost=True,lambd=0,keep_prob=1):
    grads={}
    costs=[]
    m=X.shape[1]
    layers_dims=[X.shape[0],20,3,1]
    #Initialize=initialize_parameters(layers_dims)
    parameters=initialize_parameters(layers_dims)
    for i in range(0, num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:   #当keep_prob<1时，使用drop_out,其中keep_prob between 0 and 1
            a3, cache,cache_Dl = forward_propagation_with_dropout(X, parameters, keep_prob)

        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:   #如果lambd不等于0，那么使用regulrization
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        # Backward propagation.
        assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout,
                                            # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propogation_with_dropout(X,Y,cache,cache_Dl,keep_prob)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

parameters=model(train_x,train_y,learning_rate=0.3,num_iterations=30000,print_cost=True,lambd=0,keep_prob=1)
print (train_x.shape)
print("On the train do not use regularization and dropout")
train_prediction=predict(train_x,train_y,parameters)
print ("On the test do not use regularization and dropout")
test_prediction=predict(test_x,test_y,parameters)

def compute_cost_with_regularization(A3,Y,parameters,lamda): #将L2正则化添加到公式中
    m=Y.shape[1]
    L=len(parameters)//2
    regularization_sum=0.0
    for l in range(L):
        regularization_sum += np.sum(np.square(parameters["W"+str(l+1)]))

    regularization_cost=(lamda/(2.0*m))*regularization_sum
    cost_entropy=compute_cost(A3,Y)
    cost=cost_entropy+regularization_cost
    return cost

'''
a3, Y_assess, parameters=compute_cost_with_regularization_test_case()
cost=compute_cost_with_regularization(a3,Y_assess,parameters,lamda=0.1)
print cost
'''

def backward_propagation_with_regularization(X,Y,caches,lamda):   #这个函数对于多层神经网络均适用
    L=len(caches)//4
    cache={}
    m=X.shape[1]
    grads={}
    for i in range(0,L):
        cache["Z"+str(i+1)]=caches[i*4]
        cache["A"+str(i+1)]=caches[i*4+1]
        cache["W"+str(i+1)]=caches[i*4+2]
        cache["b"+str(i+1)]=caches[i*4+3]
    grads["dZ"+str(L)]=cache["A"+str(L)]-Y
    grads["dW"+str(L)]=(1.0/m)*np.dot(grads["dZ"+str(L)],cache["A"+str(L-1)].T)+np.float(lamda/m)*cache["W"+str(L)]
    grads["db"+str(L)]=(1.0/m)*np.sum(grads["dZ"+str(L)],axis=1,keepdims=True)
    grads["dA"+str(L-1)]=np.dot(cache["W"+str(L)].T,grads["dZ"+str(L)])
    for l in reversed(range(1,L)):
        if l > 1:
            grads["dZ"+str(l)]=np.multiply(grads["dA"+str(l)],np.int64(cache["A"+str(l)]>0))
 #       grads["dW"+str(l)]=cache["A"+str(L)]-Y
            grads["dW"+str(l)]=(1.0/m)*np.dot(grads["dZ"+str(l)],cache["A"+str(l-1)].T)+np.float(lamda/m)*cache["W"+str(l)]
            grads["db"+str(l)]=(1.0/m)*np.sum(grads["dZ"+str(l)],axis=1,keepdims=True)
            grads["dA"+str(l-1)]=np.dot(cache["W"+str(l)].T,grads["dZ"+str(l)])
        elif l==1:
            grads["dZ"+str(1)]=np.multiply(grads["dA"+str(1)],np.int64(cache["A"+str(1)]>0))
            grads["dW"+str(1)]=(1.0/m)*np.dot(grads["dZ"+str(1)],X.T)+np.float(lamda/m)*cache["W"+str(1)]
            grads["db"+str(1)]=(1.0/m)*np.sum(grads["dZ"+str(1)],axis=1,keepdims=True)
    return grads


#test the correctness of backward_propagation
"""
X_assess,Y_assess,cache=backward_propagation_with_regularization_test_case()
grads=backward_propagation_with_regularization(X_assess,Y_assess,cache,lamda=0.7)
print ("dW1 :"+str(grads["dW1"]))
print ("dW2 :"+str(grads["dW2"]))
print ("dW3 :"+str(grads["dW3"]))

"""

#remerber to recover the codes below
parameters_with_regularization=model(train_x,train_y,learning_rate=0.3,num_iterations=30000,print_cost=True,lambd=0.7,keep_prob=1)
print("On the train with regularization")
train_prediction_regularization=predict(train_x,train_y,parameters_with_regularization)

print ("On the test with regularization")
test_prediction_regularization=predict(test_x,test_y,parameters_with_regularization)

plt.title("Optimize with regularization")
axes=plt.gca()
axes.set_xlim([-0.7,0.6])
axes.set_ylim([-0.7,0.6])
plot_decision_boundary(lambda x: predict_dec(parameters_with_regularization,x.T),train_x,train_y)


plt.title("Optimize without regularization")
axes=plt.gca()
axes.set_xlim([-0.7,0.6])
axes.set_ylim([-0.7,0.6])
plot_decision_boundary(lambda x: predict_dec(parameters,x.T),train_x,train_y)

def forward_propagation_with_dropout(X,parameters,keep_prob=0.5):
    cache=[] #list
    cache_D={}  #dictionary
    L=len(parameters)//2
    W1=parameters["W1"]
    b1=parameters["b1"]
    Z1=np.dot(W1,X)+b1
    A1=relu(Z1)
    cache=[Z1,A1,W1,b1]
    np.random.seed(1)
    for l in range(1,L):
        cache_D["D"+str(l)]=np.random.rand(cache[4*(l-1)+1].shape[0],cache[4*(l-1)+1].shape[1])   #randn(A1.shape[0],A1.shape[]1)
        cache_D["D"+str(l)]=np.where(cache_D["D"+str(l)]<=keep_prob,1,0)    # D1
        cache[4*(l-1)+1]=np.multiply(cache[4*(l-1)+1],cache_D["D"+str(l)])  # A1,eliminate some neure randomly
        cache[4*(l-1)+1]=cache[4*(l-1)+1]/keep_prob
        cache_4l=np.dot(parameters["W"+str(l+1)],cache[4*(l-1)+1])+parameters["b"+str(l+1)]  # cache[4*l]  Z2
        cache.append(cache_4l)
        if l==(L-1):
            cache_4l_plus1=sigmoid(cache[4*l])  #A2,A3,...
        elif l<L:
            cache_4l_plus1=relu(cache[4*l])
        cache.append(cache_4l_plus1)    #A2,A3,....
        cache_4l_plus2=parameters["W"+str(l+1)]  # W2,W3,...
        cache.append(cache_4l_plus2)
        cache_4l_plus3=parameters["b"+str(l+1)]  #b2,b3,...
        cache.append(cache_4l_plus3)
    return cache[4*(L-1)+1],cache,cache_D

"""
x_assess,parameters=forward_propagation_with_dropout_test_case()
AL,cache,cache_D=forward_propagation_with_dropout(x_assess,parameters,keep_prob=0.7)
print ("AL"+str(AL))
"""

#define the function forward_propogation_with_dropout
def backward_propogation_with_dropout(X,Y,cache,cache_D,keep_prob=0.5):
    grads={}
    L=len(cache)//4
    m=X.shape[1]
    AL=cache[4*(L-1)+1]    #(1,5)
    grads["dZ"+str(L)]=AL-Y   #dZ3 (1,5)
    grads["dW"+str(L)]=(1.0/m)*np.dot(grads["dZ"+str(L)],cache[4*(L-2)+1].T)   # dW3=dZ*A.T (1,3)=(1,5)*(3,5).T
    grads["db"+str(L)]=(1.0/m)*np.sum(grads["dZ"+str(L)],axis=1,keepdims=True)  #db3
    grads["dA"+str(L-1)]=np.dot(cache[4*(L-1)+2].T,grads["dZ"+str(L)])   #dA2  (3,5)  W3*dZ
    for l in reversed(range(1,L)):
        grads["dA"+str(l)]=grads["dA"+str(l)]/keep_prob    #dA2 (3,5)
        grads["dA"+str(l)]=np.multiply(grads["dA"+str(l)],cache_D["D"+str(l)])  #dA2 (3,5) ,D2(3,5)
        grads["dZ"+str(l)]=np.multiply(grads["dA"+str(l)],np.int64(cache[4*(l-1)]>0)) #dZ2(3,5)
        if l>1:
            grads["dW"+str(l)]=(1.0/m)*np.dot(grads["dZ"+str(l)],cache[4*(l-2)+1].T)   # dW2(3,2)=dZ2(3,5)*A1(2,5).T
        elif l==1:
            grads["dW"+str(l)]=(1.0/m)*np.dot(grads["dZ"+str(l)],X.T)
        grads["db"+str(l)]=(1.0/m)*np.sum(grads["dZ"+str(l)],axis=1,keepdims=True)
        grads["dA"+str(l-1)]=np.dot(cache[4*(l-1)+2].T,grads["dZ"+str(l)])
    return grads

parameters_with_dropout=model(train_x,train_y,keep_prob=0.86,learning_rate=0.3)
print ("On the train set with dropout:")
predictions_train = predict(train_x, train_y, parameters_with_dropout)
print ("On the test set with dropout:")
predictions_test = predict(test_x, test_y, parameters_with_dropout)

plt.title("the cost curve with ")
axes=plt.gca()
axes.set_xlim([-0.7,0.6])
axes.set_ylim([-0.7,0.6])
plot_decision_boundary(lambda x:predict_dec(parameters_with_dropout,x.T),train_x,train_y)
import numpy as np
X = np.zeros((400,2))
#print(X)
ix = range(0, 2000)
#print(ix)
#指定间隔返回数据量
t = np.linspace(1*3.12,(1+1)*3.12,200) + np.random.randn(200)*0.2
r = 4*np.sin(4*t) + np.random.randn(200)*0.2
#print(t)
#print(r)


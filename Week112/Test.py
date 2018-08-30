import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))


# axex: 设置坐标轴边界和表面的颜色、坐标刻度值大小和网格的显示
# backend: 设置目标暑促TkAgg和GTKAgg
# figure: 控制dpi、边界颜色、图形大小、和子区( subplot)设置
# font: 字体集（font family）、字体大小和样式设置
# grid: 设置网格颜色和线性
# legend: 设置图例和其中的文本的显示
# line: 设置线条（颜色、线型、宽度等）和标记
# patch: 是填充2D空间的图形对象，如多边形和圆。控制线宽、颜色和抗锯齿设置等。
# savefig: 可以对保存的图形进行单独设置。例如，设置渲染的文件的背景为白色。
# verbose: 设置matplotlib在执行期间信息输出，如silent、helpful、debug和debug-annoying。
# xticks和yticks: 为x,y轴的主刻度和次刻度设置颜色、大小、方向，以及标签大小。
#最简单的入门是从类 MATLAB API 开始，它被设计成兼容 MATLAB 绘图函数。

import matplotlib.pyplot as plt
labels='frogs','hogs','dogs','logs'
sizes=15,20,45,10
colors='yellowgreen','gold','lightskyblue','lightcoral'
explode=0,0.1,0,0
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
plt.axis('equal')
plt.show()


from numpy import *
x = linspace(0, 5, 10)
y = x ** 2
plt.plot(x, y, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('title')
plt.show()


#设置多张图片
# 1D data
x = [1,2,3,4,5]
y = [2.3,3.4,1.2,6.6,7.0]
plt.figure(figsize=(12,6))
plt.subplot(231)
plt.plot(x,y)
plt.title("plot")

plt.subplot(232)
plt.scatter(x, y)
plt.title("scatter")

plt.subplot(233)
plt.pie(y)
plt.title("pie")

plt.subplot(234)
plt.bar(x, y)
plt.title("bar")

# 2D data
import numpy as np
delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z = Y**2 + X**2

plt.subplot(235)
plt.contour(X,Y,Z)
plt.colorbar()
plt.title("contour")

# read image
import matplotlib.image as mpimg
img=mpimg.imread('marvin.jpg')

plt.subplot(236)
plt.imshow(img)
plt.title("imshow")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)
# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from pylab import *
x = np.arange(-5.0, 5.0, 0.02)
y1 = np.sin(x)
plt.figure(1)
plt.subplot(211)
plt.plot(x, y1)

plt.subplot(212)
#设置x轴范围
xlim(-2.5, 2.5)
#设置y轴范围
ylim(-1, 1)
plt.plot(x, y1)
plt.show()


import matplotlib.pyplot as plt
plt.figure(1)                # 第一张图
plt.subplot(211)             # 第一张图中的第一张子图
plt.plot([1,2,3])
plt.subplot(212)             # 第一张图中的第二张子图
plt.plot([4,5,6])

plt.figure(2)                # 第二张图
plt.plot([4,5,6])            # 默认创建子图subplot(111)

plt.figure(1)                # 切换到figure 1 ; 子图subplot(212)仍旧是当前图
plt.subplot(211)             # 令子图subplot(211)成为figure1的当前图
plt.title('Easy as 1,2,3')   # 添加subplot 211 的标题
plt.show()


import numpy as np
import matplotlib.pyplot as plt
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)
# 数据的直方图
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
#添加标题
plt.title('Histogram of IQ')
#添加文字
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()


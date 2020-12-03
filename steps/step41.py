# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [zeor kara deep learning]
# when : 2020.11.16/2020.11.29
if '__file__' in globals():  #global変数に格納されているかの確認
  import os, sys
  sys.path.append(os.path.join(os.path.dirname('__file__'), ".."))

import numpy as np
from dezero import Variable
import dezero.functions  as F
from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt
from dezero.utils import sum_to

def f(x):
  y = x ** 4 - 2 * x ** 2
  return y

def gx2(x):
  return 12 * x ** 2 - 4
  

def mean_squared_error(x0, x1):
  diff = x0 - x1
  return F.sum(diff ** 2) / len(diff)


# x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = F.reshape(x, (6,))
# y.backward(retain_grad=True)
# print(x.grad)

#--------function test------
# x = np.array([[1, 2, 3], [4, 5, 6]])
# y = sum_to(x, (1, 3))
# print(y)

# y = sum_to(x, (2, 1))
# print(y)
# sys.exit()



def toy_data1():
  np.random.seed(0)
  x = np.random.rand(100, 1)
  y = 5 + 2 * x + np.random.rand(100, 1)
  return Variable(x), Variable(y)
  
x, y = toy_data1()
# plt.scatter(x, y)
# plt.savefig("../png/step42.png",bbox_inches="tight")

lr = 0.1
iters = 100
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))
# print(W.shape, b.shape)


#--------function test------


def predict(x):
  y = F.matmul(x, W) + b
  return y

for i in range(iters):
  y_pred = predict(x)
  # print(y_pred)
  # sys.exit()
  loss = mean_squared_error(y, y_pred)
  # print(loss)
  # sys.exit()
  #gradient init...
  W.cleargrad()
  b.cleargrad()
  # print(W.shape, b.shape)
  # sys.exit()
  #backforward...
  loss.backward()
  W.data -= lr * W.grad.data
  b.data -= lr * b.grad.data
  print(W.grad.data, b.grad.data, loss)
  

# x = Variable(np.random.randn(2, 3))
# W = Variable(np.random.randn(3, 5))

# y = F.matmul(x,W)
# y.backward()

# print(x.grad.shape)
# print(W.grad.shape)



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




def toy_data1():
  np.random.seed(0)
  x = np.random.rand(100, 1)
  y = 5 + 2 * x + np.random.rand(100, 1)
  return Variable(x), Variable(y)

def toy_data2(isVariable=False):
  np.random.seed(0)
  x = np.random.rand(100, 1)
  # y = 5 + 2 * x + np.random.rand(100, 1)
  y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
  
  if isVariable:
    return Variable(x), Variable(y)
  return x, y

# -------------------- 
# datasets ...

np.random.seed(0)
x, y = toy_data2()
# print(type(x))
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

def predict(x):
  y = F.linear(x, W1, b1)
  y = F.sigmoid_simple(y)
  y = F.linear(y, W2, b2)
  return y

lr = 0.2
iters = 10000

# print(y)
# sys.exit()

#fitting...
for i in range(iters):
  y_pred = predict(x)
  loss = F.mean_squared_error(y, y_pred)
  
  W1.cleargrad()
  b1.cleargrad()
  W2.cleargrad()
  b2.cleargrad()
  loss.backward()

  # print(i,W1.grad.shape, b1.grad.shape, W2.grad.shape, b2.grad.shape)
  # print(i,W1.grad.shape, b1.grad.shape, W2.grad.shape, b2.grad.shape)
  # sys.exit()

  W1.data -= lr * W1.grad.data
  b1.data -= lr * b2.grad.data
  W2.data -= lr * W2.grad.data
  b2.data -= lr * b2.grad.data

  if i % 100 == 0:
    print("N={}".format(i),loss)

y_pred = predict(x)
# print(y_pred)
# sys.exit()
plt.scatter(x, y)
plt.plot(x,y_pred.data)
plt.savefig("../png/step43_01.png",bbox_inches="tight")



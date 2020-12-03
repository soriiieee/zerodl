# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [zeor kara deep learning]
# when : 2020.11.16/2020.11.29
if '__file__' in globals():  #global変数に格納されているかの確認
  import os, sys
  sys.path.append(os.path.join(os.path.dirname('__file__'), ".."))

import numpy as np
from dezero import Variable,Parameter
import dezero.functions  as F
from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt
from dezero.utils import sum_to

# from dezero.layers import Layer
import dezero.layers as L

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

#layers
l1 = L.Linear(10)
l2 = L.Linear(1)

def predict(x):
  y = l1(x)
  y = F.sigmoid_simple(y)
  y = l2(y)
  return y

lr = 0.2
iters = 10000

for i in range(iters):
  y_pred = predict(x)
  loss = F.mean_squared_error(y, y_pred)
  
  l1.cleargrads()
  l2.cleargrads()
  loss.backward()

  #update weight and bias..
  for l in [l1, l2]:
    for p in l.params():
      p.data -= lr * p.grad.data
  
  if i % 1000 == 0:
    print(i, loss)

y_pred = predict(x)
# print(y_pred)
# sys.exit()
plt.scatter(x, y,s=1)
plt.scatter(x,y_pred.data,color="r",s=1)
plt.savefig("../png/step44_01.png",bbox_inches="tight")



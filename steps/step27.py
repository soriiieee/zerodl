# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [zeor kara deep learning]
# when : 2020.11.16/2020.11.29

import numpy as np
import sys, os
import math
sys.path.append("/Users/soriiieee/work/zerodl")
# print(sys.path)
# sys.exit()
from dezero.core_simple import Function
from dezero.core_simple import Variable

from dezero.utils import plot_dot_graph
#(output, verbose=True, to_file="png/sample.png")



class Sin(Function):
  def forward(self, x):
    y = np.sin(x)
    return y
  
  def backward(self, gy):
    x = self.inputs[0].data
    gx = gy * np.cos(x)
    return gx

def sin(x):
  return Sin()(x)


def my_sin(x, threshold=0.0001):
  y = 0
  for i in range(1000000):
    c = (-1)**i / math.factorial(2 * i + 1)
    t = c * x ** (2 * i + 1)
    y = y + t
    if abs(t.data) < threshold:
      break
  return y


x = Variable(np.array(np.pi / 4))
# y = sin(x)
y = my_sin(x,threshold=0.0000000001)
y.backward()

plot_dot_graph(y, verbose=False, to_file="../png/my_sin1.png")
print(y.data)
print(x.grad)
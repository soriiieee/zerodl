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
# from dezero.core_simple import Function
from dezero.core_simple import Variable
from dezero.utils import plot_dot_graph
#(output, verbose=True, to_file="png/sample.png")


def rosenbrock(x0, x1):
  y = 100 * (x1 - x0 ** 2)** 2 + (x0 - 1)** 2
  return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

lr = 0.001
iters = 10000

for i in range(iters):
  print(x0, x1)
  
  y = rosenbrock(x0, x1)
  x0.cleargrad()
  x1.cleargrad()
  y.backward()

  x0.data -= lr* x0.grad
  x1.data -= lr * x1.grad
  
# x0 = Variable(np.array(1.0))

y = rosenbrock(x0, x1)
y.backward()

print(x0.grad, x1.grad)
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

def f(x):
  y = x ** 4 - 2 * x ** 2
  return y

def gx2(x):
  return 12 * x ** 2 - 4
  
x = Variable(np.array(1.0))
y = F.tanh(x)

x.name = "x"
y.name = "y"
y.backward(create_graph=True)

iters = 2

logs = [ y.data.flatten() ]

for i in range(iters):
  # logs.append(x.grad.data.flatten())
  gx = x.grad
  x.cleargrad()
  gx.backward(create_graph=True)

gx = x.grad
gx.name = "gx" + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file="../png/35_01_tanh.png")

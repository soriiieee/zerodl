# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [zeor kara deep learning]
# when : 2020.11.16/2020.11.29
if '__file__' in globals():  #global変数に格納されているかの確認
  import os, sys
  sys.path.append(os.path.join(os.path.dirname('__file__'), ".."))

import numpy as np
from dezero import Variable,Parameter,Model,Layer
import dezero.functions  as F
from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt
from dezero.utils import sum_to

# from dezero import Layer
import dezero.layers as L


class TwoLayerNet(Model):
  def __init__(self, hidden_size, out_size):
    super().__init__()
    self.l1 = L.Linear(hidden_size)
    self.l2 = L.Linear(out_size)
  
  def forward(self, x):
    y = F.sigmoid_simple(self.l1(x))
    y = self.l2(y)
    return y

x = Variable(np.random.randn(5, 10), name="x")
model = TwoLayerNet(100, 10)
model.plot(x)
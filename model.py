# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [zeor kara deep learning]
# when : 2020.11.16/2020.11.22
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np

import unittest
# ----------------
#---------------------------------------------------------------------------
# initial
#

class Variable:
  def __init__(self,data):
    if data is not None:
      if not isinstance(data,np.ndarray):
        raise TypeError('{} is not supported..'.format(type(data)))
    self.data = data
    self.grad = None #init
    self.creator = None #functionの登録を行うためにinstance変数を作成する

  def set_creator(self,func):
    self.creator = func

  def backward(self):
    if self.grad is None:
      self.grad = np.ones_like(self.data)
    
    funcs = [self.creator]
    while funcs:
      f = funcs.pop()
      # x,y = f.input, f.output
      gys = [ output.grad for output in f.outputs ]
      gxs = f.backward(*gys)  # kahen longht input 
      if not isinstance(gxs, tuple):
        gxs = (gxs,)

      for x, gx in zip(f.inputs, gxs):
        x.grad = gx
        if x.creator is not None:
          funcs.append(x.creator)
      x.grad = f.backward(y.grad)
      if x.creator is not None:
        funcs.append(x.creator)

class Function:
  def __call__(self,*inputs):
    # python 可変長引数の定義を行う
    # x = input.data
    xs = [ x.data for x in inputs]
    ys = self.forward(*xs)
    if not isinstance(ys,tuple):
      ys = (ys,)
    # y = self.forward(x)
    outputs = [ Variable(as_array(y)) for y in ys]
    for output in outputs:
      output.set_creator(self) #このクラスはfunctioninsタンスなので、funcを持っている
    
    self.inputs = inputs
    self.outputs = outputs
    return outputs if len(outputs) >1 else outputs[0]

  def forward(self,x):
    raise NotImplemntError()
  def backward(self,x):
    raise NotImplemntError()

class Add(Function):
  def forward(self,x0,x1):
    y = x0+x1
    return
  def backward(self, gy):
    return gy,gy

def add(x0,x1):
  return Add()(x0,x1)

class Square(Function):
  def forward(self,x):
    return x**2
  def backward(self,gy):
    x = self.input.data
    return gy * 2*x 

class Exp(Function):
  def forward(self,x):
    return np.exp(x)
  def backward(self,gy):
    x = self.input.data
    return gy* np.exp(x)

def square(x):
  f = Square()
  return f(x)

def exp(x):
  f = Exp()
  return f(x)

def numerical_diff(f,x,eps=1e-4):
  x0 = Variable(x.data - eps)
  x1 = Variable(x.data + eps)
  y0 = f(x0)
  y1 = f(x1)
  return (y1.data-y0.data)/(2*eps)

def as_array(x):
  if np.isscalar(x):
    return np.array(x)
  return x

class SquareTest(unittest.TestCase):
  def test_forward(self):
    x = Variable(np.array(2.0))
    y = square(x)
    expected = np.array(4.0)
    self.assertEqual(y.data, expected)
  
  def test_backward(self):
    x = Variable(np.array(3.0))
    y = square(x)
    y.backward()
    expected = np.array(6.0)
    self.assertEqual(x.grad, expected)

  def test_gradient_check(self):
    x = Variable(np.random.rand(1))
    y = square(x)
    y.backward()
    num_grad = numerical_diff(square, x)
    flg = np.allclose(x.grad, num_grad)
    self.assertEqual(flg)



if __name__ =="__main__":
  #-----------------------------
  # test code assert------------
  # unittest.main()
  # sys.exit()

  # x = Variable(np.array(0.5))
  # x = Variable(None)
  x0 = Variable(np.array(2))
  x1 = Variable(np.array(3))
  y = add(x0,x1)
  print(y.data)
  sys.exit()

  sys.exit()
  a = square(x)
  b = exp(a)
  y = square(b)

  #逆伝播
  y.backward()
  print(x.grad)

  # assert y.creator == C
  # assert y.creator.input ==b
  # assert y.creator.input.creator ==B
  # assert y.creator.input.creator.input ==a
  # print(x.grad)
  sys.exit()
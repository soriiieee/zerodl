# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [zeor kara deep learning]
# when : 2020.11.16/2020.11.22/ 2020.11.23
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np
import weakref #2020.11.23

import unittest
# ----------------
#---------------------------------------------------------------------------
# initial
#

# 2020.11.23 add
class Config:
  enable_backprop = True


class Variable:
  def __init__(self,data):
    if data is not None:
      if not isinstance(data,np.ndarray):
        raise TypeError('{} is not supported..'.format(type(data)))
    self.data = data
    self.grad = None #init
    self.creator = None  #functionの登録を行うためにinstance変数を作成する
    self.generation = 0

  def set_creator(self,func):
    self.creator = func
    self.generation = func.generation + 1
  
  def cleargrad(self):
    self.grad = None

  def backward(self, retain_grad=False):
    if self.grad is None:
      self.grad = np.ones_like(self.data)
    
    #-- 2020.11.23----------------------------
    funcs = []
    seen_set = set()
    def add_func(f):
      if f not in seen_set:
        funcs.append(f)
        seen_set.add(f)
        funcs.sort(key=lambda x: x.generation)
    add_func(self.creator)
    #-----------------------------------------

    while funcs:
      f = funcs.pop()
      # x,y = f.input, f.output
      # change 2020.11.22------
      # gys = [ output.grad for output in f.outputs ]
      gys = [ output().grad for output in f.outputs ]
      gxs = f.backward(*gys)  # kahen longht input 
      if not isinstance(gxs, tuple):
        gxs = (gxs,)

      for x, gx in zip(f.inputs, gxs):
        if x.grad is None:
          x.grad = gx
        else:
          x.grad = x.grad + gx
        if x.creator is not None:
          add_func(x.creator)
      
      #2020.11.23
      if not retain_grad:
        for y in f.outputs:
          y().grad = None # y is weakref...
      # x.grad = f.backward(y.grad)
      # if x.creator is not None:
      #   funcs.append(x.creator)

class Function:
  def __call__(self,*inputs):
    # python 可変長引数の定義を行う
    # x = input.data
    xs = [ x.data for x in inputs]
    ys = self.forward(*xs)
    if not isinstance(ys,tuple):
      ys = (ys,)
    # y = self.forward(x)
    outputs = [Variable(as_array(y)) for y in ys]

    if Config.enable_backprop:
      self.generation = max([x.generation for x in inputs])  #関数に(世代)の記憶を持たせる
      for output in outputs:
        output.set_creator(self) #このクラスはfunctioninsタンスなので、funcを持っている
    
    self.inputs = inputs
    # self.outputs = outputs #memory before
    self.outputs = [ weakref.ref(output) for output in outputs] #memory before
    return outputs if len(outputs) >1 else outputs[0]

  def forward(self,x):
    raise NotImplemntError()
  def backward(self,x):
    raise NotImplemntError()

class Add(Function):
  def forward(self,x0,x1):
    y = x0+x1
    return y
  def backward(self, gy):
    return gy,gy

def add(x0,x1):
  return Add()(x0,x1)

class Square(Function):
  def forward(self, x):
    y = x**2
    return y
  def backward(self,gy):
    # x = self.input.data :before change 
    x = self.inputs[0].data  #:before change
    # print(gy)
    # sys.exit()
    gx = 2 * x * gy
    return gx

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
  #-----------------------------
  # sample test------------
  x0 = Variable(np.array(1.0))
  x1 = Variable(np.array(1.0))
  t = add(x0, x1)
  y = add(x0, t)
  y.backward()
  print(y.grad, t.grad)
  print(x0.grad, x1.grad)
  sys.exit()
  x = Variable(np.array(2.0))
  a = square(x)
  y = add(square(a), square(a))
  y.backward()
  print(y.data)
  print(x.grad)
  # assert y.creator == C
  # assert y.creator.input ==b
  # assert y.creator.input.creator ==B
  # assert y.creator.input.creator.input ==a
  # print(x.grad)
  sys.exit()
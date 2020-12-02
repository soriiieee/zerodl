# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [zeor kara deep learning]
# when : 2020.11.16/22/23/28
#------------------------------------------------------------------------
# import matplotlib.pyplot as plt
import sys

import numpy as np
import weakref #2020.11.23
import unittest
import contextlib
# ----------------
#---------------------------------------------------------------------------
# initial
#

# 2020.11.23 add
class Config:
  enable_backprop = True


class Variable:
  __array_priority__ = 200 #arrayの優先度をあげるようにする
  
  def __init__(self,data,name=None):
    if data is not None:
      if not isinstance(data,np.ndarray):
        raise TypeError('{} is not supported..'.format(type(data)))
    self.data = data
    self.grad = None  #init
    self.name = name
    self.creator = None  #functionの登録を行うためにinstance変数を作成する
    self.generation = 0
  
  def __len__(self): #通常用いるlenのカスタマイズ機能を実装する仕組み(特殊メソッド)
    return len(self.data)

  def __mul__(self, other):
    return mul(self,other)

  def __repr__(self): #python のprint関数のカスタマイズ
    if self.data is None:
      return 'Variable(None)'
    p = str(self.data).replace("\n", "\n" + " " * 9)
    return 'Variable(' + p + ')'
    
  
  # 2020.11.28 add - in 
  @property #x.shape() ではなく、x.shape で取り出せる
  def shape(self):
    return self.data.shape
  @property
  def ndim(self):
    return self.data.ndim
  @property
  def size(self):
    return self.data.size
  @property
  def dtype(self):
    return self.data.dtype

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
    inputs = [ as_variable(x) for x in inputs ]# 2020.11.28

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
    return gy * np.exp(x)

class Mul(Function):
  def forward(self, x0, x1):
    y = x0 * x1
    return y 
  def backward(self, gy):
    x0, x1 = self.inputs[0].data, self.inputs[1].data
    return gy * x1, gy * x0 #2020.11.26 反対のinputをかけて出力する(定数とみなすので!)


class Neg(Function):
  def forward(self, x):
    return - x
  def backward(self, gy):
    return - gy

class Sub(Function):
  def forward(self, x0, x1):
    y = x0 - x1
    return y
  def backward(self, gy):
    return gy, -gy

class Div(Function):
  def forward(self, x0, x1):
    y = x0 / x1
    return y
  def backward(self, gy):
    x0, x1 = self.inputs[0].data, self.inputs[1].data
    gx0 = gy / x1
    gx1 = gy * (-x0 / x1 ** 2)
    return gx0, gx1

class Pow(Function):
  def __init__(self,c):
    self.c = c
  def forward(self, x):
    y = x ** self.c
    return y
  def backward(self, gy):
    x = self.inputs[0].data
    c = self.c
    gx = c * x ** (c - 1) * gy
    return gx
  
    

def add(x0, x1):
  x1 = as_array(x1)
  return Add()(x0,x1)
def square(x):
  return Square()(x)
def exp(x):
  return Exp()(x)
def mul(x0, x1):
  x1 = as_array(x1)
  return Mul()(x0, x1)
def neg(x):
  return Neg()(x)

#引き算は順番の違いで異なる結果になるので、別々に定義
def sub(x0, x1):
  x1 = as_array(x1)
  return Sub()(x0, x1)
def rsub(x0, x1):
  x1 = as_array(x1)
  return Sub()(x1, x0)
#割り算は順番の違いで異なる結果になるので、別々に定義
def div(x0, x1):
  x1 = as_array(x1)
  return Div()(x0, x1)
def rdiv(x0, x1):
  x1 = as_array(x1)
  return Div()(x1, x0)
def pow(x, c):
  return Pow(c)(x)

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

#2020.11.26
def as_variable(obj):
  if isinstance(obj, Variable):
    return obj
  return Variable(obj)


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


@contextlib.contextmanager
def using_config(name, value):
  old_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    #write pre - processig
    yield
    #write after- processig
  finally:
    setattr(Config, name, old_value)

def no_grad():
  return using_config("enable_backprop", False)

def setup_variable():
  Variable.__mul__ = mul
  Variable.__add__ = add
  Variable.__rmul__ = mul
  Variable.__radd__ = add
  Variable.__neg__ = neg
  Variable.__sub__ = sub
  Variable.__rsub__ = rsub
  Variable.__truediv__ = div
  Variable.__rtruediv__ = rdiv
  Variable.__pow__ = pow

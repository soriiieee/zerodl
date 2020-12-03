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
import dezero #2020.12.01
# ----------------
#---------------------------------------------------------------------------
# initial
#

# 2020.11.23 add ---------------------
# config
#-------------------------------------
class Config:
  enable_backprop = True

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


# 2020.11.16 add ---------------------
# Variable / Function
#-------------------------------------
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
  @property
  def T(self):
    return dezero.functions.transpose(self)

  def set_creator(self,func):
    self.creator = func
    self.generation = func.generation + 1
  
  def cleargrad(self):
    self.grad = None
  
  #2020.12.01
  def reshape(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
      shape = shape[0]
    return dezero.functions.reshape(self, shape)

  def sum(self, axis=None, keepdims=False):
    return dezero.functions.sum(self,axis,keepdims)
  
  def transpose(self):
    return dezero.functions.transpose(self)

  def backward(self, retain_grad=False, create_graph=False):
    if self.grad is None:
      # self.grad = np.ones_like(self.data) #simple_code.py
      self.grad = Variable(np.ones_like(self.data)) #simple_code.py 2020.11.29
    
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
      gys = [output().grad for output in f.outputs]
      with using_config("enable_backprop", create_graph):
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
  def forward(self, x0, x1):
    self.x0_shape,self.x1_shape = x0.shape, x1.shape #2020.12.02
    y = x0+x1
    return y

  def backward(self, gy):
    gx0,gx1 = gy,gy
    if self.x0_shape != self.x1_shape:
      gx0 = dezero.functions.sum_to(gx0,self.x0_shape)
      gx1 = dezero.functions.sum_to(gx1,self.x1_shape)
    return gx0,gx1

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

class Mul(Function):
  def forward(self, x0, x1):
    y = x0 * x1
    return y 
  def backward(self, gy):
      x0, x1 = self.inputs
      gx0 = gy * x1
      gx1 = gy * x0
      if x0.shape != x1.shape:  # for broadcast
          gx0 = dezero.functions.sum_to(gx0, x0.shape)
          gx1 = dezero.functions.sum_to(gx1, x1.shape)
      return gx0, gx1

class Neg(Function):
  def forward(self, x):
    return - x
  def backward(self, gy):
    return - gy

class Sub(Function):
  def forward(self, x0, x1):
    self.x0_shape, self.x1_shape = x0.shape, x1.shape
    y = x0 - x1
    return y

  def backward(self, gy):
      gx0 = gy
      gx1 = -gy
      if self.x0_shape != self.x1_shape:  # for broadcast
          gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
          gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
      return gx0, gx1

class Div(Function):
  def forward(self, x0, x1):
    y = x0 / x1
    return y

  def backward(self, gy):
      x0, x1 = self.inputs
      gx0 = gy / x1
      gx1 = gy * (-x0 / x1 ** 2)
      if x0.shape != x1.shape:  # for broadcast
          gx0 = dezero.functions.sum_to(gx0, x0.shape)
          gx1 = dezero.functions.sum_to(gx1, x1.shape)
      return gx0, gx1

class Pow(Function):
  def __init__(self,c):
    self.c = c
  def forward(self, x):
    y = x ** self.c
    return y
  def backward(self, gy):
    # x = self.inputs
    x, = self.inputs
    c = self.c

    # print(x,c)
    gx = c * x ** (c - 1) * gy
    return gx
  
def add(x0, x1):
  x1 = as_array(x1)
  return Add()(x0,x1)
def square(x):
  return Square()(x)
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


class Parameter(Variable):
  pass

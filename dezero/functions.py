import numpy as np
from dezero.core import Function



#------------------------------------
# Functions : class

class Sin(Function):
  def forward(self, x):
    y = np.sin(x)
    return y
  
  def backward(self, gy):
    x, = self.inputs #tuple
    gx = gy * cos(x)
    return gx

class Cos(Function):
  def forward(self, x):
    y = np.cos(x)
    return y
  
  def backward(self, gy):
    x, = self.inputs #tuple
    gx = gy * -sin(x)
    return gx

class Tanh(Function):
  def forward(self, x):
    y = np.tanh(x)
    return y
  
  def backward(self, gy):
    y = self.outputs[0]() 
    gx = gy * (1 - y * y)
    return gx

#2020.12.02
class Reshape(Function):
  def __init__(self, shape):
    self.shape = shape
  
  def forward(self, x):
    self.x_shape = x.shape
    y = x.reshape(self.shape)
    return y
  
  def backward(self, gy):
    return reshape(gy,self.x_shape)

class Transpose(Function):
  def forward(self, x):
    y = np.transpose(x)
    return y
  
  def backward(self, gy):
    gx = transpose(gy)
    return gx

class Sum(Function):
  def __init__(self, axis, keepdims):
    self.axis = axis
    self.keepdims = keepdims


  def forward(self, x):
    self.x_shape = x.shape
    y = x.sum(axis=self.axis, keepdims=self.keepdims)

    return y
  
  def backward(self, gy):
    gy = utils.reshape_sum_backward(sy, self.x_shape, self.axis, self.keepdims)
    gx = broadcast_to(gy, self.x_shape)
    return gx

class BroadcastTo(Function):
  def __init__(self, shape):
    self.shape = shape
  def forward(self, x):
    self.x_shape = x.shape
    y = np.broadcast_to(x, self.shape)
    return y
  
  def backward(self, gy):
    gx = sum_to(gy, self.x_shape)
    return gx

class SumTo(Function):
  def __init__(self, shape):
    self.shape = shape
  
  def forward(self, x):
    self.x_shape = x.shape
    y = utils.sum_to(x, self.shape)
    return y

  def backward(self, gy):
    gx = broadcast_to(gy, self.x_shape)
    return gx




    

#------------------------------------
# Functions : instance
def sin(x):
  return Sin()(x)
def cos(x):
  return Cos()(x)
def tanh(x):
  return Tanh()(x)

def reshape(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return Reshape(shape)(x)

def transpose(x):
  return Transpose()(x)

def sum(x,axis=None,keepdims = False):
  return Sum(axis, keepdims)(x)
  
def broadcast_to(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return BroadcastTo(shape)(x)

def sum_to(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return SumTo(shape)(x)


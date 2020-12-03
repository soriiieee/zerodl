import numpy as np
import dezero
from dezero import utils
from dezero.core import Function, Variable, as_variable, as_array



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
    gx = transpose(gy) #local make finction
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
    gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
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

def broadcast_to(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return BroadcastTo(shape)(x)

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

def sum_to(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return SumTo(shape)(x)

class Exp(Function):
  def forward(self,x):
    return np.exp(x)
  def backward(self, gy):
    y = self.outputs[0]()  # weakref
    gx = gy * y
    return gx

def exp(x):
  return Exp()(x)


class MatMul(Function):
  def forward(self, x, W):
    y = x.dot(W)
    return y
  
  def backward(self, gy):
    x, W = self.inputs
    gx = matmul(gy, W.T)
    gW = matmul(x.T, gy)
    return gx, gW

#2020.12.03
class MeanSquaredError(Function):
  def forward(self, x0, x1):
    diff = x0 - x1
    y = (diff ** 2).sum() / len(diff)
    return y
  def backward(self, gy):
    x0, x1 = self.inputs
    diff = x0 - x1
    gy = broadcast_to(gy, diff.shape)

    gx0 = gy * diff * (2./len(diff))
    gx1 = -gx0
    return gx0, gx1
    
def mean_squared_error(x0, x1):
  return MeanSquaredError()(x0, x1)
  
def linear_simple(x, W, b=None):
  x, W = as_varible(x), as_varible(W)
  t = matmul(x, W)
  if b is None:
    return t
  
  y = t + b
  t.data = None
  return y

# Function super ---- from github...
class Linear(Function):
  def forward(self, x, W, b):
    y = x.dot(W)
    if b is not None:
      y += b
    return y

  def backward(self, gy):
    x, W, b = self.inputs
    #gradient...
    gb = None if b.data is None else sum_to(gy, b.shape)
    gx = matmul(gy, W.T)
    gW = matmul(x.T, gy)
    return gx, gW, gb
    
def linear(x, W, b=None):
    return Linear()(x, W, b)

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
  

def matmul(x, W):
  return MatMul()(x, W)
  
# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================

class Sigmoid(Function):
  def forward(self, x):
    y = 1 / (1 + dezero.core.exp(-x))
    return y
  
  def backward(self, gy):
    y = self.outputs[0]()
    gx = gy * y * (1 - y)
    return gx

def sigmoid_simple(x):
  x = as_variable(x)
  y = 1 / (1 + exp(-x))
  return y

def sigmoid(x):
  return Sigmoid()(x)
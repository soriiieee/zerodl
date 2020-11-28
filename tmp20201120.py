# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [zeor kara deep learning]
# when : 2020.11.16/


from model import *

if __name__ == "__main__":
  Variable.__mul__ = mul
  Variable.__add__ = add

  a = Variable(np.array(3.0))
  b = Variable(np.array(2.0))
  c = Variable(np.array(1.0))

  # y = add(mul(a, b), c)
  y = a*b +c
  y.backward()


  print(y)
  print(a.grad)
  print(b.grad)
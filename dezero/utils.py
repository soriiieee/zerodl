if '__file__' in globals():  #global変数に格納されているかの確認
  import os, sys
  sys.path.append(os.path.join(os.path.dirname('__file__'), ".."))

import numpy as np
from dezero import Variable
import os
import subprocess
# from dezero.utils import get_dot_graph

def _dot_var(v, verbose=False):
  dot_var = '{} [label="{}", color=orange , style=filled]\n'
  name = '' if v.name is None else v.name

  # verbose=True and データもある場合
  if verbose and v.data is not None:
    if v.name is not None:
      name += ': '
    name += str(v.shape) + ' ' + str(v.dtype)
  
  return dot_var.format(id(v), name)

def _dot_func(f):
  dot_func = '{} [label="{}", color=lightblue , style=filled, shape=box]\n'
  txt = dot_func.format(id(f), f.__class__.__name__)
  dot_edge = '{} -> {}\n'

  #inputのデータを関数のinputから取り出し、繋げるスクリプト
  for x in f.inputs:
    txt += dot_edge.format(id(x), id(f))
  for y in f.outputs:
    txt += dot_edge.format(id(f), id(y()))  #y=weak_ref
  return txt

def get_dot_graph(output, verbose):
  txt = ''
  funcs = []
  seen_set = set()
  def add_func(f):
    if f not in seen_set:
      funcs.append(f)
      seen_set.add(f)
  
  add_func(output.creator)
  txt += _dot_var(output, verbose)
  while funcs:
    func = funcs.pop()
    txt += _dot_func(func)
    for x in func.inputs:
      txt += _dot_var(x, verbose)
      if x.creator is not None:
        add_func(x.creator)
  
  return 'digraph g{\n' + txt + '}'
  
def plot_dot_graph(output, verbose=True, to_file="png/sample.png"):
  dot_graph = get_dot_graph(output, verbose)
  
  # 1: dot dataをファイル保存する
  tmp_dir = os.path.join(os.path.expanduser("~"), '.dezero')
  if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)
  
  gragh_path = os.path.join(tmp_dir, "tmp_graph.dot")
  with open(gragh_path, "w") as f:
    f.write(dot_graph)

  # 2: dot dataをファイル保存する
  extension = os.path.splitext(to_file)[1][1:]
  cmd = 'dot {} -T {} -o {}'.format(gragh_path, extension, to_file)
  subprocess.run(cmd, shell=True)
    
if __name__ == "__main__":
  print("start..")
  x0 = Variable(np.array(1.0))
  x1 = Variable(np.array(1.0))

  y = x0 + x1
  txt = _dot_func(y.creator)
  print(txt)
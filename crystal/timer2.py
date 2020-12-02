# -*- coding: utf-8 -*-
import sys,os
import numpy as np
from datetime import datetime, timedelta


class Timer:
  def __init__(self):
    self.isRun = False  #計測状態
    self.count = 0
    self.laps = []
    self.splits = []

    self.lap_data = None
    self.split_data = None

  def start(self):
    if self.isRun == False:
      self.st = datetime.now()
      self.isRun = True
      if self.count == 0:
        self.st0 = self.st
    else:
      print("running...")

  def stop(self):
    if self.isRun == True:
      self.et = datetime.now()
      self.isRun = False
      self.count +=1
      #lap and split data append -> [],[]
      self.lap()
      self.split()
    else:
      print("stopping !")

  def lap(self):
    if self.isRun == False:
      if self.count != 0:
        self.lap_data = self.et - self.st
        self.laps.append(self.lap_data)
      else:
        print("no lap ...")
    else:
      print("running...")

  def split(self):
    if self.isRun == False:
      if self.count != 0:
        self.split_data = self.et - self.st0
        self.splits.append(self.split_data)
      else:
        print("no split ...")
    else:
      print("running...")


  def reset(self):
    self.__init__()

#実行コマンド
if __name__ == "__main__":
  timer = Timer()

  # for i in range(10):
  timer.start()
  k=0
  for i in range(100):
    for j in range(10000):
      k += j ** 2
    print(k)
  timer.stop()

  print("lap:", timer.lap_data, "split:", timer.split_data)
  print()
    # print("laps", timer.laps)
  



    
  
# -*- coding: utf-8 -*-
import sys,os
import numpy as np
from datetime import datetime, timedelta


class Timer:
  def __init__(self):
    #全体
    self.st0 = 0  #最初のタイマー時刻
    #個別
    self.st = 0 #start - time (datetime)
    self.et = 0 #end - time (datetime)
    self.lapn = []  #計測するラップの名前
    self.lapc = 0  #計測するラップのカウント
    self.laps = []  #計測するラップの時間リスト
    #状態
    self.isRun = False #計測状態

  def push(self, name=None):
    if name is None: 
      sys.exit("[error] input lap name..")
    if self.isRun == False:
      self.st = datetime.now()
      if self.lapc == 0:
        self.st0 = self.st
      self.isRun = True
      self.lapc += 1
    else:
      self.et = datetime.now()
      lap = self.et - self.st
      self.lapn.append(name)
      self.laps.append(lap)
      self.isRun = False

  #timer のリセット機能
  def reset(self):
    self.__init__()
  
  def show(self, file_path=None):
    if not file_path:
      print(f"results ------")
      for name ,lap in zip(self.lapn, self.laps):
        print(f"lapname:{name} --> laptime:{lap}")
      print(f"N_laps:{self.lapc} --> all_time:{self.et - self.st0}")
    else:
      with open(file_path, "w") as f:
        f.write(f"lapname,laptime\n")
        for name, lap in zip(self.lapn, self.laps):
          f.write(f"{name},{lap}\n")
        

#実行コマンド
if __name__ == "__main__":
  timer = Timer()
  for i in range(5):
    timer.push(i)
    for j in range(10000):
      j += j ** 2
    timer.push(i)
  sys.exit()
  
  # timer.show(file_path="./log_timer.csv")
  timer.reset()
  for i in range(3):
    timer.push(i)
    for j in range(10000):
      j += j ** 2
    timer.push(i)

  timer.show(file_path="./log_timer.csv")
  
  
  



    
  
from datetime import datetime
import time
import os
import matplotlib.pyplot as plt
import numpy as np

class Running_Time:

    def __init__(self):
        self.start_time = time.time()

    def get_running_time(self):
        current_time = time.time()
        # print("Total Execution Time:")
        return f"{((current_time - self.start_time)//60):.0f}:{(current_time - self.start_time)%60:.0f} min:sec"

class Execution_Time:

    executions_time = None
    executions_time_with_microsec = None
    __isInitialized = False

    def __init__(self):
        Execution_Time.__set_executions_time()

    # def get_executions_time(self, add_microsec=False):
    #     if(add_microsec): return self.executions_time
    #     else: return self.executions_time_with_microsec

    # def _set_executions_time(self, add_microsec=False):
    #     time = datetime.today()
    #     self.executions_time = time.strftime('%Y-%m-%d_%H-%M-%S.%f')
    #     self.executions_time_with_microsec = time.strftime('%Y-%m-%d_%H-%M-%S')
    #     Execution_Time.executions_time = self.executions_time
    #     Execution_Time.executions_time_with_microsec = self.executions_time_with_microsec

    @staticmethod
    def initialize():
        Execution_Time.__set_executions_time()

    @staticmethod
    def get_executions_time(add_microsec=False):
        if(not Execution_Time.__isInitialized):
            Execution_Time.__set_executions_time()
        if(add_microsec): return Execution_Time.executions_time
        else: return Execution_Time.executions_time_with_microsec

    @staticmethod
    def __set_executions_time(add_microsec=False):
        if(Execution_Time.__isInitialized):
           return 
        time = datetime.today()
        Execution_Time.executions_time = time.strftime('%Y-%m-%d_%H-%M-%S.%f')
        Execution_Time.executions_time_with_microsec = time.strftime('%Y-%m-%d_%H-%M-%S')
        Execution_Time.__isInitialized = True


def get_new_executions_time(add_microsec=False):
    if(add_microsec): return datetime.today().strftime('%Y-%m-%d_%H-%M-%S.%f')
    else: return datetime.today().strftime('%Y-%m-%d_%H-%M-%S')        



class SavingPath():

  def __init__(self, args, saveModelDirPath):
    SavingPath._lr = args.lr
    if(args.debug_mode):
      SavingPath._model_name = "_Debugging"
    else:
      if(args.model_name):
        SavingPath._model_name = "_" + args.model_name
      else:
        SavingPath._model_name = ""

    if not os.path.exists(saveModelDirPath):
      os.makedirs(saveModelDirPath)

    SavingPath._path = saveModelDirPath + time.strftime('%b-%d-%Y_%H.%M.%S', time.localtime()) + SavingPath._model_name + "_lr-" + str(SavingPath._lr)
    
  @staticmethod
  def get_path(epoch = -1, suffix = ""):
    if(epoch == -1):
      return SavingPath._path + suffix
    else:
      return SavingPath._path +"_epoch-" + str(epoch) + suffix

class PlotData():
  def __init__(self, length):
    self.index = 0
    self.__maxindex = length
    self.valuesList = np.zeros((2, length))
  def isFull(self):
    return self.index >= self.__maxindex

class ModelStatistics():

  def __init__(self, length, plotNames):
    self.fig, self.ax = plt.subplots()
    self.listDict = {}
    for name in plotNames:
      self.listDict[name] = PlotData(length)

  def AddData(self, name, x, y):
    if(self.listDict[name].isFull()):
      raise Exception("Array is full. Possibly initialized with the wrong length")
    self.listDict[name].valuesList[0][self.listDict[name].index] = x
    self.listDict[name].valuesList[1][self.listDict[name].index] = y
    self.listDict[name].index = self.listDict[name].index + 1

  def Show(self):
    self.ax.cla()
    for name in self.listDict.keys():
      self.ax.plot(self.listDict[name].valuesList[0][:self.listDict[name].index], self.listDict[name].valuesList[1][:self.listDict[name].index], label=name)
    self.fig.legend()
    self.fig.show()

  def Save(self, path):
    self.ax.cla()
    for name in self.listDict.keys():
      self.ax.plot(self.listDict[name].valuesList[0][:self.listDict[name].index], self.listDict[name].valuesList[1][:self.listDict[name].index], label=name)
    self.fig.legend()
    self.fig.savefig(path)
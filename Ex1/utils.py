import time
import os
import matplotlib.pyplot as plt
import numpy as np

class Running_Time:
  '''Class that stores the initialized time and returns the running time'''
  def __init__(self):
    self.start_time = time.time()

  def get_running_time(self):
    current_time = time.time()
    # print("Total Execution Time:")
    return f"{((current_time - self.start_time)//60):.0f}:{(current_time - self.start_time)%60:.0f} min:sec"

class SavingPath():
  ''' class to creates the correct saving path '''

  def __init__(self, args, saveDirName):
    
    SavingPath._lr = args.lr
    if(args.debug_mode):
      SavingPath._model_name = "_Debugging"
    else:
      if(args.model_name):
        SavingPath._model_name = "_" + args.model_name
      else:
        SavingPath._model_name = ""

    dir_path = os.path.dirname(os.path.realpath(__file__))
    savePath = os.path.join(dir_path, saveDirName)

    if not os.path.exists(savePath):
      os.makedirs(savePath)

    SavingPath._path = os.path.join(savePath, time.strftime('%b-%d-%Y_%H.%M.%S', time.localtime()) + SavingPath._model_name + "_lr-" + str(SavingPath._lr))
    
  @staticmethod
  def get_path(epoch = -1, suffix = ""):
    ''' A static method that return the correct saving path '''
    if(epoch == -1):
      return SavingPath._path + suffix
    else:
      return SavingPath._path +"_epoch-" + str(epoch) + suffix

class PlotData():
  '''class that serves as a kind of tuple, for easier using the ModelStatistics class'''

  def __init__(self, length):
    self.index = 0
    self.__maxindex = length
    self.valuesList = np.zeros((2, length))
  def isFull(self):
    return self.index >= self.__maxindex

class ModelStatistics():
  '''Class that stores events to be used in a plot'''

  def __init__(self, length, plotNames):
    self.fig, self.ax = plt.subplots()
    self.listDict = {}
    for name in plotNames:
      self.listDict[name] = PlotData(length)

  def addData(self, name, x, y):
    if(self.listDict[name].isFull()):
      raise Exception("Array is full. Possibly initialized with the wrong length")
    self.listDict[name].valuesList[0][self.listDict[name].index] = x
    self.listDict[name].valuesList[1][self.listDict[name].index] = y
    self.listDict[name].index = self.listDict[name].index + 1

  def show(self):
    self.ax.cla()
    for name in self.listDict.keys():
      self.ax.plot(self.listDict[name].valuesList[0][:self.listDict[name].index], self.listDict[name].valuesList[1][:self.listDict[name].index], label=name)
    self.fig.legend()
    self.fig.show()

  def save(self, path, title=""):
    self.ax.cla()
    if(title):
      self.fig.suptitle(title, fontsize=12)
    for name in self.listDict.keys():
      self.ax.plot(self.listDict[name].valuesList[0][:self.listDict[name].index], self.listDict[name].valuesList[1][:self.listDict[name].index], label=name)
    self.fig.legend()
    self.fig.savefig(path)
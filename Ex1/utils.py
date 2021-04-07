from datetime import datetime
import time

class Running_Time:

    def __init__(self):
        self.start_time = time.time()

    def print_running_time(self):
        current_time = time.time()
        print("Total Execution Time:")
        print(f"--- {str((current_time - self.start_time)//60)}:{str(current_time - self.start_time)} min:sec ---")

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


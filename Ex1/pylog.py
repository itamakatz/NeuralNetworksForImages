import os
import logging
from utils import Execution_Time
import matplotlib.pyplot as plt

# ========================= Debugging functions ===================== #

INDEX = 0
LOGGER = None
LOG_LEVEL = logging.DEBUG
LOG_DIR_PATH = r'../Logs/'

def reset_log():
	global INDEX
	INDEX = 0

# ========================= Logging ===================== #

def initialize_log(level = logging.DEBUG):
	reset_log();
	
	global LOGGER

	# create LOGGER
	LOGGER = logging.getLogger('Main Logger')
	LOGGER.setLevel(level)

	# create console handler and set level to debug
	ch = logging.StreamHandler()
	
	if not os.path.exists(LOG_DIR_PATH):
		os.makedirs(LOG_DIR_PATH)

	fh = logging.FileHandler(f'{LOG_DIR_PATH}tensorflow_log_{Execution_Time.get_executions_time(True)}.log')
	ch.setLevel(level)
	ch.setLevel(level)

	# create formatter
	formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

	# add formatter to ch and fh
	ch.setFormatter(formatter)
	fh.setFormatter(formatter)

	# add ch and fh to LOGGER
	LOGGER.addHandler(ch)
	LOGGER.addHandler(fh)

	# code examples:
		# LOGGER.debug('debug message')
		# LOGGER.info('info message')
		# LOGGER.warning('warn message')
		# LOGGER.error('error message')
		# LOGGER.critical('critical message')

def get_logging_func(log_level):
	'''
	Returning a the correct logging function associated with the level
	'''
	global LOGGER

	if (LOGGER.level > logging.DEBUG): return
	return {
		logging.DEBUG: 		LOGGER.debug,
		logging.INFO: 		LOGGER.info,
		logging.WARNING: 	LOGGER.warning,
		logging.ERROR: 		LOGGER.error,
		logging.CRITICAL: 	LOGGER.critical,
	}[log_level]

# ►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► STRING FORMATTING ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

# region : STRING FORMATTING

def shape_str(obj, pre_txt="Shape"):
	return_str = pre_txt + ": "
	return_str += str(obj.shape) + ". "
	return return_str
	
def size_str(obj, pre_txt="Size"):
	return_str = pre_txt + ": "
	return_str += str(obj.size) + ". "
	return return_str
	
def len_str(obj, pre_txt="Length"):
	return_str = pre_txt + ": "
	return_str += str(len(obj)) + ". "
	return return_str
	
def type_str(obj, pre_txt="Type"):
	return_str = str(pre_txt) + ": "
	return_str += str(type(obj)) + ". "
	return return_str

def format_name(name=None):
	global INDEX
	INDEX += 1
	return_str = str(INDEX) + ". "
	if(name is not None):
		return_str += name + " - "
	return return_str

# endregion : STRING FORMATTING

# ►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► COMMON LOGGING ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

# region : COMMON LOGGING

def common_log_print(obj, log_level, name=None):
	
	global LOGGER

	if (LOGGER.level > log_level): return
	log_func = get_logging_func(log_level)

	print_str = format_name(name)
	print_str += str(obj)

	log_func(print_str)

def common_log_type(obj, log_level, name=None):
	
	global LOGGER

	if (LOGGER.level > log_level): return
	log_func = get_logging_func(log_level)
	
	print_str = format_name(name)
	print_str += type_str(obj)

	log_func(print_str)
	
def common_log_numpy(numpy_array, log_level, name=None):
	
	global LOGGER

	if (LOGGER.level > log_level): return
	log_func = get_logging_func(log_level)

	print_str = format_name(name)
	print_str += shape_str(numpy_array)
	print_str += size_str(numpy_array)

	log_func(print_str)

def common_log_list(list_obj, log_level, name=None):
	
	global LOGGER

	if (LOGGER.level > log_level): return
	log_func = get_logging_func(log_level)
	
	print_str = format_name(name)
	print_str += len_str(list_obj)
	if(len(list_obj) > 0):
		print_str += type_str(list_obj[0], pre_txt="Elements Type: ")
	else:
		print_str += "."
	
	log_func(print_str)
	
def common_log_tensor(tensor_obj, log_level, name=None):
	
	global LOGGER

	if (LOGGER.level > log_level): return
	log_func = get_logging_func(log_level)
	
	print_str = format_name(name)
	print_str += shape_str(tensor_obj)
	
	log_func(print_str)

def common_log_image(gray, log_level, name=None):
	
	global LOGGER

	if (LOGGER.level > logging.DEBUG): return
	
	print_str = format_name(name)
	plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1, label=print_str)
	plt.show()

# endregion : COMMON LOGGING

# ►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► DEBUG LOGGING ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

# region : DEBUG LOGGING
	
def debug_print(obj, name=None):
	common_log_print(obj, logging.DEBUG, name=None)

def debug_type(obj, name=None):
	common_log_type(obj, logging.DEBUG, name=None)
	
def debug_numpy(numpy_array, name=None):
	common_log_numpy(numpy_array, logging.DEBUG, name=None)

def debug_list(list_obj, name=None):
	common_log_list(list_obj, logging.DEBUG, name=None)
	
def debug_tensor(tensor_obj, name=None):
	common_log_tensor(tensor_obj, logging.DEBUG, name=None)
	
def debug_image(gray, name=None):
	common_log_image(gray, logging.DEBUG, name=None)

# endregion : DEBUG LOGGING

# ►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► INFO LOGGING ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

# region : INFO LOGGING
	
def info_print(obj, name=None):
	common_log_print(obj, logging.INFO, name=None)

def info_type(obj, name=None):
	common_log_type(obj, logging.INFO, name=None)
	
def info_numpy(numpy_array, name=None):
	common_log_numpy(numpy_array, logging.INFO, name=None)

def info_list(list_obj, name=None):
	common_log_list(list_obj, logging.INFO, name=None)
	
def info_tensor(tensor_obj, name=None):
	common_log_tensor(tensor_obj, logging.INFO, name=None)
	
def info_image(gray, name=None):
	common_log_image(gray, logging.INFO, name=None)

# endregion : INFO LOGGING

# ►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

# Levels Numeric value:
# 	CRITICAL = 50
# 	ERROR = 40
# 	WARNING = 30
# 	INFO = 20
# 	DEBUG = 10
# 	NOTSET = 0

initialize_log(LOG_LEVEL)
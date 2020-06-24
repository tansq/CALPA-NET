# @Time    : 2019-05-11 09:54
# @Author  : WeiLong Wu
# @File    : setup_general_srnet.py
# @Function: common config, import general libs and setup path

# library imports
import math
import tensorflow as tf
from generator_psm import *
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from functools import partial
import ConfigParser

from utils_multistep_lr_srnet import *

# Input
# dataset path
Dataset_Path = '../../datas/'  # The Input path depends on you.

# thinet dependence path
Thinet_Dependence_Path = '../../datas/thinet_data_dependence/'

# Output
# SRNet outpath base
SRNet_OutPath_Base = '/xxx/xxx/CALPA_output/'  # The Output path depends on you.

# For SRNet output ----- ----- ----- Jpeg domain
SRNet_QF75_JU_40_OutPath = SRNet_OutPath_Base + 'exp/srnet_qf75_juniward_40/'

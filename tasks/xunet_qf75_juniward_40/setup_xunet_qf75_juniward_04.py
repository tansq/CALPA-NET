# @Time    : 2019-05-22 20:06
# @Author  : wuwl
# @File    : setup_xunet_qf75_juniward_04.py
# @Function: Task related configurations, library imports, and path definitions

import os
import sys

sys.path.append('../../models/')
from XuNet2 import *

sys.path.append('../../libs/psm/')
from detect_xunet import *

# juniward40
JUNIWARD_40_TRAIN_COVER_DIR = Dataset_Path + 'qf75/seed123/juniward40/train_cover.txt'
JUNIWARD_40_TRAIN_STEGO_DIR = Dataset_Path + 'qf75/seed123/juniward40/train_stego.txt'
JUNIWARD_40_VALID_COVER_DIR = Dataset_Path + 'qf75/seed123/juniward40/val_cover.txt'
JUNIWARD_40_VALID_STEGO_DIR = Dataset_Path + 'qf75/seed123/juniward40/val_stego.txt'
JUNIWARD_40_TEST_COVER_DIR = Dataset_Path + 'qf75/seed123/juniward40/test_cover.txt'
JUNIWARD_40_TEST_STEGO_DIR = Dataset_Path + 'qf75/seed123/juniward40/test_stego.txt'

# train config
train_batch_size = 32
valid_batch_size = 40
max_iter = 320000
train_interval = 875  # 1 epochs
valid_interval = 875
test_interval = 875
save_interval = 875
num_runner_threads = 10
starter_learning_rate = 0.001
train_ori_xunet_log_path = XuNet_QF75_JU_40_OutPath + 'output/train_output/'  # path for a log direcotry

# test config
test_batch_size = 40
test_ori_xunet_path = train_ori_xunet_log_path

# detect config
THINET_COVER_DIR = Thinet_Dependence_Path + 'Input_JU_75_0.4/cover.txt'
THINET_STEGO_DIR = Thinet_Dependence_Path + 'Input_JU_75_0.4/stego.txt'

# valid_acc: The best performance of the original XuNet model, used as a reference in the detect process
valid_acc = 0.9609

detect_ori_srnet_path = train_ori_xunet_log_path
LOAD_CKPT = detect_ori_srnet_path + 'Model_xxxx00.ckpt'  # loading from a specific checkpoint
LOG_detect_output = XuNet_QF75_JU_40_OutPath + 'output/detect_output/'
if not os.path.exists(LOG_detect_output):
    os.makedirs(LOG_detect_output)


def initial_config_file(config_log):
    config_raw2 = ConfigParser.ConfigParser()
    config_raw2.add_section('rate_block1')
    config_raw2.add_section('rate_block2')
    config_raw2.add_section('rate_block3')
    config_raw2.add_section('rate_block4')
    config_raw2.add_section('rate_block5')

    config_raw2.add_section('acc_block1')
    config_raw2.add_section('acc_block2')
    config_raw2.add_section('acc_block3')
    config_raw2.add_section('acc_block4')
    config_raw2.add_section('acc_block5')

    config_raw2.set('rate_block1', 'layer1', '0.0')
    config_raw2.set('rate_block2', 'layer1', '0.0')
    config_raw2.set('rate_block3', 'layer1', '0.0')
    config_raw2.set('rate_block4', 'layer1', '0.0')
    config_raw2.set('rate_block5', 'layer1', '0.0')

    config_raw2.set('rate_block1', 'layer2', '0.0')
    config_raw2.set('rate_block2', 'layer2', '0.0')
    config_raw2.set('rate_block3', 'layer2', '0.0')
    config_raw2.set('rate_block4', 'layer2', '0.0')
    config_raw2.set('rate_block5', 'layer2', '0.0')

    config_raw2.set('acc_block1', 'layer1', '0.0')
    config_raw2.set('acc_block2', 'layer1', '0.0')
    config_raw2.set('acc_block3', 'layer1', '0.0')
    config_raw2.set('acc_block4', 'layer1', '0.0')
    config_raw2.set('acc_block5', 'layer1', '0.0')

    config_raw2.set('acc_block1', 'layer2', '0.0')
    config_raw2.set('acc_block2', 'layer2', '0.0')
    config_raw2.set('acc_block3', 'layer2', '0.0')
    config_raw2.set('acc_block4', 'layer2', '0.0')
    config_raw2.set('acc_block5', 'layer2', '0.0')
    config_raw2.write(open(config_log, 'w'))


# open log files
is_detect = False     # before detecting, change to True
if is_detect == True:
    f_log_valid_Thin = open(LOG_detect_output + 'log_valid_Thi.txt', 'a+')
    f_log_S_Thin = open(LOG_detect_output + 'log_S_Thi.txt', 'a+')
    f_log_valid_Li = open(LOG_detect_output + 'log_valid_Li.txt', 'a+')
    f_log_S_Li = open(LOG_detect_output + 'log_S_Li.txt', 'a+')

    config_log_05_thinet = LOG_detect_output + 'xunet_juniward_04_threshold05-thinet.cfg'
    config_log_05_li = LOG_detect_output + 'xunet_juniward_04_threshold05-li.cfg'
    if not os.path.exists(config_log_05_thinet):
        initial_config_file(config_log_05_thinet)
    if not os.path.exists(config_log_05_li):
        initial_config_file(config_log_05_li)

# initial_train
config_log_05_thinet = LOG_detect_output + 'xunet_juniward_04_threshold05-thinet.cfg'
config_log_05_li = LOG_detect_output + 'xunet_juniward_04_threshold05-li.cfg'

initial_train_ori_xunet_log_path_pruned_05 = XuNet_QF75_JU_40_OutPath + 'output/initial_train_output'

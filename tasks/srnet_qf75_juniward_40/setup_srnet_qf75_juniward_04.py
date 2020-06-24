# @Time    : 2019-05-11 10:01
# @Author  : WeiLong Wu
# @File    : setup_srnet_qf75_juniward_04.py
# @Function: Task related configurations, library imports, and path definitions

import os
import sys

sys.path.append('../../models/')
from SRNet import *

sys.path.append('../../libs/psm/')
from detect_srnet import *

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
max_iter = 500000
train_interval = 875
valid_interval = 875
test_interval = 875
save_interval = 875
num_runner_threads = 10
learning_rate = [0.001, 0.0001]
learning_rate_boundaries = [400000]  # learning rate adjustment at iteration 400K
train_ori_srnet_log_path = SRNet_QF75_JU_40_OutPath + 'output/train_output/'  # path for a log direcotry

# test config
test_batch_size = 40
test_ori_srnet_path = train_ori_srnet_log_path

# detect config
THINET_COVER_DIR = Thinet_Dependence_Path + 'Input_JU_75_0.4/cover.txt'
THINET_STEGO_DIR = Thinet_Dependence_Path + 'Input_JU_75_0.4/stego.txt'

# valid_acc: The best performance of the original SRNet model, used as a reference in the detect process
valid_acc = 0.90

detect_ori_srnet_path = train_ori_srnet_log_path
LOAD_CKPT = detect_ori_srnet_path + '/Model_xxxx00.ckpt'  # loading from a specific checkpoint
LOG_detect_output = SRNet_QF75_JU_40_OutPath + 'output/detect_output/'
if not os.path.exists(LOG_detect_output):
    os.makedirs(LOG_detect_output)


def initial_config_file(config_log):
    config_raw2 = ConfigParser.ConfigParser()
    config_raw2.add_section('Thinet_rate')
    config_raw2.add_section('Thinet_acc')
    config_raw2.add_section('L1_rate')
    config_raw2.add_section('L1_acc')
    config_raw2.set('Thinet_rate', 'layer3', '0.0')
    config_raw2.set('Thinet_rate', 'layer4', '0.0')
    config_raw2.set('Thinet_rate', 'layer5', '0.0')
    config_raw2.set('Thinet_rate', 'layer6', '0.0')
    config_raw2.set('Thinet_rate', 'layer7', '0.0')
    config_raw2.set('Thinet_rate', 'layer8', '0.0')
    config_raw2.set('Thinet_rate', 'layer9', '0.0')
    config_raw2.set('Thinet_rate', 'layer10', '0.0')
    config_raw2.set('Thinet_rate', 'layer11', '0.0')
    config_raw2.set('Thinet_rate', 'layer12', '0.0')

    config_raw2.set('Thinet_acc', 'layer3', '0.0')
    config_raw2.set('Thinet_acc', 'layer4', '0.0')
    config_raw2.set('Thinet_acc', 'layer5', '0.0')
    config_raw2.set('Thinet_acc', 'layer6', '0.0')
    config_raw2.set('Thinet_acc', 'layer7', '0.0')
    config_raw2.set('Thinet_acc', 'layer8', '0.0')
    config_raw2.set('Thinet_acc', 'layer9', '0.0')
    config_raw2.set('Thinet_acc', 'layer10', '0.0')
    config_raw2.set('Thinet_acc', 'layer11', '0.0')
    config_raw2.set('Thinet_acc', 'layer12', '0.0')

    config_raw2.set('L1_rate', 'layer8', '0.0')
    config_raw2.set('L1_rate', 'layer9', '0.0')
    config_raw2.set('L1_rate', 'layer10', '0.0')
    config_raw2.set('L1_rate', 'layer11', '0.0')
    config_raw2.set('L1_rate', 'layer12', '0.0')

    config_raw2.set('L1_acc', 'layer8', '0.0')
    config_raw2.set('L1_acc', 'layer9', '0.0')
    config_raw2.set('L1_acc', 'layer10', '0.0')
    config_raw2.set('L1_acc', 'layer11', '0.0')
    config_raw2.set('L1_acc', 'layer12', '0.0')
    config_raw2.write(open(config_log, 'w'))


# open log files
is_detect = False       # before detecting, change to True
if is_detect == True:
    f_log_valid_Thin = open(LOG_detect_output + '/log_valid_Thi.txt', 'a+')
    f_log_S_Thin = open(LOG_detect_output + '/log_S_Thi.txt', 'a+')
    f_log_valid_Li = open(LOG_detect_output + '/log_valid_Li.txt', 'a+')
    f_log_S_Li = open(LOG_detect_output + '/log_S_Li.txt', 'a+')
    config_log_02 = LOG_detect_output + '/srnet_juniward_04_threshold02.cfg'
    config_log_05 = LOG_detect_output + '/srnet_juniward_04_threshold05.cfg'
    config_log_10 = LOG_detect_output + '/srnet_juniward_04_threshold10.cfg'
    config_log_20 = LOG_detect_output + '/srnet_juniward_04_threshold20.cfg'
    config_log_50 = LOG_detect_output + '/srnet_juniward_04_threshold50.cfg'
    if not os.path.exists(config_log_02):
        initial_config_file(config_log_02)
    if not os.path.exists(config_log_05):
        initial_config_file(config_log_05)
    if not os.path.exists(config_log_10):
        initial_config_file(config_log_10)
    if not os.path.exists(config_log_20):
        initial_config_file(config_log_20)
    if not os.path.exists(config_log_50):
        initial_config_file(config_log_50)

# initial_train
initial_train_ori_srnet_log_path_pruned_05 = SRNet_QF75_JU_40_OutPath + 'output/initial_train_output'


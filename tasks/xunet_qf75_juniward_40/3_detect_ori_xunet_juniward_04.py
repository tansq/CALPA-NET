# @Time    : 2019-05-18 19:22
# @Author  : wuwl
# @File    : 3_detect_ori_xunet_juniward_04.py
# @Function: Using Thinet and Li to detect the original XuNet

from setup_xunet_qf75_juniward_04 import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # set a GPU (with GPU Number)

VALID_COVER_DIR = JUNIWARD_40_VALID_COVER_DIR
VALID_STEGO_DIR = JUNIWARD_40_VALID_STEGO_DIR

valid_gen = partial(gen_train_or_valid_psm, VALID_COVER_DIR, VALID_STEGO_DIR)

with open(VALID_COVER_DIR) as f:
    valid_cover = f.readlines()
    valid_cover_list = [a.strip() for a in valid_cover]

valid_ds_size = len(valid_cover_list) * 2

thinet_gen = partial(gen_train_or_valid_psm, THINET_COVER_DIR, THINET_STEGO_DIR)

# arg2, arg3 chose from (1,1)、(1,2)、(2,1)、(2,2)、(3,1)、(3,2)、(4,1)、(4,2)、(5,1)、(5,2);
# arg2 means block number
# arg3 means layer number in this block
thinet_detect_on_cov_layer(XuNet2, 1, 1, valid_gen, valid_batch_size, valid_ds_size, thinet_gen,
                           LOAD_CKPT, f_log_valid_Thin, config_log_05_thinet, valid_acc)

# arg2 chose from 1, 2, 3, 4, 5; arg3 still 1
# arg2, arg3 just like above
# Li_detect_on_conv_layer(XuNet2, 1, 1, valid_gen, valid_batch_size, valid_ds_size,
#                         LOAD_CKPT, f_log_valid_Li, config_log_05_li, valid_acc)

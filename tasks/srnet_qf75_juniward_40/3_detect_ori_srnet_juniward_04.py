# @Time    : 2019-05-11 10:58
# @Author  : WeiLong Wu
# @File    : 3_detect_ori_srnet_juniward_04.py
# @Function: Using Thinet and Li to detect the original SRNet


from setup_srnet_qf75_juniward_04 import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

valid_gen = partial(gen_train_or_valid_psm, JUNIWARD_40_VALID_COVER_DIR, JUNIWARD_40_VALID_STEGO_DIR)

with open(JUNIWARD_40_VALID_COVER_DIR) as f:
    valid_cover = f.readlines()
    valid_cover_list = [a.strip() for a in valid_cover]

valid_ds_size = len(valid_cover_list) * 2

print 'valid_ds_size: %i' % valid_ds_size
if valid_ds_size % valid_batch_size != 0:
    raise ValueError("change batch size for validation!")
thinet_gen = partial(gen_train_or_valid_psm, THINET_COVER_DIR, THINET_STEGO_DIR)

config_log_list = []
config_log_list.append(config_log_02)
config_log_list.append(config_log_05)
config_log_list.append(config_log_10)
config_log_list.append(config_log_20)
config_log_list.append(config_log_50)

# layer : 3 - 12
detect_one_layer_Thinet(SRNet, valid_gen, valid_batch_size, valid_ds_size, thinet_gen, LOAD_CKPT, 3, f_log_valid_Thin,
                        f_log_S_Thin, config_log_list, valid_acc)

# layer : 8 - 12
# detect_one_layer_Li(SRNet, valid_gen, valid_batch_size, valid_ds_size, LOAD_CKPT, 8, f_log_valid_Li, f_log_S_Li,
#                     config_log_list, valid_acc)
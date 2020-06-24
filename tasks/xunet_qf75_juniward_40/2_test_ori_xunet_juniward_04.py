# @Time    : 2019-05-11 10:09
# @Author  : wuwl
# @File    : 2_test_ori_xunet_juniward_04.py
# @Function:  Using juniward 0.4bpnzac to test the original XuNet


from setup_xunet_qf75_juniward_04 import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # set a GPU (with GPU Number)

# Testing
# Cover and Stego directories for testing
TEST_COVER_DIR = JUNIWARD_40_TEST_COVER_DIR
TEST_STEGO_DIR = JUNIWARD_40_TEST_STEGO_DIR

LOAD_CKPT = test_ori_xunet_path + 'Model_xxx.ckpt'  # loading from a specific checkpoint
test_gen = partial(gen_train_or_valid_psm, TEST_COVER_DIR, TEST_STEGO_DIR)

with open(TEST_COVER_DIR) as f:
    test_cover = f.readlines()
    test_cover_list = [a.strip() for a in test_cover]

test_ds_size = len(test_cover_list) * 2

print 'test_ds_size: %i' % test_ds_size
if test_ds_size % test_batch_size != 0:
    raise ValueError("change batch size for testing!")
xunet_test(XuNet2, test_gen, test_batch_size, test_ds_size, LOAD_CKPT)

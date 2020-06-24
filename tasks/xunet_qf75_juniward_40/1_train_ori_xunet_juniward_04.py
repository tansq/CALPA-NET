# @Time    : 2019-05-11 10:05
# @Author  : wuwl
# @File    : 1_train_ori_xunet_juniward_04.py
# @Function: Using juniward 0.4bpnzac to train the original XuNet

from setup_xunet_qf75_juniward_04 import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # set a GPU (with GPU Number)

# Cover and Stego directories for training and validation. For the spatial domain put cover and stego images in their
# corresponding direcotries. For the JPEG domain, decompress images to the spatial domain without rounding to integers and
# save them as '.mat' files with variable name \"im\". Put the '.mat' files in thier corresponding directoroies. Make sure
# all mat files in the directories can be loaded in Python without any errors.
TRAIN_COVER_DIR = JUNIWARD_40_TRAIN_COVER_DIR
TRAIN_STEGO_DIR = JUNIWARD_40_TRAIN_STEGO_DIR
VALID_COVER_DIR = JUNIWARD_40_VALID_COVER_DIR
VALID_STEGO_DIR = JUNIWARD_40_VALID_STEGO_DIR
TEST_COVER_DIR = JUNIWARD_40_TEST_COVER_DIR
TEST_STEGO_DIR = JUNIWARD_40_TEST_STEGO_DIR

train_gen = partial(gen_flip_and_rot_psm, TRAIN_COVER_DIR, TRAIN_STEGO_DIR)
valid_gen = partial(gen_train_or_valid_psm, VALID_COVER_DIR, VALID_STEGO_DIR)
test_gen = partial(gen_train_or_valid_psm, TEST_COVER_DIR, TEST_STEGO_DIR)

LOG_DIR = train_ori_xunet_log_path

load_path = None  # training from scratch

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

with open(TRAIN_COVER_DIR) as f:
    train_cover = f.readlines()
    train_cover_list = [a.strip() for a in train_cover]

with open(VALID_COVER_DIR) as f:
    val_cover = f.readlines()
    val_cover_list = [a.strip() for a in val_cover]

with open(TEST_COVER_DIR) as f:
    test_cover = f.readlines()
    test_cover_list = [a.strip() for a in test_cover]

train_ds_size = len(train_cover_list) * 2
valid_ds_size = len(val_cover_list) * 2
test_ds_size = len(test_cover_list) * 2

print 'train_ds_size: %i' % train_ds_size
print 'valid_ds_size: %i' % valid_ds_size
print 'test_ds_size: %i' % test_ds_size

if valid_ds_size % valid_batch_size != 0:
    raise ValueError("change batch size for validation")

# optimizer = AdamaxOptimizer
optimizer = tf.train.MomentumOptimizer
xunet_train(XuNet2, train_gen, valid_gen, train_batch_size, valid_batch_size, valid_ds_size,
            test_gen, test_batch_size, test_interval, test_ds_size, optimizer, starter_learning_rate, train_interval,
            valid_interval, max_iter,
            save_interval, LOG_DIR, num_runner_threads, load_path)

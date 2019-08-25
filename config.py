import os

# 验证码字符
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

CHAR_SET = NUMBER + ALPHABET
CHAR_SET_LEN = len(CHAR_SET)
CAPTCHA_LEN = 4  # 验证码长度

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

TRAIN_COUNT = 10000  # 生成训练集数量
TEST_COUNT = 1000  # 生成测试集数量

TRAIN_DATASET_PATH = "dataset" + os.path.sep + "train"
TEST_DATASET_PATH = "dataset" + os.path.sep + "test"
PREDICT_DATA_PATH = "dataset" + os.path.sep + "predict"

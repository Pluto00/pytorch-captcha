from captcha.image import ImageCaptcha
from PIL import Image
import random
import time
import config
import os


def random_code():
    """
    生成随机的验证码
    :return:
        随机验证码(string)
    """
    text = []
    for _ in range(config.CAPTCHA_LEN):
        text.append(random.choice(config.CHAR_SET))
    return ''.join(text)


def random_image():
    """
    生成验证码图片
    :return:
        text:验证码(string)
        image:验证码图片(file)
    """
    text = random_code()
    image = Image.open(ImageCaptcha().generate(text))
    return text, image


def captcha_gen(path, count):
    """
    图片保存格式：验证码_时间戳.png
    :param path: 图片保存路径
    :param count: 数量
    """
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count):
        now = str(int(time.time()))
        text, image = random_image()
        filename = text + '_' + now + '.png'
        image.save(path + os.path.sep + filename)
        print('saved %d : %s' % (i + 1, filename))


if __name__ == '__main__':
    captcha_gen(config.TRAIN_DATASET_PATH, config.TRAIN_COUNT)
    captcha_gen(config.TEST_DATASET_PATH, config.TEST_COUNT)

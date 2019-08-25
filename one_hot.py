import numpy as np
import config


def char2pos(ch):
    """
    把字符转换成one-hot的位置
    :param ch: 字符,大写字母或数字
    :return:
        pos:位置
    """
    if '0' <= ch <= '9':
        pos = ord(ch) - ord('0')
    elif 'A' <= ch <= 'Z':
        pos = ord(ch) - ord('A') + 10
    else:
        raise ValueError("Captcha is wrong")
    return pos


def one_hot_encode(text):
    """
    把验证码进行one-hot编码
    :param text: 验证码
    :return:
        vector:验证码的one—hot形式
    """
    vector = np.zeros(config.CHAR_SET_LEN * config.CAPTCHA_LEN, dtype=float)
    for i, ch in enumerate(text):
        idx = i * config.CHAR_SET_LEN + char2pos(ch)
        vector[idx] = 1.0
    return vector


def one_hot_decode(vector):
    """
    把one_hot解码成字符
    :param vector: 验证码的one—hot形式
    :return:
        text:验证码
    """
    char_pos = vector.nonzero()[0]
    text = []
    for i, char_idx in enumerate(char_pos):
        char_idx %= config.CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        else:
            raise ValueError('Vector is wrong')
        text.append(chr(char_code))
    return ''.join(text)


if __name__ == '__main__':
    s = 'A6S2'
    vec = one_hot_encode(s)
    print(vec.reshape(4, -1))
    res = one_hot_decode(vec)
    print(res)

# 项目说明
本项目利用pytorch框架搭建CNN网络来完成验证码识别这一任务，本项目用到的训练集有100000张图片，测试集有10000张图片，识别的正确率达到了94.500000 %。


# 验证码
项目中用到的验证码都是用captcha库来生成的，具体的图片如下：

![captcha](https://raw.githubusercontent.com/Pluto00/pytorch-captcha/master/dataset/test/0MQM_1566728514.png)

# CNN网络
CNN网络有五层的卷积层加上两层的全连接层构成。

# 使用说明
- 首先配置config文件，设置验证码的长度和字符以及训练集和测试集的数量，然后运行 image_gen.py 文件生成数据集
- 然后运行 train_model.py 文件即可开始训练模型
- 运行 test_model.py 文件可以测试模型的正确率

# 模型缺陷
对于字母O和数字0难以区分，其他的字符都能较好的识别

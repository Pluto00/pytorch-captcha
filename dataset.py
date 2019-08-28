import os
import config
import one_hot
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    """
    继承Dataset类，实现自由的数据读取
    """

    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        """
        获取数据集的长度
        """
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        """
        根据索引序号获取图片和标签, 相当于重载运算符[],
        :param idx: 索引
        :return:
            image:灰度图片
            label:图片的标签，one_hot形式
        """
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        label = one_hot.text2vec(image_name.split('_')[0])
        return image, label


transform = transforms.Compose([
    transforms.Grayscale(),  # 转灰度图片
    transforms.ToTensor()
])


def get_train_data_loader():
    dataset = MyDataset(config.TRAIN_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=True)


def get_test_data_loader():
    dataset = MyDataset(config.TEST_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=False)


def get_predict_data_loader():
    dataset = MyDataset(config.PREDICT_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)

"""
通过继承torch.utils.data.Dataset的这个抽象类，我们可以定义好我们需要的数据类
    __len__(self) 定义当被len()函数调用时的行为（返回容器中元素的个数）
    __getitem__(self)定义获取容器中指定元素的行为，相当于self[key]，即允许类对象可以有索引操作。


通过torch.utils.data.DataLoader类来定义一个新的迭代器，用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入
DataLoader参数说明:
* dataset (Dataset): 加载数据的数据集
* batch_size (int, optional): 每批加载多少个样本
* shuffle (bool, optional): 设置为“真”时,在每个epoch对数据打乱.（默认：False）
* sampler (Sampler, optional): 定义从数据集中提取样本的策略,返回一个样本
* batch_sampler (Sampler, optional): like sampler, but returns a batch of indices at a time 返回一批样本. 与atch_size, shuffle, sampler和 drop_last互斥.
* num_workers (int, optional): 用于加载数据的子进程数。0表示数据将在主进程中加载​​。（默认：0）
* collate_fn (callable, optional): 合并样本列表以形成一个 mini-batch.  #　callable可调用对象
* pin_memory (bool, optional): 如果为 True, 数据加载器会将张量复制到 CUDA 固定内存中,然后再返回它们.
* drop_last (bool, optional): 设定为 True 如果数据集大小不能被批量大小整除的时候, 将丢掉最后一个不完整的batch,(默认：False).
* timeout (numeric, optional): 如果为正值，则为从工作人员收集批次的超时值。应始终是非负的。（默认：0）
* worker_init_fn (callable, optional): If not None, this will be called on each worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading. (default: None)．
"""

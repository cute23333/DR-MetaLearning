import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from PIL import Image
import os

from mnist_m import MNISTM
from data.synthetic_digits import SyntheticDigits

# 尝试加载图像但是transform只做resize，以及采样10000张图像
# 耶！成功了
# 现在每个batch的图像都是[0,1]的tensor，32*32，训练集10000张，测试集不变

def load_mnist_data(batch_size, sample_size=10000, root_dir=r'./data/MNIST_data'):
    """
    加载MNIST数据集并创建DataLoader。

    参数:
    batch_size (int): 每个批次的样本数量。
    sample_size (int): 从训练集中采样的样本数量，默认为10000。
    root_dir (str): 数据集的存储路径。

    返回:
    train_loader (DataLoader): 训练集的DataLoader。
    test_loader (DataLoader): 测试集的DataLoader。
    """
    # 自定义转换函数，将单通道图像复制三次变成三通道
    def to_three_channels(tensor):
        return torch.cat((tensor, tensor, tensor), dim=0)

    # 定义数据转换，包括调整大小、转换为Tensor、复制通道以及归一化
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整图像大小到32x32
        transforms.ToTensor(),
        transforms.Lambda(to_three_channels),  # 复制单通道图像三次变成三通道
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化三通道
        # transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))  # 归一化三通道
    ])

    # 加载MNIST数据集
    full_train_dataset = datasets.MNIST(root=root_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root_dir, train=False, download=True, transform=transform)

    # 设置随机数种子以确保结果的可重复性
    np.random.seed(42)
    torch.manual_seed(42)

    # 从完整的训练集中随机采样sample_size个样本的索引
    indices = np.random.choice(len(full_train_dataset), sample_size, replace=False)
    sampled_train_dataset = Subset(full_train_dataset, indices)

    # 创建DataLoader
    train_loader = DataLoader(dataset=sampled_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_svhn_data(batch_size, sample_size=10000, root_dir=r'./data/SVHN'):
    """
    加载SVHN数据集并创建DataLoader。

    参数:
    batch_size (int): 每个批次的样本数量。
    root_dir (str): 数据集的存储路径。

    返回:
    train_loader (DataLoader): 训练集的DataLoader。
    test_loader (DataLoader): 测试集的DataLoader。
    """
    # 定义数据转换，包括调整大小、转换为Tensor以及归一化
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整图像大小到32x32
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化三通道
    ])

    # 加载SVHN数据集
    full_train_dataset = datasets.SVHN(root=root_dir, split='train', download=True, transform=transform)
    # print(len(train_dataset)) # 73257
    test_dataset = datasets.SVHN(root=root_dir, split='test', download=True, transform=transform)

    # 设置随机数种子以确保结果的可重复性
    np.random.seed(42)
    torch.manual_seed(42)

    # 从完整的训练集中随机采样sample_size个样本的索引
    indices = np.random.choice(len(full_train_dataset), sample_size, replace=False)
    sampled_train_dataset = Subset(full_train_dataset, indices)

    # 创建DataLoader
    train_loader = DataLoader(dataset=sampled_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_mnistm_data(batch_size, sample_size=10000, root_dir=r'./data/MNIST-M'):
    """
    加载MNIST-M数据集并创建DataLoader。

    参数:
    batch_size (int): 每个批次的样本数量。
    root_dir (str): 数据集的存储路径。
    train_transform (transforms.Compose): 训练集的数据转换。
    test_transform (transforms.Compose): 测试集的数据转换。
    download (bool): 是否下载数据集。

    返回:
    train_loader (DataLoader): 训练集的DataLoader。
    test_loader (DataLoader): 测试集的DataLoader。
    """
    # 定义训练集和测试集的数据转换
    transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 调整图像大小到32x32
            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化三通道
    ])
    
    # 加载MNIST-M数据集
    full_train_dataset = MNISTM(root=root_dir, train=True, transform=transform, download=True)
    test_dataset = MNISTM(root=root_dir, train=False, transform=transform, download=True)
    
    # 设置随机数种子以确保结果的可重复性
    np.random.seed(42)
    torch.manual_seed(42)

    # 从完整的训练集中随机采样sample_size个样本的索引
    indices = np.random.choice(len(full_train_dataset), sample_size, replace=False)
    sampled_train_dataset = Subset(full_train_dataset, indices)

    # 创建DataLoader
    train_loader = DataLoader(sampled_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# class SyntheticDigitsDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'): # train or valid
        """
        初始化SYN数据集。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.image_dirs = [os.path.join(root_dir, 'imgs_'+mode, str(i)) for i in range(10)]

    def __len__(self):
        """
        返回数据集中的图像数量。
        """
        return sum(len(os.listdir(dir)) for dir in self.image_dirs)

    def __getitem__(self, idx):
        """
        获取单个图像和标签。
        """
        for i, img_dir in enumerate(self.image_dirs):
            if idx < len(os.listdir(img_dir)):
                img_path = os.path.join(img_dir, os.listdir(img_dir)[idx])
                image = Image.open(img_path).convert('RGB')
                label = i
                break
            else:
                idx -= len(os.listdir(img_dir))

        if self.transform:
            image = self.transform(image)

        return image, label


# def load_syn_data(batch_size, root_dir=r'./data/SYN/synthetic_digits'):
    """
    加载SYN数据集并创建DataLoader。

    参数:
    batch_size (int): 每个批次的样本数量。
    root_dir (str): 数据集的存储路径

    返回:
    dataset (Dataset): SYN数据集的Dataset实例。

    注意，SYN dataset只有10000张训练集，所以不需要下采样
    猜测作者下采样10000就是为了迁就SYN数据集
    """
    transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = SyntheticDigitsDataset(root_dir=root_dir, transform=transform, mode='train')
    test_dataset = SyntheticDigitsDataset(root_dir=root_dir, transform=transform, mode='valid')
    
    # 创建DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_syn_data(batch_size, sample_size=10000, root_dir=r'./data/'):
    # 定义训练集和测试集的数据转换
    transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 调整图像大小到32x32
            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化三通道
    ])
    
    full_train_dataset = SyntheticDigits(root=root_dir, train=True, transform=transform, download=True)
    test_dataset = SyntheticDigits(root=root_dir, train=False, transform=transform, download=True)

    # 设置随机数种子以确保结果的可重复性
    np.random.seed(42)
    torch.manual_seed(42)

    # 从完整的训练集中随机采样sample_size个样本的索引
    indices = np.random.choice(len(full_train_dataset), sample_size, replace=False)
    sampled_train_dataset = Subset(full_train_dataset, indices)

    train_loader = DataLoader(sampled_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    # # 使用函数加载数据 - MNIST
    # train_loader, test_loader = load_mnist_data(batch_size=64)

    # 使用函数加载数据 - SVHN
    # train_loader, test_loader = load_svhn_data(batch_size=64)

    # 使用函数加载数据
    # train_loader, test_loader = load_mnistm_data(batch_size=64)

    # 使用函数加载数据
    train_loader, test_loader = load_syn_data(batch_size=64) # 注意这里SYN返回的是PIL Image………………改！为了统一还是改成归一化的tensor

    print(len(train_loader))
    for data, target in train_loader:
        print(data.shape, target.shape)
        # print(data[0])
        # print(target)
        break
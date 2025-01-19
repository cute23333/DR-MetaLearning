# 目前先复现基础的数字识别版本

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms, models
# from tqdm import tqdm

from data_loader_trans2 import load_mnist_data, load_svhn_data, load_mnistm_data, load_syn_data
from model import ResNet18
from transformation_set import batch_apply_transformation


def main():
    parser = argparse.ArgumentParser(description='Continual Domain Adaptation')
    
    # 定义参数
    # 先实现简单的digit，DR
    parser.add_argument('--task_type', type=str, default='digit recognition', help='the type of current task, include [digit recognition, PACS, Semantic scene segmentation]')
    parser.add_argument('--method', type=str, default='DR', help='include DR (domain randomization) and Meta-DR (meta-learning method in paper)')
    parser.add_argument('--protocols', type=int, default=1, help='the protocol of digit recognition, P1 or P2')

    # params for digit recognition
    parser.add_argument('--repeat_times', type=int, default=3, help='repeat n times and calc avg, std')
    parser.add_argument('--resize', type=int, default=(32,32), help='resize input images')
    parser.add_argument('--steps', type=int, default=3000, help='train H gradient steps for each domain')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--first_learning_rate', type=float, default=3e-4, help='use this learning rate when training the first domain')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate after the first domain')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gama', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.1, help='these three are hyper-params')
    # 添加更多参数...

    # 解析参数
    args = parser.parse_args()

    # 检查GPU是否可用并指定GPU编号为3
    device = torch.device("cuda:3" if torch.cuda.is_available() and torch.cuda.device_count() > 3 else "cpu")
    print("Using device:", device)

    # 模型构建
    model = ResNet18().to(device)
    
    # 训练准备
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.first_learning_rate)

    datasets = ['MNIST','MNIST-M','SYN','SVHN']

    # 训练过程
    for dataset in datasets:
        # 加载数据集
        # 怎么把PIL的transform_set加进去？？？把batch的图像tensor转回numpy再应用转换。。。
        if dataset == 'MNIST':
            train_loader, test_loader = load_mnist_data(batch_size=64)
        elif dataset == 'MNIST-M':
            train_loader, test_loader = load_mnistm_data(batch_size=64)
        elif dataset == 'SYN':
            train_loader, test_loader = load_syn_data(batch_size=64)
        elif dataset == 'SVHN':
            train_loader, test_loader = load_svhn_data(batch_size=64)

        # model = ResNet18().to(device) ###
        # optimizer = optim.Adam(model.parameters(), lr=args.first_learning_rate) ###

        # 手动调整学习率
        if dataset == 'MNIST-M':
            new_lr = args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f'Learning rate adjusted to {new_lr}')
        model.train()
        
        iter_cnt = 0
        # 这里先实现DR的功能
        for epoch in range(20): # 10000张图像，batch_size=64，等于19.2个epoch
            print('Epoch ',epoch)
            # step_cnt = 0
            for images, labels in train_loader:
                # step_cnt += 1
                iter_cnt += 1

                # 将数据移到指定的GPU
                images, labels = images.to(device), labels.to(device)
                
                # 无DR
                # optimizer.zero_grad()
                # outputs = model(images)
                # loss = criterion(outputs, labels)
                # loss.backward()
                # optimizer.step()

                # DR部分
                # image transformation
                # 将Tensor批次转换为PIL Image批次
                # print(images.shape)
                transformed_tensor_batch = batch_apply_transformation(images, 1, device)

                optimizer.zero_grad()
                DR_outputs = model(transformed_tensor_batch)
                loss = criterion(DR_outputs, labels)
                loss.backward()
                optimizer.step()
                # if step_cnt % 20 == 0:
                    # print(f'Epoch {epoch}, step',step_cnt, 'finished')
                
                # 计数 3000 steps
                if iter_cnt == args.steps:
                    break
            if iter_cnt == args.steps:
                break
            

        # 验证过程
        model.eval()
        with torch.no_grad():
            # 计算验证集上的性能
            correct = 0
            total = 0
            for images, labels in test_loader:
                # 将数据移到指定的GPU
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Test Accuracy for {dataset}: {100 * correct / total:.2f}%')
        
        # break
    
    # 遗忘后的测试
    for dataset in datasets:
        if dataset == 'MNIST':
            _, test_loader = load_mnist_data(batch_size=64)
        elif dataset == 'MNIST-M':
            _, test_loader = load_mnistm_data(batch_size=64)
        elif dataset == 'SYN':
            _, test_loader = load_syn_data(batch_size=64)
        elif dataset == 'SVHN':
            _, test_loader = load_svhn_data(batch_size=64)
        
        # 测试遗忘后的准确率
        # 验证过程
        model.eval()
        with torch.no_grad():
            # 计算验证集上的性能
            correct = 0
            total = 0
            for images, labels in test_loader:
                # 将数据移到指定的GPU
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Final Accuracy for {dataset}: {100 * correct / total:.2f}%')
    

if __name__ == "__main__":
    main()

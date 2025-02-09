# 目前先复现基础的数字识别版本 Meta DR

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import learn2learn as l2l   # 我决定用learn2learn的maml来实现meta-dr

from data_loader_trans2 import load_mnist_data, load_svhn_data, load_mnistm_data, load_syn_data
from model import ResNet18
from transformation_meta import batch_apply_transformation

def main():
    parser = argparse.ArgumentParser(description='Continual Domain Adaptation')
    
    # 定义参数
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
    parser.add_argument('--alpha', type=float, default=0.1, help='these three are hyper-params')  # 这个alpha好抽象啊，为什么比学习率大这么多，一下子偏成另一个新模型
    # 添加更多参数...

    # 解析参数
    args = parser.parse_args()

    # 检查GPU是否可用并指定GPU编号
    device = torch.device("cuda:3" if torch.cuda.is_available() and torch.cuda.device_count() > 2 else "cpu")
    print("Using device:", device)

    # 模型构建 这里改一下，用learn2learn的maml
    model = ResNet18().to(device)
    model = l2l.algorithms.MAML(model, lr=args.alpha)
    
    # 训练准备
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.first_learning_rate)

    datasets = ['MNIST','MNIST-M','SYN','SVHN']

    # 训练过程
    for i in range(len(datasets)):
        dataset = datasets[i]
        # 加载数据集   咳咳，这里四个函数比较丑陋，以后可以改一改优雅一点
        if dataset == 'MNIST':
            train_loader, test_loader = load_mnist_data(batch_size=64)
        elif dataset == 'MNIST-M':
            train_loader, test_loader = load_mnistm_data(batch_size=64)
        elif dataset == 'SYN':
            train_loader, test_loader = load_syn_data(batch_size=64)
        elif dataset == 'SVHN':
            train_loader, test_loader = load_svhn_data(batch_size=64)

        # 手动调整学习率，第二个数据集开始减小学习率。这个很有用
        if i == 1:
            new_lr = args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f'Learning rate adjusted to {new_lr}')

        model.train()

        train_iterator = iter(train_loader)
        meta_iterator = iter(train_loader)
        
        # Meta DR
        for t in range(args.steps): # 3000 steps
            if t % 300 == 0:
                print('Step',t)
            
            try:
                # (x_hat, y_hat)
                images, labels = next(train_iterator)
                images, labels = images.to(device), labels.to(device)
                # (x, y)
                images_2, labels_2 = next(meta_iterator)
                images_2, labels_2 = images_2.to(device), labels_2.to(device)

                # theta_hat更新  这里的transformation比较灵活，可以改成torch的批量操作
                transformed_tensor_batch = batch_apply_transformation(images, 1, device)

                # θ^Tt​←θt−α∇θ​LT​(T(x^),y^​;θt)
                optimizer.zero_grad()
                task_model = model.clone()
                adaptation_loss = criterion(task_model(transformed_tensor_batch), labels)
                task_model.adapt(adaptation_loss)

                # 这里就有一个问题，theta t 本身的loss是放到evalutation loss里，还是单独计算？
                transformed_tensor_batch2 = batch_apply_transformation(images_2, 1, device)
                
                loss_backward = criterion(task_model(images_2), labels_2)
                loss_forward = criterion(task_model(transformed_tensor_batch2), labels_2)
                loss_current = criterion(model(images_2), labels_2)
                evaluation_loss = loss_current + args.beta * loss_backward + args.gama * loss_forward
                evaluation_loss.backward()
                optimizer.step()
                
            except  StopIteration:
                # print('重新装载')
                train_iterator = iter(train_loader)
                meta_iterator = iter(train_loader)


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
    
    torch.save(model.state_dict(), 'checkpoints.pth')
    

if __name__ == "__main__":
    main()

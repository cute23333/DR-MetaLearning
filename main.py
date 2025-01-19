# 目前先复现基础的数字识别版本 Meta DR

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader_trans2 import load_mnist_data, load_svhn_data, load_mnistm_data, load_syn_data
from model import ResNet18
from transformation_meta import batch_apply_transformation

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
    parser.add_argument('--alpha', type=float, default=0.1, help='these three are hyper-params')  # 这个alpha好抽象啊，为什么比学习率大这么多，一下子偏成另一个新模型
    # 添加更多参数...

    # 解析参数
    args = parser.parse_args()

    # 检查GPU是否可用并指定GPU编号为3
    device = torch.device("cuda:2" if torch.cuda.is_available() and torch.cuda.device_count() > 2 else "cpu")
    print("Using device:", device)

    # 模型构建
    model = ResNet18().to(device)
    
    # 训练准备
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.first_learning_rate)

    datasets = ['MNIST','MNIST-M','SYN','SVHN']

    # 训练过程
    for i in range(len(datasets)):
        dataset = datasets[i]
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

        # 手动调整学习率，第二个数据集开始减小学习率。这个很有用
        if i == 1:
            new_lr = args.learning_rate
            # emmm，我的optimizer会不会没有更新，还对最开始那个param做更新
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

                # print(labels == labels_2)  # 每次iter(loader)都会重新shuffle，所以每次images和images_2不一样

                # theta_hat更新  这里transformation需要改！
                transformed_tensor_batch = batch_apply_transformation(images, 1, device)

                optimizer.zero_grad()
                loss = criterion(model(images_2), labels_2)
                loss.backward()
                # clone()的特性值得深思 https://blog.csdn.net/weixin_43199584/article/details/106876679
                grads = [param.grad.clone() for param in model.parameters()]  # (2,1)

                optimizer.zero_grad()
                # 对 model 的参数进行深拷贝
                # theta_t = [param.clone().detach() for param in model.parameters()]
                theta_t = [param.clone() for param in model.parameters()]

                loss_meta = criterion(model(transformed_tensor_batch), labels)
                loss_meta.backward()
                # grads_meta = [param.grad.clone() for param in model.parameters()]  # (1,1)
                # optimizer.step()
                # 啊！这里要用alpha
                new_lr = args.alpha
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

                optimizer.step()

                if i==0:
                    new_lr = args.first_learning_rate
                else:
                    new_lr = args.learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                # emm还不如eta呢
                # 说明确实是梯度传播的问题。

                transformed_tensor_batch2 = batch_apply_transformation(images_2, 1, device)
                optimizer.zero_grad()
                loss_backward = criterion(model(images_2), labels_2)
                loss_backward.backward()
                grads_backward = [param.grad.clone() for param in model.parameters()]

                optimizer.zero_grad()
                loss_forward = criterion(model(transformed_tensor_batch2), labels_2)
                loss_forward.backward()
                grads_forward = [param.grad.clone() for param in model.parameters()]

                with torch.no_grad():
                    if i==0:
                        lr = args.first_learning_rate
                    else:
                        lr = args.learning_rate
                    for param, grad1, grad2, grad3 in zip(theta_t, grads, grads_backward, grads_forward):
                        param -= lr * (grad1 + args.beta*grad2 + args.gama*grad3)  # 更新拷贝的参数
                        # 这里有问题。grad2和grad3是对theta_hat的梯度，回传到theta_t还差一步。
                
                
                for model_param, updated_param in zip(model.parameters(), theta_t):
                    model_param.data.copy_(updated_param)
                
                '''
                optimizer.zero_grad()
                loss_meta = criterion(model(transformed_tensor_batch),labels)
                loss_meta.backward()  # 获得了计算theta_hat的梯度，但是不更新
                
                # 计算hat_theta_T，不用torch.no_grad，便于之后梯度回传
                theta_t = model.state_dict()  # 这里是浅拷贝，实质上还是原本的weights tensor …… 嗯？没有拷贝梯度过来？
                hat_theta_T = {k: v - args.alpha * v.grad for k,v in theta_t.items()}  # 小心计算图不释放然后炸了。。测一下看需不需要手动释放计算图

                optimizer.zero_grad()

                outputs = model(images_2)
                current_task_loss = criterion(outputs, labels_2)
                
                # 用新模型计算backward loss和forward loss
                # model.load_state_dict(hat_theta_T)   # 这里有个问题，这个是深拷贝，会不会切断梯度传播。
                for k,v in hat_theta_T.items():
                    model.state_dict()[k].data.copy_(v)
                
                print('is equal??', theta_t == model.state_dict())  # 看一下copy_会不会影响到原来的theta_t
                exit(0)
                backward_loss = criterion(model(images_2), labels_2)

                transformed_tensor_batch2 = batch_apply_transformation(images_2, 1, device)
                forward_loss = criterion(model(transformed_tensor_batch2), labels_2)

                loss = current_task_loss + args.beta * backward_loss + args.gama * forward_loss
                model.load_state_dict(theta_t)  # 要更新原模型的梯度
                loss.backward()
                optimizer.step()

                '''

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

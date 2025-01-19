import torch
import torchvision

# 定义ResNet-18模型，替换第一层以适应MNIST数据集
class ResNet18(torch.nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        original_resnet18 = torchvision.models.resnet18(pretrained=False)

        # 输入图像大小为32*32，3通道
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = original_resnet18.bn1
        self.relu = original_resnet18.relu
        self.maxpool = original_resnet18.maxpool
        self.layer1 = original_resnet18.layer1
        self.layer2 = original_resnet18.layer2
        self.layer3 = original_resnet18.layer3
        self.layer4 = original_resnet18.layer4
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 定义一个函数来打印每一层的输出形状
def print_shape(tensor, layer_name):
    print(f"{layer_name} output shape: {tensor.size()}")


if __name__ == '__main__':
    # 创建模型实例
    model = ResNet18()

    # 创建一个随机输入张量，模拟32x32的图像，batch size为1，channel为1
    input_tensor = torch.randn(1, 3, 32, 32)
    # 测试每一层的输出形状
    x = input_tensor
    print_shape(x, "Input")
    x = model.conv1(x)
    print_shape(x, "After conv1")
    x = model.bn1(x)
    print_shape(x, "After bn1")
    x = model.relu(x)
    print_shape(x, "After relu")
    x = model.maxpool(x)
    print_shape(x, "After maxpool")

    x = model.layer1(x)
    print_shape(x, "After layer1")
    x = model.layer2(x)
    print_shape(x, "After layer2")
    x = model.layer3(x)
    print_shape(x, "After layer3")
    x = model.layer4(x)
    print_shape(x, "After layer4")

    x = model.avgpool(x)
    print_shape(x, "After avgpool")
    x = x.view(x.size(0), -1)
    print_shape(x, "After view")
    x = model.fc(x)
    print_shape(x, "After fc")

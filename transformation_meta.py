from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import random
import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import os

def sample_transform(transformation_set):
    if transformation_set == 1:
        brightness = random.uniform(0.2, 1.8)
        color = random.uniform(0.2, 1.8)
        contrast = random.uniform(0.2, 1.8)
        solarize_level = random.randint(75, 255)
        ifsolar = random.random()
        ifgray = random.random()
        ifinver = random.random()
        return [brightness,color,contrast,solarize_level,ifsolar,ifgray,ifinver]
    if transformation_set == 2:
        brightness = random.uniform(0.2, 1.8)
        color = random.uniform(0.2, 1.8)
        contrast = random.uniform(0.2, 1.8)
        angle = random.uniform(-60, 60)
        solarize_level = random.randint(75, 255)
        ifsolar = random.random()
        ifgray = random.random()
        ifinver = random.random()
        return [brightness,color,contrast,angle,solarize_level,ifsolar,ifgray,ifinver]
    if transformation_set == 3:
        brightness = random.uniform(0.2, 1.8)
        color = random.uniform(0.2, 1.8)
        contrast = random.uniform(0.2, 1.8)
        solarize_level = random.randint(75, 255)
        angle = random.uniform(-60, 60)
        noise_level = random.uniform(0.0, 30.0)
        ifsolar = random.random()
        ifgray = random.random()
        ifinver = random.random()
        ifGauss = random.random()
        ifblur = random.random()
        return [brightness,color,contrast,angle,noise_level,solarize_level,ifsolar,ifgray,ifinver,ifGauss,ifblur]


def apply_transformation(image, transformation_set, sample):
    """
    应用指定的变换集到图像上
    """
    if transformation_set == 1:
        # 变换集1: 亮度、颜色、对比度、Solarize、Grayscale、Invert
        brightness = sample[0]
        color = sample[1]
        contrast = sample[2]
        solarize_level = sample[3]
        
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Color(image).enhance(color)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        if sample[4] > 0.5:  # 随机选择是否应用Solarize
            image = ImageOps.solarize(image, solarize_level)
        if sample[5] > 0.5:  # 随机选择是否转换为灰度
            image = image.convert('L')
            image = image.convert('RGB')
        if sample[6] > 0.5:  # 随机选择是否应用反色
            image = ImageOps.invert(image)
    
    elif transformation_set == 2:
        """
        应用变换集2到图像上，包括亮度、颜色、对比度、旋转、Solarize、Grayscale和Invert
        """
        brightness = sample[0]
        color = sample[1]
        contrast = sample[2]
        angle = sample[3]
        solarize_level = sample[4]
        
        # 应用亮度、颜色和对比度增强
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Color(image).enhance(color)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        
        # 随机应用Solarize
        if sample[5] > 0.5:
            image = ImageOps.solarize(image, solarize_level)
        
        # 随机转换为灰度
        if sample[6] > 0.5:
            image = image.convert('L')
            image = image.convert('RGB')
        # 随机应用反色
        if sample[7] > 0.5:
            image = ImageOps.invert(image)
        
        # 应用旋转
        image = image.rotate(angle)
    
    elif transformation_set == 3:
        """
        应用变换集3到图像上，包括亮度、颜色、对比度、Solarize、Grayscale、Invert、旋转、高斯噪声和模糊
        """
        brightness = sample[0]
        color = sample[1]
        contrast = sample[2]
        solarize_level = sample[3]
        angle = sample[4]
        noise_level = sample[5]
        
        # 应用亮度、颜色和对比度增强
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Color(image).enhance(color)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        
        # 随机应用Solarize
        if sample[6] > 0.5:
            image = ImageOps.solarize(image, solarize_level)
        
        # 随机转换为灰度
        if sample[7] > 0.5:
            image = image.convert('L')
            image = image.convert('RGB')
        # 随机应用反色
        if sample[8] > 0.5:
            image = ImageOps.invert(image)
        
        # 应用旋转
        image = image.rotate(angle)
        
        # 应用高斯噪声
        if sample[9] > 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=noise_level))
        
        # 应用模糊
        if sample[10] > 0.5:
            image = image.filter(ImageFilter.BLUR)
    
    return image

def batch_apply_transformation(batch_tensor_images, transformation_set_index, device):
    """
    对一个data_loader得到的图像tensor batch，应用变换集
    """
    transformed_images = []

    mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).to(device)

    sampled_transform_args = sample_transform(transformation_set_index)

    for i in range(batch_tensor_images.size(0)):
        # print(batch_tensor_images[i].shape)
        # 获取单张图像tensor
        img_tensor = batch_tensor_images[i].unsqueeze(0)  # 增加一个批次维度，以便进行广播操作

        # 逆归一化
        img_unnormalized = img_tensor * std.view(3, 1, 1) + mean.view(3, 1, 1)

        # 确保像素值在0到255之间，并转换为整数
        img_unnormalized = img_unnormalized.clamp(0, 1).mul(255).byte()

        # 将Tensor转换为PIL图像
        pil_image = Image.fromarray(img_unnormalized.squeeze().permute(1, 2, 0).cpu().numpy())

        # pil_image = to_pil(batch_tensor_images[i])
        # print(len(pil_image.split()))
        # 转为PIL
        # 应用随机转换
        transformed_image = apply_transformation(pil_image, transformation_set_index, sampled_transform_args)

        # print(len(transformed_image.split()))

        # 将PIL图像转换为Tensor
        transform_to_tensor = transforms.ToTensor()

        # 归一化Tensor
        transform_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # 将图像转换为Tensor
        tensor_image = transform_to_tensor(transformed_image)

        # 归一化Tensor
        normalized_tensor_image = transform_normalize(tensor_image)

        transformed_images.append(normalized_tensor_image)


    # 将列表转换回批次
    return torch.stack([img for img in transformed_images]).to(device)


if __name__ == '__main__':

    # 加载MNIST数据集，不应用任何转换
    train_dataset = datasets.MNIST(root=r'./data/MNIST_data', train=True, download=True, transform=transforms.ToTensor())

    # 选择一个图像进行变换
    image_index = 0
    image_tensor, label = train_dataset[image_index]

    # 将Tensor转换为PIL Image以便应用变换
    pil_image = transforms.ToPILImage()(image_tensor)

    # 应用变换集1
    transformed_pil_image = apply_transformation(pil_image, 1)

    # 指定保存图像的路径
    save_path = './transformed_images'  # 创建一个目录来保存图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存原始图像和变换后的图像
    original_image_path = os.path.join(save_path, f'original_image_{image_index}.png')
    transformed_image_path = os.path.join(save_path, f'transformed_image_{image_index}.png')

    pil_image.save(original_image_path)
    transformed_pil_image.save(transformed_image_path)

    print(f'Original image saved to: {original_image_path}')
    print(f'Transformed image saved to: {transformed_image_path}')
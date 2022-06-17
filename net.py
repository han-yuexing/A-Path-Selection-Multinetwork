import torch
import torchvision
import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
import resnet

def set_param_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False  # 不更新提取的参数梯度


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
    # use_pretrained refers to use already trained param or not
    if model_name is "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)
        # 从ImageNet返回预训练的模型和参数
        set_param_requires_grad(model_ft, feature_extract)  # 提取的参数梯度不更新
        num_ftrs = model_ft.fc.in_features  # 最后的全连接层
        model_ft.fc = nn.Linear(num_ftrs, num_classes)  # 输出两个class
        input_size = 224
    elif model_name is "resnext101":
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_param_requires_grad(model_ft, feature_extract)  # 提取的参数梯度不更新
        num_ftrs = model_ft.fc.in_features  # 最后的全连接层
        model_ft.fc = nn.Linear(num_ftrs, num_classes)  # 输出两个class
        input_size = 224
    elif model_name is "resnext50":
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_param_requires_grad(model_ft, feature_extract)  # 提取的参数梯度不更新
        num_ftrs = model_ft.fc.in_features  # 最后的全连接层
        model_ft.fc = nn.Linear(num_ftrs, num_classes)  # 输出两个class
        input_size = 224
    elif model_name is "Attresnext50":
        model_ft = resnet.resnext50_32x4d(pretrained=use_pretrained)
        set_param_requires_grad(model_ft, feature_extract)  # 提取的参数梯度不更新
        num_ftrs = model_ft.fc.in_features  # 最后的全连接层
        model_ft.fc = nn.Linear(num_ftrs, num_classes)  # 输出两个class
        input_size = 224

    return model_ft, input_size
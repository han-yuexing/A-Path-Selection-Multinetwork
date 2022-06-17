import torch
import torchvision
import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
import copy
import resnet
from config import get_arguments
import functions
from net import initialize_model

parser = get_arguments()
parser.add_argument('--PATH', default='models', help='where to save checkpoints')
parser.add_argument('--data_dir', type=str, help='path of Sep dataset', required=True)
parser.add_argument('--scenarios', default='lower', help='upper, lower, eye')
parser.add_argument('--num_classes', default=2, help='number of classes, for BeginNet is 3, for Subnets is 2')

opt = parser.parse_args()

torch.backends.cudnn.benchmark = False

if opt.no_cuda:
    device = torch.device("cpu")
else:
    device = torch.device(opt.cudaDevice)

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 数据加强
        transforms.Resize(256),  # 将图片Resize成ResNet所需大小
        transforms.RandomCrop(opt.input_size),
        transforms.ToTensor(),
    ]),
    "val": transforms.Compose([
        transforms.Resize(opt.input_size),  # 将图片Resize成ResNet所需大小
        # transforms.CenterCrop(input_size),  # 在中心截取图片
        transforms.ToTensor(),
    ])
}
print("Initializing Datasets and Dataloaders...")

image_datasets = []
dataloader_dict = []
class_names = []
# 分为三个dataloader分别训练
for i in range(3):
    # print(i)
    image_datasets.insert(i, {
        x: datasets.ImageFolder(os.path.join(os.path.join(opt.data_dir, x), str(i)), data_transforms[x]) for x in
        ['train', 'val']})
    # print(i)
    dataloader_dict.insert(i, {x: torch.utils.data.DataLoader(image_datasets[i][x],
                                                              batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                                                              pin_memory=True) for x in ['train', 'val']})
    class_names.insert(i, image_datasets[i]['train'].classes)  # Labels


def train_model(model, dataloaders, criterion, optimizer, num_epochs=opt.num_epochs):
    since = now = time.time()
    val_acc_history = []
    val_loss_his = []
    best_model_wts = copy.deepcopy(model.state_dict())  # 保存准确率最高的参数
    best_acc = 0.
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        now = time.time()
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        for phase in ["train", "val"]:
            running_loss = 0.
            running_corrects = 0.
            if phase is "train":
                model.train()
            else:
                model.eval()

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.autograd.set_grad_enabled(phase == "train"):  # 梯度管理器
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                #                     print(outputs)
                _, preds = torch.max(outputs, 1)
                # 返回每一行最大的数和索引， preds位置是索引的位置
                if phase is "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1))  # view(-1)展开至一维并计算

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print("{} Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))
            time_in_one_epoch = time.time() - now

            if phase is "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase is "val":
                val_acc_history.append(epoch_acc)  # 记录每一个验证集的准确率
                val_loss_his.append(epoch_loss)
        print("Epoch {} complete in {}m {}s".format(epoch, time_in_one_epoch // 60, time_in_one_epoch % 60))
        print()

    time_elapsed = time.time() - since
    print("Training complete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {}".format(best_acc))

    model.load_state_dict(best_model_wts)  # 把最新的参数复制到model中
    return model, val_acc_history, val_loss_his

##################### SubnetX ##################################
SubnetX, inputSize = initialize_model(opt.model_name, opt.num_classes+1,
                                               feature_extract=opt.featureExtract, use_pretrained=False)
SubnetX = SubnetX.to(device)
optimizer = optim.SGD(SubnetX.parameters(), lr=opt.lrX, momentum=opt.momentum)
criterion = nn.CrossEntropyLoss()
SubnetXModel, valAccHist, lossHist = train_model(SubnetX, dataloader_dict[0],
                                               criterion, optimizer,
                                               num_epochs=opt.num_epochs)

trainTime = time.time()
scenarios = opt.scenarios
ModelDirName = opt.model_name + '_' + str(trainTime)
ModelName = 'SubnetX.pth'

saveDirPath = os.path.join(opt.PATH, 'Subnets', scenarios, ModelDirName, 'SubnetX')
if not os.path.exists(saveDirPath):
    os.makedirs(saveDirPath)
savePath = os.path.join(saveDirPath, ModelName)
torch.save(SubnetXModel, savePath)

functions.savetxt(saveDirPath, 'SubnetX', valAccHist, lossHist)
functions.savePlot(opt, valAccHist, lossHist, saveDirPath, netName='SubnetX', epochs=opt.num_epochs)


##################### SubnetY ##################################

# SubnetY, inputSize = initialize_model(opt.model_name, opt.num_classes,
#                                                feature_extract=opt.featureExtract, use_pretrained=False)
# SubnetY = SubnetY.to(device)
# optimizer = optim.SGD(SubnetY.parameters(), lr=opt.lrY, momentum=opt.momentum)
# criterion = nn.CrossEntropyLoss()
# SubnetYModel, valAccHist, lossHist = train_model(SubnetY, dataloader_dict[1],
#                                                criterion, optimizer,
#                                                num_epochs=opt.num_epochs + 200)

# ModelName = 'SubnetY.pth'

# saveDirPath = os.path.join(opt.PATH, 'Subnets', scenarios, 'SubnetY', ModelDirName)
# if not os.path.exists(saveDirPath):
#     os.makedirs(saveDirPath)
# savePath = os.path.join(saveDirPath, ModelName)
# torch.save(SubnetYModel, savePath)

# functions.savetxt(saveDirPath, 'SubnetY', valAccHist, lossHist)
# functions.savePlot(opt, valAccHist, lossHist, saveDirPath, netName='SubnetY')


##################### SubnetZ ##################################

SubnetZ, inputSize = initialize_model(opt.model_name, opt.num_classes,
                                               feature_extract=opt.featureExtract, use_pretrained=False)
SubnetZ = SubnetZ.to(device)
optimizer = optim.SGD(SubnetZ.parameters(), lr=opt.lrZ, momentum=opt.momentum)
criterion = nn.CrossEntropyLoss()
SubnetZModel, valAccHist, lossHist = train_model(SubnetZ, dataloader_dict[2],
                                               criterion, optimizer,
                                               num_epochs=opt.num_epochs)

ModelName = 'SubnetZ.pth'

saveDirPath = os.path.join(opt.PATH, 'Subnets', scenarios, ModelDirName, 'SubnetZ')
if not os.path.exists(saveDirPath):
    os.makedirs(saveDirPath)
savePath = os.path.join(saveDirPath, ModelName)
torch.save(SubnetZModel, savePath)

functions.savetxt(saveDirPath, 'SubnetZ', valAccHist, lossHist)
functions.savePlot(opt, valAccHist, lossHist, saveDirPath, netName='SubnetZ', epochs=opt.num_epochs)


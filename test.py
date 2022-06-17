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
from config import get_arguments

softmax = nn.Softmax(dim=1)

torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = get_arguments()
# parser.add_argument('--PATH', default='models', help='where to save checkpoints')
parser.add_argument('--savePath', default='results', help='path to save the test results')
parser.add_argument('--data_dir', type=str, help='dataset for test', required=True)
parser.add_argument('--SubnetsPath', type=str, help='path to Subnets', required=True)
#saveDirPath = os.path.join(opt.PATH, 'Subnets', scenarios, ModelDirName)
parser.add_argument('--beginNetPath', type=str, help='path to BeginNet', required=True)
parser.add_argument('--newPathSelection', action='store_true', default=False,)
# saveDirPath = os.path.join(opt.PATH, 'BeginNet', scenarios, ModelDirName)

# parser.add_argument('--input_size', type=int, default=224)


opt = parser.parse_args()
preddict = []
labeldict = []
model_path = []
subnets = ['SubnetX', 'SubnetY', 'SubnetZ']
for subnet in subnets:
    pthName = subnet + '.pth'
    model_path.append(os.path.join(opt.SubnetsPath, subnet, pthName))

BeginModel_path = os.path.join(opt.beginNetPath, 'BeginNet.pth')

cm = [[0] * 7 for i in range(7)]

batch_size = 1


print(device)
model_0 = torch.load(model_path[0])
model_1 = torch.load(model_path[1])
model_2 = torch.load(model_path[2])
beginModel = torch.load(BeginModel_path)
beginModel = beginModel.to(device)
model_0 = model_0.to(device)
model_1 = model_1.to(device)
model_2 = model_2.to(device)

data_transforms = {

    'test': transforms.Compose([
        transforms.Resize(opt.input_size),  # 将图片Resize成ResNet所需大小
        transforms.ToTensor(),
    ])
}
print("Initializing Datasets and Dataloaders...")

image_datasets = dict({'test': datasets.ImageFolder(os.path.join(opt.data_dir, "test"), data_transforms['test'])})
dataloader_dict = dict({'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size,
                                                            shuffle=True, num_workers=opt.num_workers, pin_memory=True)})
class_names = image_datasets['test'].classes  # Labels
print(class_names)
running_corrects = 0.
Begin_corrects = 0.
for input, label in dataloader_dict['test']:
    input = input.to(device)
    label = label.to(device)

    # 仅在ConcateDB中需要！！！ 因为在读入表情标签时会出现ipynb_checkpoints
    #         label-=1

    with torch.no_grad():
        Beginout = softmax(beginModel(input))
        # outputs = torch.cat((out(model_0,input), out(model_1,input), out(model_2,input)), 1)
        _, Beginpreds = torch.max(Beginout, 1)
        x = torch.zeros(1, 3).to(device)
        y = torch.zeros(1, 2).to(device)
        # print(model_0(input).size())
        if opt.newPathSelection:
            if Beginpreds.item() == 0:
                if label.view(-1) == 0 or label.view(-1) == 1 or label.view(-1) == 2:
                    Begin_corrects += 1
#                 outputsX = model_0(input)
            elif Beginpreds.item() == 1:
                if label.view(-1) == 3 or label.view(-1) == 4:
                    Begin_corrects += 1
#                 outputs = torch.cat((x, model_1(input), y), 1)
            else:
                if label.view(-1) == 5 or label.view(-1) == 6:
                    Begin_corrects += 1
#                 outputs = torch.cat((x, y, model_2(input)), 1)
            outputsX = softmax(model_0(input)) * Beginout[0][0]
            outputsY = softmax(model_1(input)) * Beginout[0][1]
            outputsZ = softmax(model_2(input)) * Beginout[0][2]
            outputs = torch.cat((outputsX, outputsY, outputsZ), 1)
        else:
            if Beginpreds.item() == 0:
                if label.view(-1) == 0 or label.view(-1) == 1 or label.view(-1) == 2:
                    Begin_corrects += 1
                outputs = softmax(model_0(input))
            elif Beginpreds.item() == 1:
                if label.view(-1) == 3 or label.view(-1) == 4:
                    Begin_corrects += 1
                outputs = torch.cat((x, softmax(model_1(input)), y), 1)
            else:
                if label.view(-1) == 5 or label.view(-1) == 6:
                    Begin_corrects += 1
                outputs = torch.cat((x, y, softmax(model_2(input))), 1)
        # print(outputs)
        _, preds = torch.max(outputs, 1)
        # print(preds.view(-1))
        # print(label.view(-1))
        preddict.append(preds.view(-1))
        labeldict.append(label.view(-1))
        running_corrects += torch.sum(preds.view(-1) == label.view(-1))
        # print(running_corrects)
acc = running_corrects / len(dataloader_dict['test'].dataset)
Beginacc = Begin_corrects / len(dataloader_dict['test'].dataset)
for i in range(len(preddict)):
    print(preddict[i], labeldict[i])
    cm[labeldict[i].view(-1)][preddict[i].view(-1)] += 1;

print("Test Acc : {}".format(acc))
print("BeginAcc: {}".format(Beginacc))
print(cm)

if not os.path.exists(opt.savePath):
    os.makedirs(opt.savePath)

name = opt.data_dir + str(time.time()) + '.txt'
file = open(os.path.join(opt.savePath, name), "w", encoding='utf-8')
file.write(opt.SubnetsPath)
file.write('\n')
file.write(opt.beginNetPath)
file.write('\n')
file.write("Test Acc: \n")
file.write(str(acc))
file.write('\n')
file.write("BeginAcc: \n")
file.write(str(Beginacc))
file.write('\n')
file.write("Confusion Matrix: \n")
file.write(str(cm))
file.write('\n')
file.close()





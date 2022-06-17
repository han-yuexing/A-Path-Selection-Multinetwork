import torch
import torchvision as tv
import matplotlib.pyplot as plt

import os
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
import torch.nn as nn
softmax = nn.Softmax(dim=1)
from config import get_arguments

torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = get_arguments()
parser.add_argument('--img_path', type=str, default="./predictedPic/eyeMasked", help='folder contains unknown images')
# parser.add_argument('--PATH', default='models', help='where to save checkpoints')
parser.add_argument('--savePath', default='results', help='path to save the test results')
# parser.add_argument('--data_dir', type=str, help='dataset for test', required=True)
parser.add_argument('--SubnetsPath', type=str, help='path to Subnets', required=True)
#saveDirPath = os.path.join(opt.PATH, 'Subnets', scenarios, ModelDirName)
parser.add_argument('--beginNetPath', type=str, help='path to BeginNet', required=True)
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

transforms = transforms.Compose([
                        transforms.Resize(224), #将图片Resize成ResNet所需大小
                        transforms.CenterCrop(224),
                        transforms.Grayscale(3),
                        transforms.ToTensor(),])
images_path = []
files = os.listdir(opt.img_path)
for file in files:
    if file.lower().endswith('jpg') or file.endswith('png'):
        images_path.append(os.path.join(opt.img_path, file))

x = torch.zeros(1, 3).to(device)
y = torch.zeros(1, 2).to(device)
BeginClasses = [0, 1, 2]
classes = ['Happiness', 'Normal', 'Sadness', 'Anger', 'Fear', 'Disgust', 'Surprise']

def imshow(img, title='None', img_name='None'):
    img = img.cpu()
    #print(img.shape)
    npimg = img.numpy()
    #print(npimg.shape)
    plt.imshow(np.reshape(npimg,(224,224,3)))
    title = 'Pred:' + title + '\nLabel:' + img_name[:-4]
    plt.title(title)
    plt.show()
    str = img_name[:-4] + 'Predicted.png'
#     plt.savefig(str)

def predictImg(model0, model1, model2, BeginModel, images):
    Bmodel = BeginModel.to(device)
    with torch.no_grad():
        Bmodel.eval()
        for img_name in images:
            img = Image.open(img_name)
            img = transforms(img).unsqueeze(0)
            #print(img.shape)
            img = img.to(device)
            Boutputs = Bmodel(img)
            Boutputs.to(device)
            _, preds = torch.max(Boutputs, 1)
            print(softmax(Boutputs))
            print(preds)
            if BeginClasses[preds] == 0:
                model = model0.to(device)
                outputs = softmax(model(img))
            elif BeginClasses[preds] == 1:
                model = model1.to(device)
                outputs = torch.cat((x, softmax(model(img)), y), 1)
            else:
                model = model2.to(device)
                outputs = torch.cat((x, y, softmax(model(img))), 1)
            print(outputs)
            _, predicted = torch.max(outputs, 1)
            print(predicted)
            #plt.ion()
            plt.figure()
            imshow(img.squeeze(0), classes[predicted], img_name)


model0 = torch.load(model_path[0])
model1 = torch.load(model_path[1])
model2 = torch.load(model_path[2])
BeginModel = torch.load(BeginModel_path)
predictImg(model0, model1, model2, BeginModel, images_path)
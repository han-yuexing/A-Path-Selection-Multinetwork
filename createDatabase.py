import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib
import cv2
import random
import shutil
import time

files = {
    'fer2013': './fer2013.csv',
    'rafdb': './rafdbface.csv',
    'jaffe': './jaffeface.csv',
    'kdef': './kdefface.csv',
}

emotionsRafdb = {
    '6': '3anger',  # 生气
    '3': '5disgust',  # 厌恶
    '2': '4fear',  # 恐惧
    '4': '0happy',  # 开心
    '5': '2sad',  # 伤心
    '1': '6surprised',  # 惊讶
    '7': '1normal',  # 中性
}

emotionsOther = {
    '0': '3anger',  # 生气
    '1': '5disgust',  # 厌恶
    '2': '4fear',  # 恐惧
    '3': '0happy',  # 开心
    '4': '2sad',  # 伤心
    '5': '6surprised',  # 惊讶
    '6': '1normal',  # 中性
}

datasets = ['fer2013', 'rafdb', 'jaffe', 'kdef']
# scenarios = ['ConcatDB', 'BConcatDB', 'eyeCDB']
scenarios = ['eyeCDB']

def createDir(dir):
    if os.path.exists(dir) is False:
        os.makedirs(dir)


def processImg(data, begin=0, end=0, dirName='', scene='', dataset=''):
    for index in range(begin, end):
        # 解析每一行csv文件内容
        emotion_data = data.loc[index][0]
        image_data = data.loc[index][1]
        # 将图片数据转换成48*48
        data_array = list(map(float, image_data.split()))
        data_array = np.asarray(data_array)
        iimage = data_array.reshape(48, 48)
        if scene == 'ConcatDB':
            iimage[0:24, 0:48] = 0  # 遮挡上半部分
        elif scene == 'BConcatDB':
            iimage[24:48,0:48] = 0 # 遮挡下半部分
        elif scene == 'eyeCDB':
            classfier = cv2.CascadeClassifier("./haarcascade_eye_tree_eyeglasses.xml")
            iimage = Image.fromarray(iimage).convert('RGB')
            iimage = cv2.cvtColor(np.asarray(iimage), cv2.COLOR_RGB2BGR)
            iimage = cv2.resize(iimage, (224, 224))
            Rects = classfier.detectMultiScale(iimage, scaleFactor=1.1, minNeighbors=10, minSize=(1, 1))
            # print(Rects)
            if len(Rects) > 0:
                for Rect in Rects:
                    x, y, w, h = Rect
                    iimage[y + 10:y + h - 10, x:x + w, 0] = np.zeros((h - 20, w))
                    iimage[y + 10:y + h - 10, x:x + w, 1] = np.zeros((h - 20, w))
                    iimage[y + 10:y + h - 10, x:x + w, 2] = np.zeros((h - 20, w))
        emotionName = emotions[str(emotion_data)]
        imagePath = os.path.join(scene, dirName, emotionName)
        createDir(scene)
        createDir(imagePath)
        if dataset == 'rafdb':
            fix = '_rafdb'
        if dataset == 'jaffe':
            fix = '_jaffe'
        if dataset == 'kdef':
            fix = '_kdef'
        if dataset == 'fer2013':
            fix = '_fer2013'
        imageName = os.path.join(imagePath, str(index) + fix + '.png')
        Image.fromarray(iimage).convert('RGB').save(imageName)


def copyFile(fileDir, class_name, train_rate, save_dir):
    image_list = os.listdir(fileDir)  # 获取图片的原始路径
    image_number = len(image_list)

    train_number = int(image_number * train_rate)
    train_sample = random.sample(image_list, train_number)  # 从image_list中随机获取0.8比例的图像.
    test_sample = list(set(image_list) - set(train_sample))
    sample = [train_sample, test_sample]

    # 复制图像到目标文件夹
    for k in range(len(save_dir)):
        if os.path.isdir(save_dir[k] + class_name):
            for name in sample[k]:
                shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k] + class_name + '/', name))
        else:
            os.makedirs(save_dir[k] + class_name)
            for name in sample[k]:
                shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k] + class_name + '/', name))


# for dataset in datasets:
#     for scene in scenarios:
#         if dataset == 'rafdb':   #rafdb的表情标签映射关系与其他不同
#             emotions = emotionsRafdb
#         else:
#             emotions = emotionsOther
#         print('processing ', dataset, ', ', scene)
#         file = files[dataset]
#         data = pd.read_csv(file)
#         print(len(data))
#         processImg(data, begin=0, end=len(data), dirName='all', scene=scene, dataset=dataset)
#         print(dataset, ', ', scene, ', ', 'Done!')

print('Split to train, validation and test set')
for scene in scenarios:

    time_start = time.time()
    # 原始数据集路径
    origion_path = str('./' + scene + '/all/')
    # 保存路径
    save_train_dir = str('./' + scene + '/train/')
    save_test_dir = str('./' + scene + '/testval/')
    save_dir = [save_train_dir, save_test_dir]
    # 训练集比例
    train_rate = 0.8
    # 数据集类别及数量
    file_list = os.listdir(origion_path)
    num_classes = len(file_list)

    for i in range(num_classes):
        class_name = file_list[i]
        image_Dir = os.path.join(origion_path, class_name)
        copyFile(image_Dir, class_name, train_rate, save_dir)
        print('%s划分完毕！' % class_name)

    time_end = time.time()
    print('---------------')
    print('训练集划分共耗时%s!' % (time_end - time_start))

    time_start = time.time()
    # 原始数据集路径
    origion_path = str('./' + scene + '/testval/')
    # 保存路径
    save_train_dir = str('./' + scene + '/val/')
    save_test_dir = str('./' + scene + '/test/')
    save_dir = [save_train_dir, save_test_dir]
    # 训练集比例
    train_rate = 0.5
    # 数据集类别及数量
    file_list = os.listdir(origion_path)
    num_classes = len(file_list)

    for i in range(num_classes):
        class_name = file_list[i]
        image_Dir = os.path.join(origion_path, class_name)
        copyFile(image_Dir, class_name, train_rate, save_dir)
        print('%s划分完毕！' % class_name)

    time_end = time.time()
    print('---------------')
    print('测试集和验证集划分共耗时%s!' % (time_end - time_start))

print('All done!')



import os
import numpy as np
import cv2
import csv

base_dir = './Rafdb/Image/aligned/aligned'
Datas = np.zeros([15339, 48 * 48], dtype=np.uint8)
Labels = []
Usage = []
index = 0


with open(r"./Rafdb/EmoLabel/list_patition_label.txt", "r") as labelfile:
    labels = labelfile.read()
    labels = labels.split('\n')
    labels.pop()
    for label in labels:
        print(label.split(' ')[1])
        expr = label.split(' ')[1]
        Labels.append(expr)

for i, j, k in os.walk(base_dir):
    print(i)
    print(j)
    print(k)
    for img in k:
        # print(img)
        name = img.split('.')[0]
        usage = name.split('_')[0]
        Usage.append(usage)
        img = os.path.join(i, img)
        # print(img)
        img = cv2.imread(img, 1)
        # print(img)
        dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.resize(dst, (48, 48))
        Datas[index][0:48 * 48] = np.ndarray.flatten(dst)
        index += 1




with open(r"./rafdbface.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['emotion', 'pixels', 'usage'])
    for i in range(len(Datas)):
        data_list = list(Datas[i])
        b = " ".join(str(x) for x in data_list)
        # print(b)
        l = np.hstack([Labels[i], b, Usage[i]])
        writer.writerow(l)
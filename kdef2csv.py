import os
import numpy as np
import cv2
import csv

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
base_dir = './KDEF_and_AKDEF/KDEF/'
datas = np.zeros([980, 48 * 48], dtype=np.uint8)
label = np.zeros([980], dtype=int)
# datas = []
# label = []
# np.array(datas)
# np.array(label)
index = 0
for i, j, k in os.walk(base_dir):
    print(i)

    for img in k:
        # print(img)
        name = img.split('.')[0]
        No = name[:4]
        expression = name[4:6]
        angle = name[6:]
        if angle != 'S':
            continue
        img = os.path.join(i, img)
        # print(img)
        img = cv2.imread(img, 1)
        # print(img)
        dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detect(dst, cascade)

        # print(angle)

        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img,(x1+10,y1+20),(x2-10,y2),(0,255,255),2)
            # 调整截取脸部区域大小
            img_roi = np.uint8([y2-(y1+20), (x2-10)-(x1+10)])
            roi = dst[y1+20:y2, x1+10:x2-10]
            img_roi = roi
            re_roi = cv2.resize(img_roi, (48,48))
            # if expression == 'AN':
            #     np.append(label, 0)
            # elif expression == 'DI':
            #     np.append(label, 1)
            # elif expression == 'FE':
            #     np.append(label, 2)
            # elif expression == 'HA':
            #     np.append(label, 3)
            # elif expression == 'SA':
            #     np.append(label, 4)
            # elif expression == 'SU':
            #     np.append(label, 5)
            # elif expression == 'NE':
            #     np.append(label, 6)
            #
            # np.append(datas, np.ndarray.flatten(re_roi))


            if expression == 'AN':
                label[index] = 0
            elif expression == 'DI':
                label[index] = 1
            elif expression == 'AF':
                label[index] = 2
            elif expression == 'HA':
                label[index] = 3
            elif expression == 'SA':
                label[index] = 4
            elif expression == 'SU':
                label[index] = 5
            elif expression == 'NE':
                label[index] = 6

            datas[index][0:48 * 48] = np.ndarray.flatten(re_roi)
            index += 1


        # data =  No + '/' + img + ',' + str(swtich_expression[expression]) + ',' + str(switch_angle[angle])
        # datas.append(1)

# print(len(datas))
# def save(fileName, contents):
#     fh = open(fileName, 'w', encoding='utf-8')
#     for item in contents:
#         fh.write(str(item)+"\n")
#     fh.close()
#
# save('data/all.txt', datas)
print(index)
with open(r"./kdefface.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['emotion', 'pixels'])
    for i in range(len(label)):
        data_list = list(datas[i])
        b = " ".join(str(x) for x in data_list)
        l = np.hstack([label[i], b])
        writer.writerow(l)
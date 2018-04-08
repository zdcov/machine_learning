import cv2
from imutils import contours
import imutils
from imutils.perspective import four_point_transform
import numpy as np
import os
import math
data_dir='F:\\0_9'

def find_lcd(image):
    image_copy=image.copy()
    gray=cv2.cvtColor(image_copy,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    (ret, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    print('OSTU计算得到的thresh：',ret)
    (_, thresh) = cv2.threshold(blurred, ret - 22, 255, cv2.THRESH_BINARY_INV)
    _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    peri = cv2.arcLength(cnts[0], True)
    approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)
    warped = four_point_transform(image, approx.reshape(4, 2))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return thresh,warped,warped_gray

def process_lcd(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(blurred, ret + 23, 255, cv2.THRESH_BINARY)
    thresh=imutils.resize(thresh,400)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,11))
    thresh_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel1)
    return thresh_close

def split_number(image):
    all_rows=[]
    w, h = (image.shape[1], image.shape[0])
    for i in range(3):
        every_row=image[int((i)*h/3):int((i+1)*h/3),0:int(0.84*w)]
        all_rows.append(every_row)
    all_rows.append(image[0:h, math.floor(0.84 * w):w])
    return all_rows

def get_number(image):
    _, cnts, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_num=[]
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if h>48 and w>8:
            valid_num.append(c)
    valid_num=contours.sort_contours(valid_num,method='left-to-right')[0]
    return valid_num

def knn_classifier(test):
    directories = []
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        if os.path.isdir(path):
            directories.append(path)
    i = 0
    train_data = []
    train_label = []
    for directory in directories:
        for filename in os.listdir(directory)[0:53]:
            path = os.path.join(directory, filename)
            x = cv2.imread(path, 0)
            x = cv2.resize(x, (60, 108))
            x=x/255.
            x = x.reshape(1, 6480)
            train_data.append(x[0])
            train_label.append(i)
        i = i + 1
    samples = np.array(train_data, np.float32)
    labels = np.array(train_label, np.float32)
    labels = labels.reshape((labels.size, 1))
    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, labels)
    retval, results, neigh_resp, dists = model.findNearest(test, 7)
    return results.ravel()

src=cv2.imread('watch_e.jpg')
thresh,warped,warped_gray=find_lcd(src)
thresh_2=process_lcd(warped_gray)
input1=imutils.resize(warped,400)
all_rows=split_number(thresh_2)
all_rows1=split_number(input1)


test_data=[]

for i in range(3):
    numbers=get_number(all_rows[i])
    for number in numbers:
        (x, y, w, h) = cv2.boundingRect(number)
        num=all_rows[i]
        x=num[y:y+h,x:x+w]
        x=cv2.resize(x,(60,108))
        x=x/255.
        x=x.reshape(1,6480)
        test_data.append(x[0])

test_samples=np.array(test_data,np.float32)

s=knn_classifier(test_samples)
digits=s.tolist()
print(digits)
_, cnts, _ = cv2.findContours(all_rows[2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
i=0
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if h>48 and w>8:
        cv2.rectangle(all_rows1[2],(x,y),(x+w,y+h),(0,255,0),1)
        print(h,w)

cv2.imshow('thresh',thresh)
cv2.imshow('wrapped',warped)
cv2.imwrite('thresh.jpg',thresh)
cv2.imwrite('wrapped.jpg',warped)
cv2.waitKey(0)
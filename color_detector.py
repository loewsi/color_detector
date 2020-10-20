#!/usr/bin/env python3
import cv2
import numpy as np
from time import sleep
import math
import os


cap = cv2.VideoCapture(2)


n_splits = int(os.environ['N_SPLITS'])
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 10
top_colors = np.zeros((n_splits, 3))


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #Put here your code!
    # You can now treat output as a normal numpy array
    # Do your magic here
    Z = frame.reshape((-1, 3))
    Z_float = np.float32(Z)
    ret, label, center = cv2.kmeans(Z_float, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center_int = np.uint8(center)
    res = center_int[label.flatten()]
    split_length = np.size(label)/n_splits

    for i in range(0, n_splits):
        begin = int(math.floor(i*split_length))
        end = int(math.floor((i+1)*split_length))
        label_split = label[begin:end]
        unique, counts = np.unique(label_split, return_counts=True)
        ind = np.argmax(counts)
        top_color = center_int[ind, :]
        top_colors[i, :] = top_color

    print(top_colors)

    sleep(1)

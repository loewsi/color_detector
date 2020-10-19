#!/usr/bin/env python3
import cv2
import numpy as np
from time import sleep
import math
import os


cap = cv2.VideoCapture(2)


n_splits = int(os.environ['N_SPLITS'])
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
top_colors = np.zeros((n_splits, 3))


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #Put here your code!
    # You can now treat output as a normal numpy array
    # Do your magic here
    Z = frame.reshape((-1, 3))
    Z_length = np.size(Z)
    split_length = Z_length/n_splits
    Z_float = np.float32(Z)
    print()
    for i in range(0, (n_splits-1)):
        begin = int(math.floor(i*split_length))
        print(begin)
        end = int(math.floor((i+1)*split_length))
        print(end)
        Z_split = Z_float[begin:end]
        print(Z_split)
        ret, label, center = cv2.kmeans(Z_split, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        unique, counts = np.unique(label, return_counts=True)
        ind = np.argmax(counts)
        top_color = center[ind, :]
        top_colors[i, :] = top_color
        print(i)

    print(top_colors)

    sleep(3)

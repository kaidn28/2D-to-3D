import cv2
import numpy as np
from prototypes import *

class LikelihoodMapper:
    def __init__(self, kernel_size = 11):
        self.kernel_size = 11
        self.prototype1 = Prototype1(kernel_size)
        self.prototype2 = Prototype2(kernel_size)

    def pad(self, img):
        sub_size = self.kernel_size // 2
        padded_img = np.zeros((img.shape[0]+self.kernel_size-1, img.shape[1]+self.kernel_size -1))
        print(img.shape)
        print(padded_img.shape)
        padded_img[sub_size: -sub_size, sub_size: -sub_size] = img
        return padded_img

    def map(self, img, n = 10, t = 0.8):
        padded_img = self.pad(img)
        self.conved_img = img.copy()
        h_img, w_img = img.shape
        print(h_img, w_img)
        counter = 0
        for u in range(h_img):
            for v in range(w_img):
                fA, fB, fC, fD = self.prototype1._conv(padded_img[u:u+self.kernel_size, v: v+self.kernel_size])
                m = 0.25*(fA + fB + fC + fD)
                s11 = min(min(fA, fB) - m, m - min(fC, fD))
                s12 = min(m - min(fA, fB), min(fC, fD) - m)
                fA, fB, fC, fD = self.prototype2._conv(padded_img[u:u+self.kernel_size, v:v+self.kernel_size])
                m = 0.25*(fA+fB + fC+ fD)
                s21 = min(min(fA, fB) - m, m - min(fC, fD))
                s22 = min(m - min(fA, fB), min(fC, fD) - m)
                c = max(s11, s12, s21, s22)
                self.conved_img[u,v] = c
                counter +=1
                print(counter)
        print(self.conved_img)
        self.conved_img = self.conved_img*255/self.conved_img.max()
        cv2.imshow('abc', self.conved_img)
        cv2.waitKey()
        return self.conved_img

l = LikelihoodMapper(kernel_size=11)
path = '../chessboard.jpg'
img = cv2.imread(path, 0)
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
conved_img = l.map(img)
print(conved_img)

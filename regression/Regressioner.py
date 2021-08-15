import pandas as pd
import numpy as np
import os 
import cv2
import time
class Regression:
    def __init__(self, args):
        self.data = pd.read_csv(args.data)
        self.data.info()
        drawingBoard = np.zeros((700, 700)) + 255
        drawingBoard = cv2.merge([drawingBoard, drawingBoard, drawingBoard])
        
        data_draw = self.data.applymap(lambda x: int(x*10 + 300)).to_numpy()
        print(data_draw)
        for x1, y1, x2, y2 in data_draw:
            cv2.line(drawingBoard, (x1,y1), (x2,y2), (0, 255, 0), 2)
            cv2.circle(drawingBoard, (x1,y1), 2, (0,0,255), -1)
            cv2.circle(drawingBoard, (x2,y2), 2, (255,0,0), -1)
        # cv2.imshow('abc', drawingBoard)
        
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # cv2.imwrite('visualize.jpg', drawingBoard)
        print(self.data)
        self.drawingBoard = drawingBoard
        #print(data_proc)
    def grad(self, k, q, X, Y):
        dJ = 0
        for i, x in enumerate(X):
            y = Y[i]
            #print(k-x)
            #print('dJ: ', dJ)
            dJ += (k-x)*4*(np.linalg.norm(k-x)/np.linalg.norm(x-y) -q)/np.linalg.norm(x-y)
        dJ = dJ/len(X)
        #print('dJ: ', dJ)
        return dJ
    def train(self, lr = 0.1, max_iteration = 1000, print_after= 5):
        data_proc = self.data.to_numpy()
        #print(data_proc)
        X_train = data_proc[:, :2]
        Y_train = data_proc[:, 2:]
        k = (np.random.rand(2) - 0.5)*30
        loss = 0
        for iter in range(max_iteration):
            qs = []
            rs = []
            loss = 0
            for i, x in enumerate(X_train):
                y = Y_train[i]
                qs.append(np.linalg.norm((k-x))/np.linalg.norm((x-y)))
            q =  np.mean(qs) 
            for i, x in enumerate(X_train):
                y = X_train[i]
                qi = qs[i]
                loss += 2*(q - qi)**2/len(X_train)
            g = self.grad(k, q, X_train, Y_train)
            if np.linalg.norm(g) < 1e-6:
                break
            k -= lr*self.grad(k, q, X_train, Y_train)
            #print(k)
            print(loss)
        #print(k)
        #print(q)
        np.save(open('./regression/regression_data.npy', 'wb'),[k,q], dtype="object")

    def getRegressionData(self, data = './regression/regression_data.npy'):
        k, q = np.load(open(data, 'rb'))
        self.k = k
        self.q = q
        print(k, q)

    def predict(self, x, y):
        reg_x = x + (self.k[0]-x)/self.q
        reg_y = y + (self.k[1]-x)/self.q
        return reg_x, reg_y




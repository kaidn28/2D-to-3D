import pandas as pd
import numpy as np
import os 
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
class Regression:
    def __init__(self, args):
        self.data = pd.read_csv(args.data)
        self.data.info()
        drawingBoard = np.zeros((700, 700)) + 255
        drawingBoard = cv2.merge([drawingBoard, drawingBoard, drawingBoard])
        
        data_draw = self.data.applymap(lambda x: int(x*10 + 300)).to_numpy()
        # print(data_draw)
        for x1, y1, x2, y2 in data_draw:
            cv2.line(drawingBoard, (x1,y1), (x2,y2), (0, 255, 0), 2)
            cv2.circle(drawingBoard, (x1,y1), 2, (0,0,255), -1)
            cv2.circle(drawingBoard, (x2,y2), 2, (255,0,0), -1)
        #cv2.imshow('abc', drawingBoard)
        
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # cv2.imwrite('visualize.jpg', drawingBoard)
        # print(self.data)
        self.drawingBoard = drawingBoard
        #print(data_proc)
    
    def foldSplit(self, data, k):
        #print(data.shape[0])
        fold_length = data.shape[0]//k
        folds = []
        for i in range(k):
            ith_fold = data[i*fold_length: (i+1)*fold_length]
            #print(ith_fold.shape)
            folds.append(ith_fold)
        #print(folds)
        return folds
    ##linear no bias
    def ithFoldTrainTestSplit(self, folds, i):
        folds_ = folds.copy()
        #print(len(folds_))
        val = folds_.pop(i)

        x_val = val[:, :2]
        y_val = val[:, 2:]
        #print(len(folds_))
        train = np.concatenate(folds_)
        x_train = train[:, :2]
        y_train = train[:, 2:]
        #print(train.shape)
        return x_train, x_val, y_train, y_val
    def grad_no_bias(self, k, q, X, Y):
        dJ = 0
        for i, x in enumerate(X):
            y = Y[i]
            #print(k-x)
            #print('dJ: ', dJ)
            dJ += (k-x)*4*(np.linalg.norm(k-x)/np.linalg.norm(x-y) -q)/(np.linalg.norm(x-y)*np.linalg.norm(k-x))
        dJ = dJ/len(X)
        #print('dJ: ', dJ)
        return dJ
    def calculateParams_nobias(self, k, X, Y):
        q_sum = 0
        for i, x in enumerate(X):
            y = Y[i]
            q_sum += np.linalg.norm((k-x))/np.linalg.norm((x-y))
        q =  q_sum/len(X)
        return q 
    def calculateLoss_nobias(self, k, q, X, Y):
        loss = 0
        for i, x in enumerate(X):
            y = Y[i]
            qi = np.linalg.norm((k-x))/np.linalg.norm((x-y))
            loss += 2*(q - qi)**2/len(X)
        return loss
    ##linear with bias
    def grad_bias(self, k, q, m, X, Y):
        dJ = np.zeros(2)
        for i, x in enumerate(X):
            y = Y[i]
            dJ += 2*((np.linalg.norm(k-x)+m)/np.linalg.norm(x-y)-q)*2*(k-x)/(np.linalg.norm(x-y)*np.linalg.norm(k-x))
        dJ = dJ/len(X)
        #print(dJ)
        return dJ
    def calculateParams_bias(self, k, X, Y):
        a = len(X)
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        for i, x in enumerate(X):
            y = Y[i]
            b +=(-1/np.linalg.norm(x - y))
            c +=np.linalg.norm(k -x)/np.linalg.norm(x -y)
            d = -b
            e += (-1/(np.linalg.norm(x-y)**2))
            f +=np.linalg.norm(k-x)/(np.linalg.norm(x-y)**2)
        matA = np.array([a,b,d,e]).reshape(2,2)
        matb = np.array([c,f])
        matAinv = np.linalg.inv(matA)
        #print(np.dot(matAinv, matb))
        return np.dot(matAinv, matb) 
    
    def calculateLoss_bias(self, k, q, m, X, Y):
        loss = 0
        for i, x in enumerate(X):
            y = Y[i]
            loss += (q-((np.linalg.norm(k-x)+m)/np.linalg.norm(x-y)))**2
        return loss
    
    ## exponential

    # def grad(self, k, q, m, X, Y):
    #     dJ = np.zeros(2)
    #     for i, x in enumerate(X):
    #         y = Y[i]
    #         dJ += -4*(k-x)*(q-(np.log(np.linalg.norm(k-x))+m)/np.linalg.norm(x-y))/(np.linalg.norm(k-x)*np.linalg.norm(x-y))
    #     dJ = dJ/len(X)
    #     #print(dJ)
    #     return dJ
    # def calculateParams(self, k, X, Y):
    #     a = len(X)
    #     b = 0
    #     c = 0
    #     d = 0
    #     e = 0
    #     f = 0
    #     for i, x in enumerate(X):
    #         y = Y[i]
    #         b +=(-1/np.linalg.norm(x - y))
    #         c +=np.log(np.linalg.norm(k -x))/np.linalg.norm(x -y)
    #         d = -b
    #         e += (-1/(np.linalg.norm(x-y)**2))
    #         f +=(np.log(np.linalg.norm(k-x))/(np.linalg.norm(x-y)**2))
    #     matA = np.array([a,b,d,e]).reshape(2,2)
    #     matb = np.array([c,f])
    #     matAinv = np.linalg.inv(matA)
    #     return np.dot(matAinv, matb) 
    # def calculateLoss(self, k, q, m, X, Y):
    #     loss = 0
    #     for i, x in enumerate(X):
    #         y = Y[i]
    #         loss += (q-(np.log(np.linalg.norm(k-x))+m)/(np.linalg.norm(x-y)))**2
    #     return loss
    def train(self, lr = 0.1, max_iteration = 1000, print_after= 5):
        #print(1)
        #print(self.data.to_numpy())
        data_proc = self.data.sample(frac=1).to_numpy()
        X_proc = data_proc[:, :2]
        Y_proc = data_proc[:, 2:]
        #print(data_proc.shape)
        folds = self.foldSplit(data_proc, 4)
        for j in range(4):
            print("_________Fold {}: __________".format(j+1))
            X_train, X_val, Y_train, Y_val = self.ithFoldTrainTestSplit(folds, j)
            "interpolation error: "
            # print(X_val[j])
            # print(Y_val[j])
            #print(X_val - Y_val)
            err_xs = [np.abs(p[0]) for p in X_val - Y_val]
            err_ys = [np.abs(p[1]) for p in X_val - Y_val]
            errs = [np.sqrt(p[0]**2+p[1]**2) for p in X_val-Y_val]
            # for e in range(len(X_val)):
                # print("sample {}: ".format(e))
                # print("err x: ")
                # print(err_xs[e])
                # print("err y: ")
                # print(err_ys[e])
                # print("err euclid: ")
                # print(errs[e])
                # print([np.abs(p[1]) for p in X_val - Y_val])
                # print([np.sqrt(p[0]**2+p[1]**2) for p in X_val-Y_val])
            mean_err_x = np.mean([np.abs(p[0]) for p in X_val - Y_val])
            mean_err_y = np.mean([np.abs(p[1]) for p in X_val - Y_val])
            mean_err = np.mean([np.sqrt(p[0]**2+p[1]**2) for p in X_val-Y_val])
            # print("mean x")
            # print(mean_err_x)
            # print("mean y")
            # print(mean_err_y)
            # print("mean euclid")
            # print(mean_err)
            #print(Y_train)
            #print(X_val - Y_val)
            k = (np.random.rand(2) - 0.5)*30
            loss = 0
            # for iter in range(max_iteration):
            #     qs = []
            #     rs = []
            #     loss = 0
            #     for i, x in enumerate(X_train):
            #         y = Y_train[i]
            #         qs.append(np.linalg.norm((k-x))/np.linalg.norm((x-y)))
            #     q =  np.mean(qs) 
            #     for i, x in enumerate(X_train):
            #         y = X_train[i]
            #         qi = qs[i]
            #         loss += 2*(q - qi)**2/len(X_train)
            #     g = self.grad(k, q, X_train, Y_train)
            #     if np.linalg.norm(g) < 1e-6:
            #         break
            #     k -= lr*self.grad(k, q, X_train, Y_train)
            #     #print(k)
            #     #print(loss)
            # print('k: ',k)
            # print('q: ', q)
            # print('loss: ', loss)
            # errs = []
            # for i, x in enumerate(X_val):
            #     y = Y_val[i]
            #     reg_y = x + (k-x)/q
            #     print('regression result: ', reg_y)
            #     print('ground truth: ', y)
            #     err = np.linalg.norm(reg_y - y)
            #     errs.append(err)
            #     #print('err: ', reg_y - y)
            # print(np.mean(errs))
            # cv2.circle(self.drawingBoard, (int(k[0]*10 +300), int(k[1]*10 + 300)), 5, (0,0, 255), -1)
            # #cv2.imshow('abc', self.drawingBoard)
            # #cv2.waitKey()
            # #cv2.destroyAllWindows()
            # cv2.imwrite('visualization2.jpg', self.drawingBoard)
            # dis = []
            # err = []
            # for i,X in enumerate(X_proc):        
            #     Y = Y_proc[i]
            #     dis.append(np.linalg.norm(k - Y))
            #     err.append(np.linalg.norm(X - Y))

            #     #print(X)
            # #print(dis, err)
            # plt.scatter(dis, err)
            # plt.show()
            # np.save(open('./regression/regression_data.npy', 'wb'),np.array([k,q], dtype='object'))
            q = 0
            for iter in range(max_iteration):
                q = self.calculateParams_nobias(k, X_train, Y_train)
                g = self.grad_no_bias(k, q, X_train, Y_train)
                if np.linalg.norm(g) < 1e-6:
                    break
                k -= lr*g
                loss = self.calculateLoss_nobias(k, q, X_train, Y_train)
                #print("__loss-no-bias__: ", loss)
            print('no bias')
            print("k: ", k)
            print("q: ", q)
            # cv2.circle(self.drawingBoard, (int(k[0]*10 +300), int(k[1]*10 + 300)), 5, (0,0, 255), -1)
            # cv2.imshow('abc', self.drawingBoard)
            # cv2.waitKey()
            # cv2.destroyAllWindows()  
            err_xs = []
            err_ys = []
            errs = []
            for i, x in enumerate(X_val):
                y = Y_val[i]
                reg_y = x + (k-x)/q
                #print('regression result: ', reg_y)
                #print('ground truth: ', y)
                err_x = np.abs((reg_y - y)[0])
                err_y = np.abs((reg_y - y)[1]) 
                err = np.linalg.norm(reg_y - y)
                # print("sample {}: ".format(i))
                # print("err x: ")
                # print(err_x)
                # print("err x: ")
                # print(err_y)
                # print("err euclid: ")
                # print(err)
                errs.append(err)
                err_xs.append(err_x)
                err_ys.append(err_y)
                #print('err: ', reg_y - y)
            # print("err x")
            # print(err_xs)
            #print("mean x")
            #print(np.mean(err_xs))
            #print("mean y")
            #print(np.mean(err_ys))
            # print("err y")
            # print(err_ys)
            # #print("mean euclid")
            # #print(np.mean(errs))
            # print("errs: ")
            # print(errs)
            dis = []
            err = []
            for i,X in enumerate(X_proc):        
                Y = Y_proc[i]
                dis.append(np.linalg.norm(k - Y)*10)
                err.append(np.linalg.norm(X - Y)*10)
                #print(X)
            #print(dis)
            #print(err)
            plot_data = [[dis[i], e] for i, e in enumerate(err)]
            plot_df = pd.DataFrame(plot_data, columns=['dis', 'err'], dtype = 'float')
            
            plot_df.loc[20, 'dis'] = q
            print(plot_df)
            plot_df.to_csv("fold{}.csv".format(j+1), index=False)
            # plt.scatter(dis, err)
            # A = [0,0]
            # B = [q, 1]
            # print(q)
            # # plt.axline(A, B, color ='red', gid="y=alpha*x")
            # # plt.xlabel("distance to convergence point (mm)")
            # # plt.ylabel("error (mm)")
            # # plt.show() 
            
            k = (np.random.rand(2) - 0.5)*30
            loss = 0
            q, m = 0,0
            for iter in range(max_iteration):
                q, m = self.calculateParams_bias(k, X_train, Y_train)
                g = self.grad_bias(k, q, m, X_train, Y_train)
                if np.linalg.norm(g) < 1e-6:
                    break
                k -= lr*g
                loss = self.calculateLoss_bias(k, q, m, X_train, Y_train)
                #print('__loss__: ', loss)
            # print('bias')
            # print(k)
            # print(q, m)
            # cv2.circle(self.drawingBoard, (int(k[0]*10 +300), int(k[1]*10 + 300)), 5, (0,0, 255), -1)
            # cv2.imshow('abc', self.drawingBoard)
            # cv2.waitKey()
            # cv2.destroyAllWindows()  
            dis = []
            err = []
            for i,X in enumerate(X_proc):        
                Y = Y_proc[i]
                dis.append(np.linalg.norm(k - Y))
                err.append(np.linalg.norm(X - Y))
            
            #print(dis, err)
            # plt.scatter(dis, err)
            # A = [m,0]
            # B = [q+m, 1]
            # print(q)
            # plt.axline(A, B, color ='red')
            # plt.xlabel("distance to convergence point (mm)")
            # plt.ylabel("error (mm)")
            # plt.show() 
            # cv2.imwrite('visualization_linear_bias.jpg', self.drawingBoard)
            errs = []
            for i, x in enumerate(X_val):
                y = Y_val[i]
                diskx = np.linalg.norm(k-x)
                disxy = (diskx+m)/q
                #print(disxy)
                y_pred = x + (k-x)*disxy/diskx
                err = np.linalg.norm(y-y_pred)
                #print(y, y_pred)
                errs.append(err)
            # print(np.mean(errs))
            #print(loss)
    def getRegressionData(self, data = './regression/regression_data.npy'):
        k, q = np.load(open(data, 'rb'))
        self.k = k
        self.q = q
        print(k, q)
        
    def predict(self, x, y):
        reg_x = x + (self.k[0]-x)/self.q
        reg_y = y + (self.k[1]-y)/self.q
        return reg_x, reg_y

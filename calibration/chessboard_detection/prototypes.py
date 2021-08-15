import numpy as np
import scipy
import cv2
def gkern(l=11, sig=1.):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel

class Prototype:
    def __init__ (self, kernel_size = 11):
        assert kernel_size % 2 == 1
        zkernel = np.zeros((kernel_size, kernel_size))
        self.kernel_size = kernel_size
        self.sub_size = kernel_size // 2
        self.kernelA = zkernel.copy()
        self.kernelB = zkernel.copy()
        self.kernelC = zkernel.copy()
        self.kernelD = zkernel.copy()
        
    def _conv(self, img):
        assert img.shape == (self.kernel_size, self.kernel_size)
        outputA = np.multiply(img, self.kernelA).sum()
        outputB = np.multiply(img, self.kernelB).sum()
        outputC = np.multiply(img, self.kernelC).sum()
        outputD = np.multiply(img, self.kernelD).sum()
        return outputA, outputB, outputC, outputD
    
class Prototype1(Prototype): 
    def __init__(self, kernel_size = 11):
        super().__init__(kernel_size=kernel_size)
        #print(kernelA)
        gkernel = gkern(l = kernel_size)
        self.kernelA[:self.sub_size,self.sub_size+1:] = gkernel[:self.sub_size, self.sub_size+1:]
        self.kernelA = self.kernelA/self.kernelA.sum()
        self.kernelB[self.sub_size+1:, :self.sub_size] = gkernel[self.sub_size+1:, :self.sub_size]
        self.kernelB = self.kernelB/self.kernelB.sum()
        self.kernelC[: self.sub_size, : self.sub_size] = gkernel[:self.sub_size, :self.sub_size]
        self.kernelC = self.kernelC/self.kernelC.sum()
        self.kernelD[self.sub_size+1:, self.sub_size+1:] = gkernel[self.sub_size+1:, self.sub_size+1:]
        self.kernelD = self.kernelD/self.kernelD.sum()

class Prototype2(Prototype):
    def __init__(self, kernel_size = 11):
        super().__init__(kernel_size=kernel_size)
        gkernel = gkern(l = kernel_size)
        for u in range(kernel_size):
            for v in range(kernel_size):
                if u < v and u+v < kernel_size -1:
                    self.kernelA[u,v] = gkernel[u,v]
                elif u > v and u+v > kernel_size -1:
                    self.kernelB[u,v] = gkernel[u,v]
                elif u < v and u+v > kernel_size -1: 
                    self.kernelC[u,v] = gkernel[u,v]
                elif u> v and u+ v <kernel_size -1:
                    self.kernelD[u,v] = gkernel[u,v]
        self.kernelA = self.kernelA/self.kernelA.sum()
        self.kernelB = self.kernelB/self.kernelB.sum()
        self.kernelC = self.kernelC/self.kernelC.sum()
        self.kernelD = self.kernelD/self.kernelC.sum()


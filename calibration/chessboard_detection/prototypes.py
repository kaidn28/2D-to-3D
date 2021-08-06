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

class Prototype1: 
    def __init__(self, kernel_size = 11):
        assert kernel_size % 2 == 1
        gkernel = gkern(l = kernel_size)
        zkernel = np.zeros((kernel_size, kernel_size))
        kernelA = zkernel.copy()
        kernelB = zkernel.copy()
        kernelC = zkernel.copy()
        kernelD = zkernel.copy()
        sub_size = kernel_size // 2
        #print(kernelA)
        kernelA[:sub_size,sub_size+1:] = gkernel[:sub_size, sub_size+1:]
        self.kernelA = kernelA/kernelA.sum()
        kernelB[sub_size+1:, :sub_size] = gkernel[sub_size+1:, :sub_size]
        self.kernelB = kernelB/kernelB.sum()
        kernelC[:sub_size, : sub_size] = gkernel[:sub_size, :sub_size]
        self.kernelC = kernelC/kernelC.sum()
        kernelD[sub_size+1:, sub_size+1:] = gkernel[sub_size+1:, sub_size+1:]
        self.kernelD = kernelD/kernelD.sum()

class Prototype2:
    def __init__(self, kernel_size = 11):
        assert kernel_size %2 == 1
        gkernel = gkern(l = kernel_size)
        zkernel = np.zeros((kernel_size, kernel_size))
        kernelA = zkernel.copy()
        kernelB = zkernel.copy()
        kernelC = zkernel.copy()
        kernelD = zkernel.copy()
        for u in range(kernel_size):
            for v in range(kernel_size):
                if u < v and u+v < kernel_size -1:
                    kernelA[u,v] = gkernel[u,v]
                elif u > v and u+v > kernel_size -1:
                    kernelB[u,v] = gkernel[u,v]
                elif u < v and u+v > kernel_size -1: 
                    kernelC[u,v] = gkernel[u,v]
                elif u> v and u+ v <kernel_size -1:
                    kernelD[u,v] = gkernel[u,v]
        self.kernelA = kernelA/kernelA.sum()
        self.kernelB = kernelB/kernelB.sum()
        self.kernelC = kernelC/kernelC.sum()
        self.kernelD = kernelD/kernelC.sum()


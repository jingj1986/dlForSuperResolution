#!/usr/bin/python
##
# Author: Guojun.jin <>
# Date:   2016-10-20
# Desc:   SSIM of tow images as the Quota
#
# Note, from ssim_quota import *   as the register when use this quota

from quota import *
import math
import numpy as np
import cv2
from skimage import color,io
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast

class SsimQuota(Quota):
    def __init__(self, end_type, conf_time=0, conf_range = 1.0):
        Quota.__init__(self,end_type,conf_time,conf_range)
        self.name = "SsimQuota"

    def block_view(self, A, block=(3, 3)):
        """Provide a 2D block view to 2D array. No error checking made.
        Therefore meaningful (as implemented) only for blocks strictly
        compatible with the shape of A."""
        # simple shape and strides computations may seem at first strange
        # unless one is able to recognize the 'tuple additions' involved ;-)
        shape = (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
        strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
        return ast(A, shape= shape, strides= strides)


    def _calculate_1(self, img1, img2, *tupleArg):
        C1=0.01**2
        C2=0.03**2
        img1 = color.rgb2ycbcr(img1)
        img2 = color.rgb2ycbcr(img2)
        bimg1 = self.block_view(img1[:,:,0].astype(np.float), (4,4))
        bimg2 = self.block_view(img2[:,:,0].astype(np.float), (4,4))
        s1  = np.sum(bimg1, (-1, -2))
        s2  = np.sum(bimg2, (-1, -2))
        ss  = np.sum(bimg1*bimg1, (-1, -2)) + np.sum(bimg2*bimg2, (-1, -2))
        s12 = np.sum(bimg1*bimg2, (-1, -2))

        vari = ss - s1*s1 - s2*s2
        covar = s12 - s1*s2

        ssim_map =  (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
        return np.mean(ssim_map) 

    def _calculate(self, img1, img2, *tupleArg):
        sd=1.5
        C1=0.01**2
        C2=0.03**2
        img1 = color.rgb2ycbcr(img1)
        img2 = color.rgb2ycbcr(img2)

        mu1 = gaussian_filter(img1.astype(np.float), sd)
        mu2 = gaussian_filter(img2.astype(np.float), sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
        sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
        sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

        ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_map = ssim_num / ssim_den
        return np.mean(ssim_map)



## Test
if __name__ == "__main__" :
    img1 = io.imread("/home/jgj/pic/baby_GT.bmp")
    img2 = io.imread("/home/jgj/pic/result_baby.bmp")
    img3 = io.imread("/home/jgj/pic/baby_SM.bmp")
    img3 = cv2.resize(img3, (512,512))

    obj = SsimQuota('END_BAD') 
    ssim = obj.get_quota(img1, img2)
    print ssim
    ssim = obj.get_quota(img1, img3)
    needEnd = obj.need_end()
    print ssim
    print needEnd

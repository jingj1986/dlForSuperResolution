#!/usr/bin/python
##
# Author: Guojun.jin <>
# Date:   2016-10-13
# Desc:   pnsr of tow images as the Quota
#
# Note, from psnr_quota import *   as the register when use this quota

from quota import *
import math
import numpy as np
import cv2
from skimage import color,io

class PsnrQuota(Quota):
    def __init__(self, end_type, conf_time=0, conf_range = 1.0):
        Quota.__init__(self,end_type,conf_time,conf_range)
        self.name = "PsnrQuota"

    def _calculate(self, img1, img2, *tupleArg):
        if len(tupleArg) > 1:
            border = tupleArg[0]
        else:
            border = 2

        if img1.shape[2] == 3:
            img1 = color.rgb2ycbcr(img1)
            img1 = img1[:,:,0].astype(np.float)
        if  img2.shape[2] == 3:
            img2 = color.rgb2ycbcr(img2)
            img2 = img2[:,:,0].astype(np.float)

        diff =  (img1 - img2)
        if border != 0:
            diff = diff[border:diff.shape[0]-border, border:diff.shape[1]-border]
        mse = np.mean( diff ** 2 )
        if mse == 0:
            return 100
        return 20 * math.log10(255.0/ math.sqrt(mse))


## Test
if __name__ == "__main__" :
    img1 = io.imread("/home/jgj/pic/baby_GT.bmp")
    img2 = io.imread("/home/jgj/pic/result_baby.bmp")
    img3 = io.imread("/home/jgj/pic/baby_SM.bmp")
    img3 = cv2.resize(img3, (512,512))

    obj = PsnrQuota('END_BAD') 
    #pnsr = obj.get_quota(img1, img2, 2)
    pnsr = obj.get_quota(img1, img2)
    print pnsr
    pnsr = obj.get_quota(img1, img3)
    needEnd = obj.need_end()
    print pnsr
    print needEnd

#!/usr/bin/python
# Author; Guojun Jin <>
# Date:   2016-10-13
# Desc:   utils 

import types
import numpy as np
from psnr_quota import *
from ssim_quota import *
from skimage import color


def str_to_class(s):
    if s in globals() and isinstance(globals()[s], types.ClassType):
        return globals()[s]
    return None


def colorize(y0, y1, y2):
    img = np.zeros((y0.shape[0], y0.shape[1], 3), np.float)
    img[:,:,0] = y0.astype(np.float)*255.0
    img[:,:,1] = y1
    img[:,:,2] = y2
    img = color.ycbcr2rgb(img)

    ha, wa, ca = np.shape(img)
    for c in range(ca):
        for w in range(wa):
            for h in range(ha):
                if img[h,w,c] > 1.0:
                    img[h,w,c] = 1.0
                elif img[h,w,c] < -1.0:
                    img[h,w,c] = -1.0

    return img


def colorize_ycbcr(y0, y1, y2):
    img = np.zeros((y0.shape[0], y0.shape[1], 3), np.float)
    img[:,:,0] = y0
    img[:,:,1] = y1
    img[:,:,2] = y2
    img = color.ycbcr2rgb(img.astype(np.float)*255.0)

    ha, wa, ca = np.shape(img)
    for c in range(ca):
        for w in range(wa):
            for h in range(ha):
                if img[h,w,c] > 1.0:
                    img[h,w,c] = 1.0
                elif img[h,w,c] < -1.0:
                    img[h,w,c] = -1.0

    return img


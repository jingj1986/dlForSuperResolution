import os
import numpy as np
import cv2
import math
import time

import sys
import caffe
import matplotlib.pyplot as plt
from skimage import transform,data,io,util,color,img_as_uint,exposure

#from evaluation import *

def colorize(y0, y1, y2):
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
 
def psnr(img1, img2, border):
    if img1.shape[2] == 3:
        img1 = color.rgb2ycbcr(img1)
        img1 = img1[:,:,0].astype(np.float)
    if img2.shape[2] == 3:
        img2 = color.rgb2ycbcr(img2)
        img2 = img2[:,:,0].astype(np.float)
    diff =  (img1 - img2)
    if border != 0:
        diff = diff[border:diff.shape[0]-border, border:diff.shape[1]-border]
    mse = np.mean( diff ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
 


zooming = 2
caffe_root = '/home/user/caffe/examples/VDSR/'
caffe.set_device(1)
caffe.set_mode_gpu()

net = caffe.Net(caffe_root + 'VDSR_deploy.prototxt',
        caffe_root + '../VDSR_filter/Model/_iter_30000.caffemodel',caffe.TEST)

input_dir = caffe_root + 'data/'
gt_dir    = caffe_root + 'label/'
train_dir    = caffe_root + 'Train/'
project_dir = caffe_root + 'result/'
im_high = io.imread('./pic/test1.png')
im_pad = io.imread('./pic/test1.png');

[h,w] = im_high.shape[:2]
h_ycbcr = color.rgb2ycbcr(im_pad).astype(np.float)/255.0

start_time = int(time.time()) 

net.blobs['data'].reshape(1, 1, h, w)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
out = net.forward_all(data=np.asarray([transformer.preprocess('data', h_ycbcr[:,:,0])]))
mat0 = out['sum'][0]
end_time = int(time.time())

print ("during %s") % (end_time - start_time)

m_pred = colorize(mat0[0,:,:], h_ycbcr[:,:,1], h_ycbcr[:,:,2])
#im_pad = color.ycbcr2rgb(im_pad.astype(float)*255.0)

#plt.imsave(caffe_root + 'data/result_caffe.png', m_pred)
#ha, wa, ca = np.shape(im_pad)
#for c in range(ca):
#    for w in range(wa):
#        for h in range(ha): 
#            if im_pad[h,w,c] > 1.0:
#                im_pad[h,w,c] = 1.0
#            elif im_pad[h,w,c] < -1.0:
#                im_pad[h,w,c] = -1.0

io.imsave('./result/result_caffe.bmp', m_pred)

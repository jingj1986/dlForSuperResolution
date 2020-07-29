#!/usr/bin/python
#DESC:  The format of dataset for caffe is H5.
#	This script is used to generate h5 dataset.
#       The input images are YCBCR format, and select 41x41 part with step 14.
#       Target image is the HR one, the input is LR one.
#DATE:  2016-11-06
#AUTHOR:Guojun Jin <jingj1986@163.com>

import os
import math
import h5py
import caffe
import random
import numpy as np

import cv2
from skimage import io, color, transform
from matplotlib import pyplot as plt 


def gen_hdf5(label_path, input_path, outfile_name):
    count = 0
    scale = 2
    #stride = 14
    stride = 36
    size_input = 41
    size_label = 41
    data = np.zeros((5000000,1, size_input,size_input), np.float)
    label = np.zeros((5000000,1, size_label,size_label), np.float)

    ### load info
    for f in os.popen('ls ' + label_path + "/*"):
        label_name = f.strip()
        img_name=label_name.split('/')[-1]
        input_name = input_path + "/" + img_name
#        img_input = io.imread(input_name)
        try:
            img_org = io.imread(label_name)
            img_input = io.imread(input_name.strip())
            img_input = img_input[:,:,:3]
            #img_input = io.imread(label_name)
        except :
            print "err read:" + input_name
            continue 
        sizeinfo = img_org.shape
        if len(sizeinfo) != 3 or len(img_input.shape) != 3:
            print "err size:" + img_name
            continue
        [h,w] = img_org.shape[:2]
        img_label = color.rgb2ycbcr(img_org).astype(np.float)/255.0
        img_input = cv2.resize(img_input, (w,h))
        img_input = color.rgb2ycbcr(img_input).astype(np.float)/255.0
        im_label = img_label[:,:,0]
        im_input = img_input[:,:,0]
 
        #for x in range(1, h - size_input - 1, stride):
        #    for y in range(1, w - size_input - 1, stride):
        for x in range(size_input+1, h - size_input - size_input -1 , stride):
            for y in range(size_input+1, w - size_input - size_input -1, stride):
                submit_input = im_input[x:x+size_input, y:y+size_input]
                submit_label = im_label[x:x+size_input, y:y+size_input]
                data[count,0,:,:]= submit_input
                label[count,0,:,:] = submit_label
                count = count + 1
    print ("count %s" % count)
    print "END LOOP IMAGE"
    order = random.sample(range(count),count)

    data = data[order,:,:,:]
    label = label[order,:,:,:]
   
    ## write
    chunksz = 64
    setname, ext = outfile_name.split('.')
    h5_filename = '{}.h5'.format(setname)
    last_read = 0
    with h5py.File(h5_filename, 'w') as h:
        h.create_dataset('data', data=data, chunks=(64,1,size_input,size_input))
        h.create_dataset('label', data=label, chunks=(64,1,size_label, size_label))

if __name__ == "__main__":
    gen_hdf5('/home/user/data/dajuezhan/org', '/home/user/data/dajuezhan/test_1', 'train_2.h5')
#    gen_hdf5('/home/user/data/org/org0', '/home/user/data/blur/blur1/org0', 'test.h5')
#    gen_hdf5('/home/user/data/org/org0', '/home/user/data/blur/blur1/org0', 'train_org0.h5')
#    gen_hdf5('/home/user/data/org/org1', '/home/user/data/blur/blur1/org1', 'train_org1.h5')
#    gen_hdf5('/home/user/data/org/org2', '/home/user/data/blur/blur1/org2', 'train_org2.h5')
#    gen_hdf5('/home/user/data/org/org3', '/home/user/data/blur/blur1/org3', 'train_org3.h5')
#    gen_hdf5('/home/user/data/org/org4', '/home/user/data/blur/blur1/org4', 'train_org4.h5')
#    gen_hdf5('/home/user/data/org/org5', '/home/user/data/blur/blur1/org5', 'train_org5.h5')
#    gen_hdf5('/home/user/data/org/org6', '/home/user/data/blur/blur1/org6', 'train_org6.h5')
#    gen_hdf5('/home/user/data/org/org7', '/home/user/data/blur/blur1/org7', 'train_org7.h5')
#    gen_hdf5('/home/user/data/org/org8', '/home/user/data/blur/blur1/org8', 'train_org8.h5')
#    gen_hdf5('/home/user/data/org/org9', '/home/user/data/blur/blur1/org9', 'train_org9.h5')

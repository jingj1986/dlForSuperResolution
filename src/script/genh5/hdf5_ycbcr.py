#!/usr/bin/python
#DESC:  The format of dataset for caffe is H5.
#	This script is used to generate h5 dataset.
#       The input images are YCBCR format, and select 41x41 part with step 14.
#       Target image is the original one, the input is down-sample one.
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



def gen_hdf5(file_path, outfile_name):
    count = 0
    scale = 2
    #stride = 14
    stride = 40
    size_input = 41
    size_label = 41
    data = np.zeros((500000,1, size_input,size_input), np.float)
    label = np.zeros((500000,1, size_label,size_label), np.float)

    ### load info
    print file_path
    for f in os.popen('ls ' + file_path + "/*"):
        image_name = f.strip()
        try:
            img_org = io.imread(image_name)
        except:
            print "err channel " + image_name
            continue 
        #for rotate_val in [0, 90, 180,270]:
            #img_org = transform.rotate(img_org, rotate_val)
            #channel = img_org.shape[2]
        sizeinfo = img_org.shape
        if len(sizeinfo) != 3 or sizeinfo[2] != 3:
            print "err resize " + image_name
            continue
        [h,w] = img_org.shape[:2]
        img_label = color.rgb2ycbcr(img_org).astype(np.float)/255.0
    	#img_label = img_org.astype(np.float)/255.0
        img = cv2.resize(img_label, (w/scale, h/scale))
        #img_input = cv2.resize(img,(w,h), interpolation=cv2.INTER_CUBIC)
        img_input = cv2.resize(img,(w,h))
        im_label = img_label[:,:,0]
        im_input = img_input[:,:,0]
 
#        for x in range(1, h - size_input - 1, stride):
#            for y in range(1, w - size_input - 1, stride):
        for x in range(41, h - size_input - 42, stride):
            for y in range(41, w - size_input - 42, stride):

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
#    with open('{}_h5.txt'.format(setname), 'w') as f:
#        f.write(h5_filename)

if __name__ == "__main__":
    gen_hdf5('/home/user/data/dileizhan/org0', 'train_org0.h5') 
#    gen_hdf5('./org8', 'train_org8.h5') 
#    gen_hdf5('./org9', 'train_org9.h5') 
#    gen_hdf5('./Set5', 'test-py.h5') 
#    for i in range(10,71):
#        img_dir = "../org" + str(i)
#        h5_name = "/data/train_lib/vdsr/train_org" + str(i) + ".h5"
#        gen_hdf5(img_dir, h5_name)

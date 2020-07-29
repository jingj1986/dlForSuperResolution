#!/usr/bin/python
#DESC:  The format of dataset for caffe is H5.
#	This script is used to generate h5 dataset.
#       The input images are RGB format, and select 41x41 part with step 14.
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
    stride = 41
    size_input = 41
    size_label = 41
    data = np.zeros((5000000,3, size_input,size_input), np.float)
    label = np.zeros((5000000,3, size_label,size_label), np.float)

    ### load info
    print file_path
    for f in os.popen('ls ' + file_path + "/*"):
        image_name = f.strip()
        print image_name
        img_org = io.imread(image_name)
        #for rotate_val in [0, 90, 180,270]:
            #img_org = transform.rotate(img_org, rotate_val)
            #channel = img_org.shape[2]
	sizeinfo = img_org.shape
	if len(sizeinfo) != 3 or sizeinfo[2] != 3:
	    continue
        [h,w] = img_org.shape[:2]
        #img_label = color.rgb2ycbcr(img_org).astype(np.float)/255.0
	img_label = img_org.astype(np.float)/255.0
        img = cv2.resize(img_label, (w/scale, h/scale))
        #img_input = cv2.resize(img,(w,h), interpolation=cv2.INTER_CUBIC)
        img_input = cv2.resize(img,(w,h))
	#img = transform.resize(img_label,(h/2,w/2))
        #img_input = transform.resize(img,(h,w))
	#print img_label[:,0,0]
	#print img_input[:,0,0]
        im_label = img_label.reshape(3,w,h)
        im_input = img_input.reshape(3,w,h)
        
        for x in range(1, w - size_input - 1, stride):
            for y in range(1, h - size_input - 1, stride):
                submit_input = im_input[:,x:x+size_input, y:y+size_input]
                submit_label = im_label[:,x:x+size_input, y:y+size_input]
                data[count,:,:,:]= submit_input
                label[count,:,:,:] = submit_label
                count = count + 1
    print count
    print "END LOOP IMAGE"
    order = random.sample(range(count),count)
    data = data[order,:,:,:]
    label = label[order,:,:,:]
    
    ## write
    chunksz = 64
    setname, ext = outfile_name.split('.')
    h5_filename = '{}.h5'.format(setname)
    with h5py.File(h5_filename, 'w') as h:
        h.create_dataset('data', data=data, chunks=(64,3,41,41))
        h.create_dataset('label', data=label, chunks=(64,3,41,41))
#    with open('{}_h5.txt'.format(setname), 'w') as f:
#        f.write(h5_filename)

if __name__ == "__main__":
#    gen_hdf5('./jgj', 'train_108.h5') 
#    gen_hdf5('./sr91', 'train_91.h5')
#    gen_hdf5('./bsd200', 'train_200.h5')
    gen_hdf5('./sr_self', 'train_55.h5')
#    gen_hdf5('./org0', 'train_org0.h5') 
#    gen_hdf5('./org1', 'train_org1.h5') 
#    gen_hdf5('./org2', 'train_org2.h5') 
#    gen_hdf5('./org3', 'train_org3.h5') 
#    gen_hdf5('./org4', 'train_org4.h5') 
#    gen_hdf5('./org5', 'train_org5.h5') 
#    gen_hdf5('./org6', 'train_org6.h5') 
#    gen_hdf5('./org7', 'train_org7.h5') 
#    gen_hdf5('./org8', 'train_org8.h5') 
#    gen_hdf5('./org9', 'train_org9.h5') 
#    gen_hdf5('./test', 'test-py.h5') 
    

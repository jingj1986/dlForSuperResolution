#!/usr/bin/python
#Author: Guojun.jin <>
#Date:   2016-10-13
#Desc:
#   1. Load config: caffemode ip,path, lasted/fix_int, sleep, prototxt
#                   Test images set
#                   Quoats 
#   2. gen quoats set according to image set
#   3. for range(10000 config) 
#        gen caffe net, get caffemode (LASTED/FIX_INT)
#        for image set
#          get echo quota, and check end
#        sleep for next       
#  
import sys
import json
from utils import *
from skimage import io
import cv2
import os
import time
import caffe

quota_list=[]
img_gts = []
img_pads = []
img_ycbcrs = []
caf_conf = {}



def load_conf(conf_file):
    f = file(conf_file)
    conf = json.load(f)
    f.close()

    caf_conf["ip"] = conf["caffe"]["ip"]
    caf_conf["path"] = conf["caffe"]["path"]
    caf_conf["step"] = conf["caffe"]["step"]
    caf_conf["sleep"] = conf["caffe"]["sleep"]
    caf_conf["deploy"] = conf["caffe"]["deploy"]
    caf_conf["fetch_type"] = conf["caffe"]["type"]
    ## 
    for img in conf["images"]:
        quotas = []
        for quota in conf["quotas"]:
            obj = str_to_class(quota["name"])(quota["var"])
            quotas.append(obj)
        quota_list.append(quotas)
        prepare_sm(img)
    print "="*5 + " end of load_conf " + "="*5

# save transfered image info, so not need to transfer again!
def prepare_sm(img_str):
    img = io.imread(img_str)
    img_gts.append(img)
    h,w = img.shape[:2]
    #img_sm = cv2.resize(img, (w/2,h/2), interpolation=cv2.INTER_CUBIC)
    img_sm = cv2.resize(img, (w/2,h/2))
    img_pad = cv2.resize(img_sm, (w,h), interpolation=cv2.INTER_CUBIC)
    #img_pad = cv2.resize(img_sm, (w,h))
    img_pads.append(img_pad)
    img_ycbcrs.append(color.rgb2ycbcr(img_pad).astype(np.float))

## new caffemodel for predict.
# call bash script to sysnc these models
def get_new_caffemodel():
    f=open('./caffemode.id')
    old_id=f.read().strip('\n')
    f.close()

    for i in range(140):
        cmd="bash ./sync_caffemodel.sh " + caf_conf["ip"] + " " + caf_conf["path"] + " " + \
                caf_conf["fetch_type"] + " " + str(caf_conf["step"]) + " " + str(old_id)
        print cmd
        os.system(cmd)
        f=open('./caffemode.id')
        new_id=f.read().strip('\n')
        f.close()
        if int(new_id) > int(old_id):
            predict(new_id) 
            old_id = new_id
        print "sleep "  + str(caf_conf["sleep"])
        time.sleep(caf_conf["sleep"])
     
##   
def predict(caffe_idx):
    local_root="/home/jgj/project/caffe/pyTest/quota/"
    net = caffe.Net(str(caf_conf['deploy']), './check_quota.caffemodel', caffe.TEST)
    
    f = open('./quota_result', 'a')
    f.writelines("\n===%s====\n" % caffe_idx) 

    need_end = 0
    val1_sum=0 
    val2_sum=0
    for i in range(len(img_ycbcrs)):
        ## TODO gray Image
        h_ycbcr = img_ycbcrs[i]
        h, w = h_ycbcr.shape[:2]
        net.blobs['data'].reshape(1, 1, h, w)
        caffe.set_mode_gpu()
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) ## 1,1,256,256
        out = net.forward_all(data=np.asarray([transformer.preprocess('data', h_ycbcr[:,:,0]/255.0)]))
        ret = out['sum'][0]
        img_pred = colorize(ret[0,:,:], h_ycbcr[:,:,1], h_ycbcr[:,:,2])
        for quota in quota_list[i]:
            val1 = quota.get_quota(img_gts[i], img_pads[i], 2)
            val2 = quota.get_quota(img_gts[i], img_pred, 2)
            val1_sum = val1_sum + float(val1)
            val2_sum = val2_sum + float(val2)
            quota_name = quota.get_name()
            if quota.need_end():
                need_end = need_end + 1 
            print i, quota_name, val1, val2
            f.writelines("%d %s %s %s\n" %(i, quota_name, val1, val2))
    val1_avg = val1_sum/len(img_ycbcrs)
    val2_avg = val2_sum/len(img_ycbcrs)
    print "sum", val1_avg, val2_avg
    f.writelines("%s %s %s %s\n" %("100", "sum", val1_avg, val2_avg))

    f.close()
    if (need_end * 2) > len(img_ycbcrs):
        print "more than half set need to end, so exit!"
        #os._exit() 

if __name__ == "__main__":
    load_conf('./config.json')
    get_new_caffemodel()
    #predict(100)


#!/bin/bash
# Author: Guojun Jin <>
# Date:   2016-10-13
# Desc:   User bash/shell ssh and scp cmd to sysnc the new caffemode
#

ip=$1
path=$2
s_type=$3
step=$4
old_id=$5

function sysnc_lasted(){
    #file_name=`ssh $ip "ls -t ${path}/*.caffemodel" | head -n 1`
    file_name=`ls -t ${path}/*.caffemodel | head -n 1`
    tmp_name=${file_name##*_}
    new_id=${tmp_name%%.*}

    if [ $new_id -gt ${old_id} ]; then
        #scp ${ip}:$file_name "./check_quota.caffemodel"
        cp $file_name "./check_quota.caffemodel"
        echo $new_id > caffemode.id
    fi
}

function sysnc_fix() {
    new_id=`echo "${old_id}+${step}" | bc -l`
    file_name="${path}/_iter_${new_id}.caffemodel"
    #exist=`ssh $ip "ls $file_name" | wc -l`
    exist=`ls $file_name | wc -l`
    if [ "$exist" == "1" ]; then
        #scp ${ip}:$file_name "./check_quota.caffemodel"
        cp $file_name "./check_quota.caffemodel"
        echo $new_id > caffemode.id
    fi
}

function main() {
    if [ $s_type == "LASTED" ]; then
        sysnc_lasted
    elif [ $s_type == "FIX_INT" ]; then
        sysnc_fix
    fi
}

main $@

#!/bin/bash
#DESC:训练时，每次迭代输出loss，数据过多，难发现变化趋势
#     按照固定间隔计算loss平均值，并通过gnuplot显示
#DATE:   2016-12-06
#AUTHOR: Guojun Jin <jingj1986@163.com>


function filter() {
	idx=0
	len=4
	array=(0 0 0 0)
	
	while read line
	do
	    array[idx]=$line
	    idx=`echo "($line+1)%$len" | bc -l`
	    sum_val=0
	    for val in ${array[@]}
	    do
	        sum_val=`echo "$sum_val + $val" | bc -l`
	    done
	    var=`echo "$sum_val/$len"| bc -l`
	    echo $var
	done < loss
}

end=125

stride=1000
function avg() {
    for i in `seq 0 $end`
    do
        startidx=`echo "$i*${stride} + 1" | bc -l`
        endidx=`echo "($i+1)*${stride}" | bc -l`
#        sum=`sed -n "${startidx},${endidx}p" loss | awk '{sum+=$1}END{printf("%ld",sum)}'`
        sum=`sed -n "${startidx},${endidx}p" loss | awk '{sum+=$1}END{print sum}'`
        echo "$sum"/${stride} | bc -l
    done 
}

avg

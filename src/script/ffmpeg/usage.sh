#DESC   FFMPEG USAGE EXAMPLES
#DATE:  2017-9-21
#AUTHOR:Guojun Jin <jingj1986@163.com>

#1. copy audio info to out.wav
ffmpeg -i doctor_low.00.avi -acodec copy -vn out.wav
#2. copy video info to out.avi
ffmpeg -i doctor_low.00.avi -vcodec copy -an out.avi
#3. merge video and audio into gen.avi
ffmpeg -y -i out.avi -i out.wav -vcodec copy -acodec copy gen.avi
#4. merge video(from images stream) and audio into gen.avi
ffmpeg -y -i img/f-%05d.png -i out.wav -vcodec copy -acodec copy gen.avi
# -vcodec: video code format, such as vcodec mpeg4
# -acodec: audio vode format

# In our test case, it works with avi, but does NOT work with rmb
mencoder -ss 0:00 -endpos 5:19 xiaobingzhangga_low_01.avi -ovc copy -oac copy -o seg/first.avi

#export image, without change except block
ffmpeg -i youjidui.avi -b 6000k -minrate 6000k -maxrate 6000k -f image2 -ss 600 img-rate/f-%06d.png

#merge two videos into one, with save the old video size
ffmpeg -i sr_org.mkv -vf "[in] scale=iw:ih, pad=2*iw:ih [left]; movie=avs_org.mkv, scale=iw:ih [right]; [left][right] overlay=main_w/2:0 [out]" -b:v 768k Output.mp4
#merge two video into one, change videos with same size
ffmpeg -i sr_org.mkv -vf "[in] scale=iw/2:ih/2, pad=2*iw:ih [left]; movie=avs_org.mkv, scale=iw:ih [right]; [left][right] overlay=main_w/2:0 [out]" -b:v 9000k Output.mp4


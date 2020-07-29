#DirectShowSource("E:\qxr\图像检测\test\newfile.avi")

#DirectShowSource("E:\qxr\图像检测\划痕图片\scratch.avi")

#DirectShowSource("E:\video\f82a43c1-4e15-4754-86d0-b3d0f842794d.MPG")

SetWorkingDir("D:\soft\AVisynth\plugins\avs_script")

DirectShowSource("D:\video\out_dl.mkv")

LoadPlugin("..\Film_Restoring_vs_06_2012\scripts\plugins\DePanEstimate.dll") # or use autoloading

LoadPlugin("..\Film_Restoring_vs_06_2012\scripts\plugins\DePan.dll")         # or use autoloading

LoadPlugin("..\Film_Restoring_vs_06_2012\scripts\plugins\removedirt.dll") 

LoadPlugin("..\Film_Restoring_vs_06_2012\scripts\plugins\MVTools2.dll")

LoadPlugin("..\Film_Restoring_vs_06_2012\scripts\plugins\removegrain.dll")

Import("..\Film_Restoring_vs_06_2012\scripts\plugins\03_RemoveDirtMC.avs")

LoadPlugin("..\Film_Restoring_vs_06_2012\scripts\plugins\MVTools.dll")

LoadPlugin("..\Film_Restoring_vs_06_2012\scripts\plugins\Motion_06Dec05B.dll")

LoadPlugin("..\Film_Restoring_vs_06_2012\scripts\plugins\autolevels_06.dll")

Loadplugin("..\Film_Restoring_vs_06_2012\scripts\plugins\warpsharp.dll")

LoadPlugin("..\Film_Restoring_vs_06_2012\scripts\plugins\Deflicker.dll")
LoadPlugin("..\Film_Restoring_vs_06_2012\scripts\plugins\SmoothDeinterlacer.dll")
LoadPlugin("..\Film_Restoring_vs_06_2012\scripts\plugins\DeSaltPepper.dll")
Import("..\Film_Restoring_vs_06_2012\scripts\plugins\RemoveNoiseMC.avs")

function DetectVCRLines(clip c,int threshold)
{
spacial_data = GeneralConvolution(ConvertToRGB(c),0,"0 -1 0 0 2 0 0 -1 0")
bar_data = ConvertToRGB(BilinearResize(spacial_data,16,c.height))
st_data = Overlay(bar_data,Trim(bar_data,1,0),mode = "subtract")
st_data2 = Greyscale(Levels(st_data,threshold,10.0,threshold+1,0,255,coring = false))
st_data3 = Greyscale(Levels(st_data2,127,10.0,128,0,255,coring = false))
st_data4 = Overlay(st_data3,st_data3,y = -1, mode = "add")
return PointResize(st_data4,c.width,c.height)
}

function deVCR(clip c,int threshold)
{
mybars = DetectVCRLines(c,threshold)
return Overlay(c,Trim(c,1,0), mask = mybars,greymask = true) 
}


i = ConvertToYV12()

mdata = DePanEstimate(i,range=3)

#qxr add
stab = DePan(i,mdata)
#stab1 = DePanInterleave(i,mdata,prev=2,next=2) #视频交叉选取生成新的long clip 为原来的prev+next+1倍
#stab = SelectEvery(stab1,5,3)  #视频帧的选择，3,5,8,10
#stab = SelectEvery(stab, 20, 1)
#CLEANING PARAMETERS
#--------------------------------------------------------------------------------------------------------------

dirt_strenght=100  # set this lower for clean films.

#stab = DePanStabilize(i, data=mdata)
#SmoothDeinterlace()
cleaned= RemoveDirtMC(stab,dirt_strenght)
#cleaned1 = deVCR(cleaned,0)
#cleaned1 = deVCR(cleaned1,1)
#cleaned1 = deVCR(cleaned1,1)
#cleaned1 = deVCR(cleaned1,1)
cleaned1 = DeSaltPepper(cleaned,opt="both",UV=true, tol=0, avg=true, feedback=10)
#cleaned1 =  RemoveDirtMC(cleaned1, 100)
#cleanned = RemoveNoiseMC(stab, rdlimit=10,rgrain=2,denoise=8)

#result1 = cleaned.converttoRGB24().converttoYV12()
result1 = cleaned1.converttoRGB24().converttoYV12()
Eval("result1")

#resultS1= stackhorizontal(subtitle(i,"original",size=28,align=1),subtitle(result1,"result",size=28,align=2))
#Eval("resultS1")



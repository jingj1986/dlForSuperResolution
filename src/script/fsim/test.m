close all;

addpath('FeatureSIM');
high = imread('1_high.png');
high = imresize(high, [480,320], 'bilinear');
low = imread('1_out.png');
%low = rgb2ycbcr(low);
%low = low(:,:,1);
%low = imresize(low, 2, 'bilinear');

[fsim, fsimc] = FeatureSIM(low, high)
%fsim
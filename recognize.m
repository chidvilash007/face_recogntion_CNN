clear
clc
load net_cele
im = imread ('2.jpg');
YPred = classify(net_cele,im);
figure
label = char(YPred);
imshow(im)
title(label)
clear all, close all;
addpath(genpath('lib'));
load ('300w.mat');
im = imread ('2.jpg');
imshow(im)
title('Original Image')
[x,y,z]=size(im);
if z==1
    imGray=im;
else
    imGray = rgb2gray(im);
end
bbox = runFaceDet(imGray);
iPts = initPts(model, bbox, 0.80); 
pts = SDMApply(imGray, iPts, model);
 
figure
imshow (im),title('Mapping of facial land marks'), hold on;
plot(pts(:,1), pts(:,2), 'g.', 'MarkerSize', 12);
mini =min(pts);
maxi =max(pts);
new=im(mini(2)-(0.10*bbox(4)):maxi(2)+(0.10*bbox(4)),mini(1):maxi(1),:);
new =imresize(new,[x y]);
figure
imshow(new)
title('Cropped Face')
load net_cele
YPred = classify(net_cele,im);
figure
label = char(YPred);
imshow(new)
title(label)
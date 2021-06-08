clear
clc
close all
 
imds = imageDatastore('C:\matlab\database at&t\ifw\lfw','IncludeSubfolders',true,'LabelSource','foldernames');
 
figure;
perm = randperm(13233,10000);
for i = 1:10
    subplot(2,5,i);
    imshow(imds.Files{perm(i)});
end
 
 
labelCount = countEachLabel(imds);
 
img = readimage(imds,1);
size(img)
 
[imdsValidation,imdsTrain] = splitEachLabel(imds,0.01,'randomize');
 
numImages = numel(imdsTrain.Files);
idx = randperm(numImages,10);
 
 
for i = 1:10
    subplot(2,5,i)
    I = readimage(imdsTrain, idx(i));
    imshow(I)
end
    layers = [
    imageInputLayer([250 250 3])
    convolution2dLayer(3,1, 'Stride',1,'Padding','same')
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,2,'Padding',[0 0])
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,3,'Padding',[0 0])
    reluLayer
    convolution2dLayer(5,3,'Padding',[0 0])
    reluLayer
    convolution2dLayer(7,3,'Padding',[0 0])
    reluLayer
    batchNormalizationLayer
    fullyConnectedLayer(5749)
    softmaxLayer
    classificationLayer];
 
options = trainingOptions('adam','LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.1,'LearnRateDropPeriod', 2, 'Shuffle','every-epoch','Plots','training-progress','MaxEpochs', 10,'MiniBatchSize', 64);
net_cele = trainNetwork(imdsTrain,layers,options);
%opimizer used adam
save net_cele 
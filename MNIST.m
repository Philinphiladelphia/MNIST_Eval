function net = MNIST (trainData,valData,trainLabels,valLabels)

trainData = reshape(trainData, [size(trainData,1),size(trainData,2),1,size(trainData,3)]);

cellData = cell([size(valData,3),1]);
for i  = 1:size(valData,3)
    cellData{i} = valData(:,:,i);
end

valData = table(cellData, categorical(valLabels));

options = trainingOptions('sgdm','MaxEpochs',3,'ValidationData',valData,'ValidationFrequency',1000,'Verbose',false, 'Plots','training-progress');

layers = [
    imageInputLayer([20 20 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer  
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

Y = categorical(trainLabels);
net = trainNetwork(trainData,Y,layers,options);


%TODO: figure out 265/ by 265 math
%TODO: split + segment network so that only edges of transformation are
%visible?
%TODO: Bother karthik about EMG signals
%TODO: Find the use of real and complex portions of fourier trans.

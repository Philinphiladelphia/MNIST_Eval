[valData, valLabels] = readMNIST('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte',2000, 0 ); %max 10,000

[trainData, trainLabels] = readMNIST('train-images-idx3-ubyte','train-labels-idx1-ubyte',12000, 0 ); %max 60,000

trainData = cat(3,trainData,valData);

trainLabels = [trainLabels;valLabels];


net = MNIST(trainData,valData,trainLabels,valLabels);

%CODE FOR K-FOLD VALIDATION

% iter = 1;
% for i = 0.1:0.1:1
%     startPoint = ceil((i-0.1)*size(trainData,3));
%     endPoint = ceil((i)*size(trainData,3));
%     currentData = cat(3,trainData(:,:,1:startPoint),trainData(:,:,endPoint:end));
%     currentLabels = [trainLabels(1:startPoint);trainLabels(endPoint:end)];
%     valData = trainData(:,:,startPoint+1:endPoint-1);
%     valLabels = trainLabels(startPoint+1:endPoint-1);
%     net = MNIST(currentData,valData,currentLabels,valLabels)
%     
%     valData = reshape(valData, [size(valData,1),size(valData,2),1,size(valData,3)]);
%     predictions = classify(net,valData);
%     accuracy(iter) = 100*numel(find(categorical(valLabels)==(predictions)))/numel(valLabels);
%     iter = iter +1;
%     
% end

%final = mean(accuracy)
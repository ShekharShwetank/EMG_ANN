features = readmatrix('Extreacted_features.csv');

labels = [ones(1, size(features, 1)/2), zeros(1, size(features, 1)/2)];

labels_one_hot = full(ind2vec(labels+1));

SVMModel = fitcsvm(features, ones(size(features, 1), 1), 'KernelScale', 'auto', 'OutlierFraction', 0.05, 'Standardize', true);

[~, scores] = predict(SVMModel, features);

[~, threshold] = min(scores);

outliers = scores < threshold;

disp('Indices of outliers:');
disp(find(outliers));

hiddenLayerSize = 24; 
net = fitnet(hiddenLayerSize, 'trainscg'); 

trainRatio = 0.7; 
valRatio = 0.15;
testRatio = 0.15;
net.divideParam.trainRatio = trainRatio;
net.divideParam.valRatio = valRatio;
net.divideParam.testRatio = testRatio;

net.trainParam.epochs = 20;

net.trainParam.goal = 1e-10;

[net,tr] = train(net,features',labels_one_hot);

predictions = net(features');
errors = gsubtract(predictions,labels_one_hot);
performance = perform(net,labels_one_hot,predictions);

predicted_labels = vec2ind(predictions);
true_labels = vec2ind(labels_one_hot);
accuracy = sum(predicted_labels == true_labels) / length(true_labels);
disp(['Accuracy: ', num2str(accuracy)]);

confusionMatrix = confusionmat(true_labels, predicted_labels);
precision = confusionMatrix(2,2) / sum(confusionMatrix(:,2));
recall = confusionMatrix(2,2) / sum(confusionMatrix(2,:));
f1_score = 2 * (precision * recall) / (precision + recall);
disp(['Precision: ', num2str(precision)]);
disp(['Recall: ', num2str(recall)]);
disp(['F1-score: ', num2str(f1_score)]);

view(net)

figure, plotperform(tr)
figure, plottrainstate(tr)
figure, plotconfusion(labels_one_hot, predictions)
figure, ploterrhist(errors)

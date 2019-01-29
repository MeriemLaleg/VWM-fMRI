function [acc1, acc2]= Apply_LeavOut_classification(X, Y)

%% Apply GNB
C = cvpartition(Y, 'LeaveOut');
err = zeros(C.NumTestSets,1);
for i = 1:C.NumTestSets
    trIdx = C.training(i);
    teIdx = C.test(i);
    [classifier] = trainClassifier(X(trIdx,:),Y(trIdx), 'nbayes');   %train classifier
    [predictions] = applyClassifier(X(teIdx,:), classifier);       %test it
    [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',Y(teIdx));
    err(i)= 1-result{1};  % rank accuracy
end
acc1= sum(err)/sum(C.TestSize);
%% Apply LR
C = cvpartition(Y, 'LeaveOut');
err = zeros(C.NumTestSets,1);
for i = 1:C.NumTestSets
    trIdx = C.training(i);
    teIdx = C.test(i);
    [classifier] = trainClassifier(X(trIdx,:),Y(trIdx), 'logisticRegression');   %train classifier
    [predictions] = applyClassifier(X(teIdx,:), classifier);       %test it
    [result,predictedLabels,trace] = summarizePredictions(predictions,classifier,'averageRank',Y(teIdx));
    err(i)= 1-result{1};  % rank accuracy
end
acc2= sum(err)/sum(C.TestSize);
% acc2= 1;

%% Apply SVM
% C = cvpartition(Y, 'LeaveOut');
% accuracy0 = zeros(C.NumTestSets,1);
% for i = 1:C.NumTestSets
%     trIdx = C.training(i);
%     teIdx = C.test(i);
%     Mdl= fitglm(X(1:70,:), Y(1:70,:),'linear','Distribution','binomial','link', 'logit');
%     
%     % yfit=trainedClassifier.predictFcn(testing_set);
%     yfit0 = Mdl.predict(X(71:80,:));
%     yfit0=yfit0-min(yfit0);yfit0=yfit0/max(yfit0);
%     yfit=double(yfit0>0.5);
%     
%     [accuracy0(i),sensitivity0,specificity0,precision0,gmean0,f1score0]=prediction_performance(Y(71:80), yfit);
%     
% end
% acc3= sum(accuracy0)/sum(C.TestSize);


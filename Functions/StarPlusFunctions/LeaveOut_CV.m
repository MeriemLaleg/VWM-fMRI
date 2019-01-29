function [cvErr err]= LeaveOut(X, Y)

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
cvErr= sum(err)/sum(C.TestSize);

end
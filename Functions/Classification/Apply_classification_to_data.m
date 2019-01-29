function [acc1, acc2]= Apply_classification_to_data(X, Y)

%% Apply GNB
[GNB] = trainClassifier(X,Y, 'nbayes');   %train classifier
[predictions] = applyClassifier(X,GNB);       %test it
[result,predictedLabels,trace] = summarizePredictions(predictions,GNB,'averageRank',Y);
acc1= 1-result{1}  % rank accuracy

%% Apply LR
[LR] = trainClassifier(X,Y,'logisticRegression');   %train classifier
[predictions] = applyClassifier(X,LR);       %test it
[result,predictedLabels,trace] = summarizePredictions(predictions,LR,'averageRank',Y);
acc2= 1-result{1}  % rank accuracy
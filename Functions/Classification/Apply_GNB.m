% p :     % proportion of rows to select for training

function Acc=Apply_GNB(p, examples, labels)

[X_train,y_train, X_test,y_test]=SplitData(p, examples,labels);

% train a Naive Bayes classifier
   [classifier] = trainClassifier(X_train,y_train,'nbayes');   %train classifier

% apply the Naive Bayes classifier to the training data (it's best to use cross
% validation, of course, to obtain an estimate of its true error).  The returned
% array 'predictions' is an array where predictions(k,j) = log P(example_k |
% class_j).

   [y_predicted] = applyClassifier(X_test,classifier);       %test it

% summarize the results of the above predictions.   

 [result,predictedLabels,trace] = summarizePredictions(y_predicted,classifier,'averageRank',y_test);
 Acc=1-result{1};  % rank accuracy
 y_test';
 
function [X_train,y_train, X_test,y_test]=SplitData(trainRatio,X,y)

N = size(X,1);  % total number of rows

[trainInd,valInd,testInd] = dividerand(N,trainRatio,0,1-trainRatio);


X_train = X(trainInd,:) ;
y_train = y(trainInd) ;

X_test = X(testInd,:) ;
y_test = y(testInd) ;
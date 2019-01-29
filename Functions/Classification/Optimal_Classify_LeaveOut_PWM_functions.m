function [accuracy, Sparse_P_ratio, Sparse_S_ratio]= Optimal_Classify_LeaveOut_PWM_functions(X,Y, intervals1)
addpath ./Leave1out_PWM

% global PWM_P PWM_S
catogries1= [1 2 3 4 5 6];

% intervals1= [-8 -1 2 5 8]+1.5;

% intervals1= [-8 -1 2 5 8];
% mu=0;sigma0=2.68;
% intervals1= mu+sigma0*[-2 -1 0 1 2]; % 1 1 1 1  0.98 1
% intervals1= mu+(1.1*sigma0)*[-2 -1 0 1 2]; % 1 1 1 1  1 1
% intervals1= mu+(1.2*sigma0)*[-2 -1 0 1 2]; % 1 1 1 1  1 1
% intervals1= mu+(1.3*sigma0)*[-2 -1 0 1 2]; % 1 1 1 1  1 1
% intervals1= mu+(1.4*sigma0)*[-2 -1 0 1 2]; % 1 1 1 0.5  1 1
% intervals1= mu+(1.5*sigma0)*[-2 -1 0 1 2]; % 1 1 1 0.5  1 1
%  intervals1= mu+(2*sigma0)*[-2 -1 0 1 2]; % 1 0.5 1 0.5  1 1
%  intervals1= mu+(2.5*sigma0)*[-2 -1 0 1 2]; % 1 0.5 1 0.5  1 1
%  intervals1= mu+(5*sigma0)*[-2 -1 0 1 2]; % 1 0.5 1 0.5  1 1

% intervals1= [-2 -1 0 3.5];% acc=[1 0.5625 1 1 0.5 1]
% intervals1= [-2 -1 0 3.5];

% intervals1= [-1 -1 -0.6 3 ];
% intervals1= [-3 -1 1 3];


%% Leave one sample Out Cross-Validation
C = cvpartition(Y, 'LeaveOut');

for num_fold = 1:C.NumTestSets
    clearvars -except X Y catogries1 catogries1 PWM_P PWM_S intervals1 intervals1 acc1 acc2 num_fold C outcome outcome2 outcome classPrior
    
    trIdx = C.training(num_fold);
    teIdx = C.test(num_fold);
    Idx= find(teIdx);
    X_train= X(trIdx,:);
    X_test= X(teIdx,:);
    
    Y_train= Y(trIdx);
    Y_test= Y(teIdx);
    
    Xp=X_train(Y_train==1,:);   Np=size(Xp, 1);
    Xs=X_train(Y_train==2,:);   Ns=size(Xs, 1);
    
    Xp= mapping_levels(Xp,intervals1, catogries1);
    Xs= mapping_levels(Xs,intervals1, catogries1);
    
    PWM_P = Generate_PWM_matrix(Xp, catogries1);
    PWM_S = Generate_PWM_matrix(Xs, catogries1);
    
    X_train_levels=[Xp;Xs];
    PWM_f_train= Generate_PWM_features(X_train_levels, PWM_P, PWM_S);
    
    X_test_levels= mapping_levels(X_test, intervals1, catogries1);
    PWM_fP_test= Generate_PWM_features(X_test_levels, PWM_P, PWM_S);

    
    %% Train and test the model
    [classifier] = trainClassifier(PWM_f_train,Y_train, 'logisticRegression');   %train classifier
 
    %% Test1 the model
    [predictions1] = applyClassifier(PWM_fP_test, classifier);       %test it
    [result1,predictedLabels1,trace1] = summarizePredictions(predictions1,classifier,'averageRank',Y_test);
    acc1(num_fold)= 1-result1{1};  % rank accuracy
end

%% Average Accuracy 
accuracy= sum(acc1)/sum(C.TestSize);
%% Find the sparsity of PWM
Sparse_P= nnz(~PWM_P);
Sparse_S= nnz(~PWM_S);
Sparse_P_ratio= (Sparse_P/(size(PWM_P,1)*size(PWM_P,2)))*100;
Sparse_S_ratio= (Sparse_S/(size(PWM_S,1)*size(PWM_S,2)))*100;

end

%% Funtions

function X=mapping_levels(X,intervals, catogries)

if size(catogries,2) ==4
    for i=1:size(X,1)
        for j=1:size(X,2)
            if X(i,j) <= intervals(1)
                X(i,j)= catogries(1);
            elseif X(i,j) <= intervals(2)
                X(i,j)= catogries(2);
            elseif X(i,j) <= intervals(3)
                X(i,j)= catogries(3);
            else
                X(i,j)= catogries(4);
            end
        end
    end

elseif size(catogries,2) ==5
    for i=1:size(X,1)
        for j=1:size(X,2)
            if X(i,j) <= intervals(1)
                X(i,j)= catogries(1);
            elseif X(i,j) <= intervals(2)
                X(i,j)= catogries(2);
            elseif X(i,j) <= intervals(3)
                X(i,j)= catogries(3);
            elseif X(i,j) <= intervals(4)
                X(i,j)= catogries(4);
            else
                X(i,j)= catogries(5);
            end
        end
    end
elseif size(catogries,2) ==6
    for i=1:size(X,1)
        for j=1:size(X,2)
            if X(i,j) <= intervals(1)
                X(i,j)= catogries(1);
            elseif X(i,j) <= intervals(2)
                X(i,j)= catogries(2);
            elseif X(i,j) <= intervals(3)
                X(i,j)= catogries(3);
            elseif X(i,j) <= intervals(4)
                X(i,j)= catogries(4);
            elseif X(i,j) <= intervals(5)
                X(i,j)= catogries(5);
            else
                X(i,j)= catogries(6);
            end
        end
    end
end
end

function PWM_matrix= Generate_PWM_matrix(X_train, catogries)
catogries=size(catogries,2);
PWM_matrix= zeros(5, size(X_train,2)); %The weight matrix of picture

for k=1:catogries
    for i=1:size(X_train, 2)
        PWM_matrix(k,i)= sum(X_train(:, i) == k)/size(X_train,1);
    end
end


end

function PWM_features= Generate_PWM_features(X_train, PWM_P, PWM_S)
    
PWM_f1= zeros(size(X_train,1), size(X_train,2));
PWM_f2= zeros(size(X_train,1), size(X_train,2));


for i=1:size(X_train,1)
    for j=1:size(X_train,2)
        pwm_idx=X_train(i,j);
        PWM_f1(i,j)= PWM_P(pwm_idx,j);
        PWM_f2(i,j)= PWM_S(pwm_idx,j);
    end
end

f1=sum(PWM_f1,2);
f2=sum(PWM_f2,2);
PWM_features=[f1 f2];

end

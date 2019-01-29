% Classify the VWM using Leave one sample out and returns the classificatiom accurcay
function [outcome, accuracy1]= Classify_LeaveOut_PWM_functions(X,Y)
addpath ./Leave1out_PWM

catogries1= [1 2 3 4 5 6];

% intervals1= [-8 -1 2 5 8]+1.5;

% intervals1= [-8 -1 2 5 8];
mu=0;sigma0=2.68;
intervals1= mu+sigma0*[-2 -1 0 1 2]; % 1 1 1 1  0.9875 1
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
    clearvars -except X Y catogries1 catogries1 VWM_P VWM_S intervals1 intervals1 acc1 acc2 num_fold C outcome outcome2 outcome classPrior
    
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
    
    VWM_P = Generate_VWM_matrix(Xp, catogries1);
    VWM_S = Generate_VWM_matrix(Xs, catogries1);
    
    X_train_levels=[Xp;Xs];
    VWM_f_train= Generate_VWM_features(X_train_levels, VWM_P, VWM_S);
    
    X_test_levels= mapping_levels(X_test, intervals1, catogries1);
    VWM_fP_test= Generate_VWM_features(X_test_levels, VWM_P, VWM_S);

    
    %% Train and test the model
    [classifier] = trainClassifier(VWM_f_train,Y_train, 'logisticRegression');   %train classifier
 
    %% Test the model
    [predictions1] = applyClassifier(VWM_fP_test, classifier);       %test it
    [result1,predictedLabels1,trace1] = summarizePredictions(predictions1,classifier,'averageRank',Y_test);
    acc1(num_fold)= 1-result1{1};  % rank accuracy
    global scores
    outcome(num_fold,:)=[Y_test  acc1(num_fold) scores];
end

%% Average Accuracy 
accuracy1= sum(acc1)/sum(C.TestSize);

end

%% Funtions
%% Replace each voxel intensity by integer number for each of the specified intervals
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
%% Computes the Voxel Probability Matrix (VPM)
function VWM_matrix= Generate_VWM_matrix(X_train, catogries)
catogries=size(catogries,2);
VWM_matrix= zeros(5, size(X_train,2)); %The weight matrix of picture

for k=1:catogries
    for i=1:size(X_train, 2)
        VWM_matrix(k,i)= sum(X_train(:, i) == k)/size(X_train,1);
    end
end


end
%% Converts the VPM to Voxel Weight Matrix (VWM)
function VWM_features= Generate_VWM_features(X_train, VWM_P, VWM_S)
    
PWM_f1= zeros(size(X_train,1), size(X_train,2)); %f1 is the first feature of VWM
PWM_f2= zeros(size(X_train,1), size(X_train,2)); %f1 is the second feature of VWM

% replace the integer values by its probability from VPM
for i=1:size(X_train,1)
    for j=1:size(X_train,2)
        pwm_idx=X_train(i,j);
        PWM_f1(i,j)= PWM_P(pwm_idx,j);
        PWM_f2(i,j)= PWM_S(pwm_idx,j);
    end
end
% sum all the probabilities to get the VWM
f1=sum(PWM_f1,2);
f2=sum(PWM_f2,2);
PWM_features=[f1 f2];

end
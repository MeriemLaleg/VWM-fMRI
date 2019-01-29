% Classify the VWM using Leave one sample out and returns the classificatiom accurcay

function accuracy= classify_VW_features(X,Y)

catogries1= [1 2 3 4 5 6];

%mu=0;sigma0=2.68;
%mu=0.3513; sigma0= 2.7381;
%mu=0; sigma0= 2.7381;
mu= 0.34909281; sigma0= 2.718391934;

intervals1= mu+sigma0*[-2 -1 0 1 2]; % 1 1 1 1  0.9875 1
% intervals1= mu+(1.1*sigma0)*[-2 -1 0 1 2]; % 1 1 1 1  1 1
% intervals1= mu+(1.2*sigma0)*[-2 -1 0 1 2]; % 1 1 1 1  1 1
% intervals1= mu+(1.3*sigma0)*[-2 -1 0 1 2]; % 1 1 1 1  1 1
% intervals1= mu+(1.4*sigma0)*[-2 -1 0 1 2]; % 1 1 1 0.5  1 1
% intervals1= mu+(1.5*sigma0)*[-2 -1 0 1 2]; % 1 1 1 0.5  1 1
%  intervals1= mu+(2*sigma0)*[-2 -1 0 1 2]; % 1 0.5 1 0.5  1 1
%  intervals1= mu+(2.5*sigma0)*[-2 -1 0 1 2]; % 1 0.5 1 0.5  1 1
%  intervals1= mu+(5*sigma0)*[-2 -1 0 1 2]; % 1 0.5 1 0.5  1 1



%% Leave one sample Out Cross-Validation
C = cvpartition(Y, 'LeaveOut');

for num_fold = 1:C.NumTestSets
    clearvars -except X Y catogries1 VWM_P VWM_S intervals1 intervals1 acc num_fold C
    
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

    X_VWM_train= [X(trIdx,:) VWM_f_train];
    X_VWM_test= [X(teIdx,:) VWM_fP_test];
    
    %% Train and test the model
    [classifier] = trainClassifier(VWM_f_train,Y_train, 'logisticRegression');   %train classifier

    %% Test the model
    [predictions1] = applyClassifier(VWM_fP_test, classifier);       %test it

    [result1,predictedLabels1,trace1] = summarizePredictions(predictions1,classifier,'averageRank',Y_test);
    acc(num_fold)= 1-result1{1};  % rank accuracy

end

%% Average Accuracy 
accuracy= sum(acc)/sum(C.TestSize);


%% Find the sparsity of VWM
% Sparse_P= nnz(~VWM_P);
% Sparse_S= nnz(~VWM_S);
% Sparse_P_ratio= (Sparse_P/(size(VWM_P,1)*size(VWM_P,2)))*100;
% Sparse_S_ratio= (Sparse_S/(size(VWM_S,1)*size(VWM_S,2)))*100;
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
VWM_matrix= zeros(5, size(X_train,2));

for k=1:catogries
    for i=1:size(X_train, 2)
        VWM_matrix(k,i)= sum(X_train(:, i) == k)/size(X_train,1);
    end
end


end
%% Converts the VPM to Voxel Weight Matrix (VWM)
function VWM_features= Generate_VWM_features(X_train, VWM_P, VWM_S)
    
VWM_f1= zeros(size(X_train,1), size(X_train,2)); %f1 is the first feature of VWM
VWM_f2= zeros(size(X_train,1), size(X_train,2)); %f1 is the second feature of VWM

% replace the integer values by its probability from VPM
for i=1:size(X_train,1)
    for j=1:size(X_train,2)
        VWM_idx=X_train(i,j);
        VWM_f1(i,j)= VWM_P(VWM_idx,j);
        VWM_f2(i,j)= VWM_S(VWM_idx,j);
    end
end
% sum all the probabilities to get the VWM
f1=sum(VWM_f1,2);
f2=sum(VWM_f2,2);
VWM_features=[f1 f2];

end
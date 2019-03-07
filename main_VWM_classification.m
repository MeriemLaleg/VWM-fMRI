% clear all; clc; close all; warning('off')
% addpath ./Functions;addpath Functions/Classification;
% addpath ./Functions/StarPlusFunctions; addpath ./Functions/VWM

%% addpath to StarPlus Dataset 

for subject=1:6
    switch(subject)
        case 1
            load('./Dataset/data-starplus-04799-v7.mat')
            convert_single_subject_data_to_matrix
            Accuracy(subject,1)= classify_VW_features(X,Y);
        case 2
            load('./Dataset/data-starplus-04820-v7.mat')
            convert_single_subject_data_to_matrix
            Accuracy(subject,1)= classify_VW_features(X,Y);
        case 3
            load('./Dataset/data-starplus-04847-v7.mat')
            convert_single_subject_data_to_matrix
            Accuracy(subject,1)= classify_VW_features(X,Y);
            
        case 4
            load('./Dataset/data-starplus-05675-v7.mat')
            convert_single_subject_data_to_matrix
            Accuracy(subject,1)= classify_VW_features(X,Y);
            
        case 5
            load('./Dataset/data-starplus-05680-v7.mat')
            convert_single_subject_data_to_matrix
            Accuracy(subject,1)= classify_VW_features(X,Y);
            
        case 6
            load('./Dataset/data-starplus-05710-v7.mat')
            convert_single_subject_data_to_matrix
            Accuracy(subject,1)= classify_VW_features(X,Y);
    end
    
    
Accuracy_av= sum(Accuracy)/size(Accuracy,1)

end

clear all
load('data-starplus-04847-v7.mat')
trials=find([info.cond]>1);

%%
[info0,data0,meta0]=transformIDM_selectTrials(info,data,meta,trials);

%% Select ROI
[info1,data1,meta1] = transformIDM_selectROIVoxels(info0,data0,meta0,{'CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'})

[infoP1,dataP1,metaP1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='P'));
[infoS1,dataS1,metaS1]=transformIDM_selectTrials(info1,data1,meta1,find([info1.firstStimulus]=='S'));
%% the second 8 seconds 
[infoP3,dataP3,metaP3]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[17:32]);
[infoS3,dataS3,metaS3]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[17:32]);

%% the first 8 seconds 
[infoP2,dataP2,metaP2]=transformIDM_selectTimewindow(infoP1,dataP1,metaP1,[1:16]);
[infoS2,dataS2,metaS2]=transformIDM_selectTimewindow(infoS1,dataS1,metaS1,[1:16]);

[X_P2,labelsP2,exInfoP2]=idmToExamples_condLabel(infoP2,dataP2,metaP2);
[X_P3,labelsP3,exInfoP3]=idmToExamples_condLabel(infoP3,dataP3,metaP3);
[X_S2,labelsS2,exInfoS2]=idmToExamples_condLabel(infoS2,dataS2,metaS2);
[X_S3,labelsS3,exInfoS3]=idmToExamples_condLabel(infoS3,dataS3,metaS3);
 
%% firdt stri : use only first stimulus

for k=1:size(infoP2,2)
    if strcmp('P',infoP2(k).firstStimulus)  
        label_P(k)=1;
    end
end
label_P=label_P';
for k=1:size(infoS2,2)

    if strcmp('S',infoS2(k).firstStimulus)    
        label_S(k)=2;
    end
end
label_S=label_S';

%%  Build clasifiaction data
X=[X_P2;X_S2];
y=[label_P;label_S];
%% Shuffle data
[X,y,shuffledRow] = shuffleRow(X,y);

%% Run Classification
for l=1:10
    Acc(l)=Apply_GBN(0.72, X, y);
    plot(Acc);
end


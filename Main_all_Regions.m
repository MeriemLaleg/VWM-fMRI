clear all; clc; close all; warning('off')
addpath ./Functions;addpath Functions/Classification;
addpath ./Functions/StarPlusFunctions; addpath ./Functions/VWM


All_regions={'CALC' 'LIPL'  'LTRIA' 'LOPER' 'LIPS' 'LDLPFC' 'LT'};
N_rg=max(size(All_regions));

cnt=0;ROI_Av_acc=[]; ROI_names=[];

for k=1:N_rg
comb = combnk(1:N_rg,k);

    for ROI=1:size(comb,1)

        for i=1:size(comb,1)

            clearvars ROI_regions
            Cmb_names='';

            for j=1: size(comb,2)
                ROI_regions{j}=All_regions{comb(ROI,j)};
                Cmb_names=strcat(ROI_regions{j}, '+', Cmb_names);
            end

            Cmb_names=Cmb_names(1:end-1)

            %% run the VWM for region
            main_VWM_classification

            % update accuravy for each combinaiton 
            cnt=cnt+1;
            ROI_names{cnt}=Cmb_names;
            ROI_Av_acc(cnt)=Accuracy_av;

        end


    end
    
    save('All_combination_ROI.mat','ROI_names','ROI_Av_acc')
end



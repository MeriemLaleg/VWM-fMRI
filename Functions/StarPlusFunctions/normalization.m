%%
function x = normalizeTrial(data)
    norm_data2= zeros(size(data1{1,1},1),size(data1{1,1},2) );

    for i=1:size(data1{1,1},1)
        norm_data2(i,:) = (data1{1,1}(i,:) - min(data1{1,1}(i,:))) / ( max(data1{1,1}(i,:)) - min(data1{1,1}(i,:)) );

    end
%% 
% m=magic(5);
% norm_data1= zeros(size(m,1),size(m,2) );
% 
% for i=1:size(m,1)
%     norm_data1(i,:) = (m(i,:) - min(m(i,:))) / ( max(m(i,:)) - min(m(i,:)) );
% 
% end


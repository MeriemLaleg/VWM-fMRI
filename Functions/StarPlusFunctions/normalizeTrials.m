function norm_data = normalizeTrials(data, all_array)

norm_data= zeros(size(data,1),size(data,2));

if all_array == string('true')
    X_max= max2(data);
    X_min= min2(data);
    X_max_min= X_max - X_min;
    
    for i=1:size(data,1)
        for j=1:size(data,2)
            norm_data(i,j)= (data(i,j) - X_min)/ X_max_min;
        end
    end


else

    for i=1:size(data,1)
        norm_data(i,:) = (data(i,:) - min(data(i,:))) / ( max(data(i,:)) - min(data(i,:))) ;
    end
end
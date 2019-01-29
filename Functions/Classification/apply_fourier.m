function [X_FT,Max_X_FT,I,X_DC] = apply_fourier(X, normalize)

X_FT= zeros(size(X,1), size(X,2));
Max_X_FT= zeros(size(X,1),1);
% X_DC= zeros(size(X,1),1);
I= zeros(size(X,1),1);

for k=1:size(X,1)
    X_FT(k,:)= angle(fft(X(k,:)));
end
X_DC= X_FT(:,1);
X_FT_without_DC=X_FT(:,2:end);
for k=1:size(X,1)
    [Max_X_FT(k,1), I(k,1)]= max(X_FT_without_DC(k,:));
end

if normalize == string('true')
    X_FT= normalizeTrials(X_FT);
    Max_X_FT= normalizeTrials(Max_X_FT);
    X_DC= normalizeTrials(X_DC);
end

% function [X_FT,Max_X_FT,I] = apply_fourier(X)
% 
% X_FT= zeros(size(X,1), size(X,2));
% Max_X_FT= zeros(size(X,1),1);
% I= zeros(size(X,1),1);
% for k=1:size(X,1)
%     X_FT(k,:)= abs(fft(X(k,:)));
%     [Max_X_FT(k,1), I(k,1)]= max(X_FT(k,:));
% end
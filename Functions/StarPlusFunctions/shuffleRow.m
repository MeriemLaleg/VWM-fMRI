function [X_perm,y_perm,shuffledRow] = shuffleRow(X,y)

[r c] = size(X);
shuffledRow = randperm(r);
X_perm = X(shuffledRow, :);
y_perm = y(shuffledRow, :);
function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
% Calculate the difference of coordinates between each sample and centroids and store it in the dimension 3 using broadcasting.
temp2 = X- reshape(centroids',1,[],K);

% calcuate the norm2 along dimension 2
temp3 = sqrt(sum(temp2.^2,2));

% pick the closest centroid for each sample. Seach min dimension 3
[_,idx] = min(temp3,[],3);


% =============================================================

end


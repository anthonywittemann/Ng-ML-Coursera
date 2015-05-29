function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


A1 = X;
A1(:, end+1) = 1; % adding the bias unit
A1 = A1(:, [end, 1:end-1]); 
%fprintf("Size A1: %f \n", size(A1));

A2 = sigmoid(Theta1 * A1');
A2 = A2';
A2(:, end+1) = 1; % adding the bias unit
A2 = A2(:, [end, 1:end-1]);
%fprintf("Size A2: %f \n", size(A2))

A3 = sigmoid(Theta2 * A2');
A3 = A3';
fprintf("A3 Rows: %f \n", size(A3, 1))
fprintf("A3 Columns: %f \n", size(A3, 2))

[predict_max, index_max] = max(A3,[], 2);
p = index_max;




% =========================================================================


end

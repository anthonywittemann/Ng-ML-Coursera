function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


totalCost = 0;
n = size(theta, 1);
for i = 1:m
    regCost = 0;
    for j=2:n
      regCost = regCost + theta(j)^2;
    endfor 

    totalCost = totalCost + ((-y(i)*log(sigmoid((X(i,:) * theta))) -
    ((1-y(i))*log(1-sigmoid((X(i,:) * theta))))) + lambda/(2*m) * regCost);
end 

% fprintf("Total Cost: %f \n", totalCost); 

J = 1 / m * totalCost;

fprintf("J: %f \n", J);


temp = theta;

totalCost1 = 0;
for i = 1:m
    totalCost1 = totalCost1 + (sigmoid(X(i,:) * temp) - y(i))*X(i,1);
endfor    
grad(1) = 1 / m * totalCost1;

    
for j=2:size(X,2)
  totalCost = 0;
  for i = 1:m
    totalCost = totalCost + (sigmoid(X(i,:) * temp) - y(i))*X(i,j) ...
                + lambda / m * temp(j);
  endfor
      
  grad(j) = 1 / m * totalCost;
endfor

fprintf("Gradient: %f \n", grad);



% =============================================================

end

function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%calculate the first part of the formula
h = X * theta;
J = 1/(2*m)*sum((h - y) .^ 2);
%calculate the second part
%make the 'theta_zero' become zero
new_theta = [0; theta(2:length(theta))];
J = J + lambda/(2*m)*sum(new_theta .^ 2);
%calculate the gradient
for (j = 1:length(grad))
  grad(j) = 1/m * sum((h - y) .* X(:,j))+((lambda/m)*new_theta(j));
  
endfor


% =========================================================================

grad = grad(:);

end

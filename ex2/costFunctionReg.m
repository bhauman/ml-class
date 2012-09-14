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

h_x = sigmoid(X * theta);
regularize_part = (sum(theta(2:length(theta)) .^ 2) * lambda) / (2 * m);
J = sum(-y .* log(h_x) - (1 - y) .* log(1 - h_x)) / length(X) + regularize_part;

partial_grad = (((h_x - y)' * X) / m)';

reg_grad_add = theta * (lambda / m);
reg_grad_add(1) = 0;
grad = partial_grad + reg_grad_add;

% =============================================================

end

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

accum = 0;
a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h_x = a3;

%for i = 1:m
%  y_i = (y(i) == 1:10);
%  accum += sum((-y_i .* log(h_x(i,:))) - ((1 - y_i) .* log(1 - h_x(i, :))));
%endfor
% J = accum / m;

ys = repmat(y, 1, num_labels) == repmat(1:num_labels, m, 1);
un_regJ = -(sum(sum(ys .* log(h_x))) + sum(sum((1 - ys) .* log(1 - h_x)))) / m;
% don't forget to strip the bias off
regular_add = lambda .* (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2))) ./ (2 * m);

J = un_regJ + regular_add;

delta_temp1 = zeros(size(Theta1));
delta_temp2 = zeros(size(Theta2));

theta2_node_size = size(Theta2(:, 2:end), 2);

d_3 = a3 - ys; % 10 x 5000
d_2 = (Theta2(:, 2:end)' * d_3')' .* sigmoidGradient(z2); % 25 * 5000

%for i = 1:1000
  % d_3 = a3(i, :) - ys(i, :);
  % d_2 = (d_3(i,:) * Theta2(:, 2:end)) .* sigmoidGradient(z2(i,:))

%  size(delta_temp2);  
  
%  a2mat = repmat(a2(i,:), size(Theta2, 1), 1) .* repmat(d_3(i,:), theta2_node_size, 1)';
%  delta_temp2 = delta_temp2 + [zeros(size(Theta2,1), 1) a2mat];

  %for i2 = 1:size(Theta2, 1)
  %  for j2 = 2:theta2_node_size
  %    delta_temp2(i2, j2) = delta_temp2(i2,j2) + a2(i,j2) * d_3(i, i2); 
  %  endfor
  % endfor
  
  
%  a1mat = [zeros(size(Theta1,1), 1) repmat(X(i,:), size(Theta1, 1), 1)] .* repmat(d_2(i,:), size(Theta1, 2), 1)';
%  delta_temp1 = delta_temp1 + a1mat;

  %for i1 = 1:size(Theta1, 1)
  %  for j1 = 2:(size(Theta1, 2) - 1)
  %    delta_temp1(i1, j1) = delta_temp1(i1,j1) + X(i, j1 - 1) * d_2(i,i1); 
  %  endfor
  %endfor

  % disp(delta_temp2);
  %disp(delta_temp1)
    
%endfor

dt2 = d_3' * a2;

delta_temp2 = dt2;

dt1 = d_2' * a1;

delta_temp1 = dt1;

Theta2_grad = delta_temp2 / m;
Theta1_grad = delta_temp1 / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

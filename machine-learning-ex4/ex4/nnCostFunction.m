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

m = size(X,1);
a1 = [ones(m,1) X];

z2 = a1 * Theta1';
a2 = [ones(size(z2,1),1), sigmoid(z2)];

z3 = a2 * Theta2';
a3 = sigmoid(z3);
H = a3;

% Recoding y vector based on label
y_vec = zeros(m,num_labels);
for idx = 1:m
    y_vec(idx,y(idx)) = 1 ; 
end

% Computing cost for each label
J =  sum(sum( (-1*y_vec).*log(H) - (1-y_vec).*log(1-H)))./m ;


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

for idx = 1:size(Theta1,1)
   Theta1_sum(idx) = sum(Theta1(idx,2:end).^2);   
end


for jdx = 1:size(Theta2,1)
   Theta2_sum(jdx) = sum(Theta2(jdx,2:end).^2);   
end

nn_regu = lambda*(sum(Theta1_sum) + sum(Theta2_sum))/(2*m);
J = J + nn_regu;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

for t= 1:m
   
    a1_bp = [1; X(t,:)'] ;
   
    z2_bp = Theta1  *a1_bp ;
    a2_bp = [ones(size(z2_bp,2),1); sigmoid(z2_bp)];

    z3_bp =  Theta2*a2_bp ;
    a3_bp = sigmoid(z3_bp);

    delta_3 = (a3_bp - y_vec(t,:)');
    
    delta_2 = Theta2'*delta_3 .* [1;sigmoidGradient(z2_bp)]; 
    delta_2 = delta_2(2:end);

    % Big delta 
	Theta1_grad = Theta1_grad + delta_2 * a1_bp';
	Theta2_grad = Theta2_grad + delta_3 * a2_bp';
end

% Regularizaed 
Theta1_grad = (Theta1_grad + lambda*[zeros(size(Theta1,1),1) Theta1(:,2:end)])/m;
Theta2_grad = (Theta2_grad + lambda*[zeros(size(Theta2,1),1) Theta2(:,2:end)])/m;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

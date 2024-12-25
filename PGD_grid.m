%%-------------------------------------------------------------------------
%This MATLAB script implements "Proximal Gradient Descent (PGD)" algorithm 
%along with grid search for lambda values. 
%Inputs:
%  y - Time-series sample
%  a - AR coefficients
%  b - AR coefficients
%  c - AR coefficients
%Output:
%  Optimal lambda values based on grid interval
%%-------------------------------------------------------------------------

clear
% Load data from .mat file
if exist('change_detection.mat', 'file')

    data = load('change_detection.mat');
    y = data.y;
    T = length(y);
    
    % Initialize variables
    a = zeros(T-3, 1); % Initialize with zeros
    b = zeros(T-3, 1);
    c = zeros(T-3, 1);
    
    % Hyperparameters
    eta = 1e-3; % Step size
    max_iter = 5000; % Maximum iterations
    tol = 1e-4; % Convergence tolerance
    
    % Grid search parameters
    lambda_values = 1:0.1:10; % Grid for lambda
    min_mse = []; % To store the minimum MSE
    lambda_a1 = [];
    lambda_b1 = [];
    lambda_c1 = [];
    
    % Iterate over grid of lambda values
    i = 1;
    for lambda = lambda_values
    	lambda_a = lambda;     % Individual sparsity for 'a'
        lambda_b = lambda;     % Individual sparsity for 'b'
        lambda_c = lambda;     % Individual sparsity for 'c'
        %Proximal Gradient Descent (PGD) algorithm
        prev_obj = Inf;
        for iter = 1:max_iter
            % Residual term
            residual = y(4:T) - a .* y(3:T-1) - b .* y(2:T-2) - c .* y(1:T-3);
        
            % Compute gradients for residual term
            grad_a = -2 * (residual .* y(3:T-1)); % Gradient for a from residuals
            grad_b = -2 * (residual .* y(2:T-2)); % Gradient for b from residuals
            grad_c = -2 * (residual .* y(1:T-3)); % Gradient for c from residuals
        
            % Add TV regularization gradients
            tv_grad_a = lambda_a * ([0; sign(diff(a))] - [sign(diff(a)); 0]);
            tv_grad_b = lambda_b * ([0; sign(diff(b))] - [sign(diff(b)); 0]);
            tv_grad_c = lambda_c * ([0; sign(diff(c))] - [sign(diff(c)); 0]);
        
            % Total gradient with TV regularization
            grad_a = grad_a + tv_grad_a;
            grad_b = grad_b + tv_grad_b;
            grad_c = grad_c + tv_grad_c;
        
            % Gradient descent update
            a = a - eta * grad_a;
            b = b - eta * grad_b;
            c = c - eta * grad_c;
        
            % Projection onto constraints
            a = max(-0.5, min(1, a));
            b = max(-0.8, min(-0.2, b));
            c = max(-0.2, min(0.2, c));
        
            % Compute objective value
            obj = sum((y(4:T) - a .* y(3:T-1) - b .* y(2:T-2) - c .* y(1:T-3)).^2) + ...
                  lambda_a * sum(abs(diff(a))) + ...
                  lambda_b * sum(abs(diff(b))) + ...
                  lambda_c * sum(abs(diff(c)));
        
            % Convergence check
            if abs(prev_obj - obj) <= tol
                disp(['Converged after ', num2str(iter), ' iterations.']);
                break;
            end
            prev_obj = obj;
        end
        % Compute MSE for current lambda
        min_mse(i) = mean((data.a - a).^2) + mean((data.b - b).^2) + mean((data.c - c).^2);
    	lambda_a1(i) = lambda_a;
    	lambda_b1(i) = lambda_b;
    	lambda_c1(i) = lambda_c;
    	i=i+1;
    end
    
    % Find the combination with the smallest MSE error
    [~, best_idx] = min(min_mse(:));
    
    % Display best lambda and corresponding MSE
    fprintf('Best Lambd_a: %.2f\n', lambda_a1(best_idx));
    fprintf('Best Lambd_b: %.2f\n', lambda_b1(best_idx));
    fprintf('Best Lambd_c: %.2f\n', lambda_c1(best_idx));
    fprintf('Minimum MSE: %.5f\n', min_mse(best_idx));

else
    disp('Error: change_detection.mat file does not exist in the current directory !!');
end

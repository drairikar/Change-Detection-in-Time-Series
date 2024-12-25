%%-------------------------------------------------------------------------
%This MATLAB script implements CVX solver solution for "Change Detection in 
%Time Series Model" along with grid search for lambda values.
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

    %Calculate number of non-zeros in a,b,c,y 
    % y(t+3) = a(t)y(t+2) + b(t)y(t+1)+ c(t)y(t)+ v(t)
    % It's a cardinality (l0-norm)-> not convex convert to convex optimization
     
    count_a = sum(data.a ~= 1) ;
    count_b = sum(data.b ~= -0.8);
    count_c = sum(data.b ~= -0.2);
    
    disp(['The number of elements not in 1: ', num2str(count_a)])
    disp(['The number of elements not in -0.8: ', num2str(count_b)])
    disp(['The number of elements not in -0.2: ', num2str(count_c)])
    
    true_a = data.a;
    true_b = data.b;
    true_c = data.c;
    y = data.y;
    
    T = length(y);
    
    % Grid search parameters
    lambda_values = 1:0.1:10; % Grid for lambda
    min_mse = []; % To store the minimum MSE
    lambda_a1 = [];
    lambda_b1 = [];
    lambda_c1 = [];
    
    % Iterate over grid of lambda values
    i = 1;
    tic; % Start timer
    for lambda = lambda_values
    	lambda_a = lambda;     % Individual sparsity for 'a'
        lambda_b = lambda;     % Individual sparsity for 'b'
        lambda_c = lambda;     % Individual sparsity for 'c'
        cvx_begin quiet
            variables a(T-3) b(T-3) c(T-3)
            
            % Prediction error
            pred_error = sum((data.y(4:end) - a.*data.y(3:end-1) - b.*data.y(2:end-2) - c.*data.y(1:end-3)).^2);
        
            % Total variation regularization
            tv_reg_a = lambda_a * sum(abs(a(2:end) - a(1:end-1)));
            tv_reg_b = lambda_b * sum(abs(b(2:end) - b(1:end-1)));
            tv_reg_c = lambda_c * sum(abs(c(2:end) - c(1:end-1)));
        	total_var = tv_reg_a + tv_reg_b + tv_reg_c;
        
            % Objective function
            minimize(pred_error + total_var)
        
            subject to  
                a >= -0.5;
                a <= 1;
                b >= -0.8;
                b <= -0.2;
                c >= -0.2;
                c <= 0.2;
        
        cvx_end
    
        % Compute MSE for current lambda
        min_mse(i) = mean((true_a - a).^2) + mean((true_b - b).^2) + mean((true_c - c).^2);
    	lambda_a1(i) = lambda_a;
    	lambda_b1(i) = lambda_b;
    	lambda_c1(i) = lambda_c;
    	i=i+1;
    end
    time_elapsed = toc; % End timer
    
    % Find the combination with the smallest MSE error
    [~, best_idx] = min(min_mse(:));
    
    % Display best lambda and corresponding MSE
    fprintf('Best Lambd_a: %.2f\n', lambda_a1(best_idx));
    fprintf('Best Lambd_b: %.2f\n', lambda_b1(best_idx));
    fprintf('Best Lambd_c: %.2f\n', lambda_c1(best_idx));
    fprintf('Minimum MSE: %.5f\n', min_mse(best_idx));
    
    disp(['CPU time: ', num2str(time_elapsed), ' seconds']);
    disp(['Number of iterations: ', num2str(cvx_slvitr)]);
    disp(['Solver used: ', cvx_solver]);
    
    fprintf('Optimization completed with status: %s\n', cvx_status)

else
    disp('Error: change_detection.mat file does not exist in the current directory !!');
end
%%-------------------------------------------------------------------------
%This MATLAB script implements CVX solver solution for "Change Detection in 
%Time Series Model"
%Inputs:
%  y - Time-series sample
%  a - AR coefficients
%  b - AR coefficients
%  c - AR coefficients
%Output:
%  Estimate of the AR coefficients
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
    
    % Lambdas for individual coefficients
    lambda_a = 4.3;
    lambda_b = 4.3;
    lambda_c = 4.3;
    
    tic; % Start timer
    cvx_begin %quiet
        variables a(T-3) b(T-3) c(T-3)
        
        % Prediction error
        pred_error = sum_square(y(4:T) - a .* y(3:T-1)- b .* y(2:T-2) - c .* y(1:T-3));
    
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
    time_elapsed = toc; % End timer
    
    disp(['CPU time: ', num2str(time_elapsed), ' seconds']);
    disp(['Number of iterations: ', num2str(cvx_slvitr)]);
    disp(['Solver used: ', cvx_solver]);
    
    fprintf('Optimization completed with status: %s\n', cvx_status)
    
    % Predicted y
    y_pred = a .* y(3:end-1) + b .* y(2:end-2) + c .* y(1:end-3);
    
    % Compute Mean Squared Error (MSE)
    mse_y = mean((y(4:end) - y_pred).^2); % MSE between true and predicted y
    
    % Compute MSE for coefficients (true vs estimated a, b, c)
    mse_a = mean((true_a - a).^2);
    mse_b = mean((true_b - b).^2);
    mse_c = mean((true_c - c).^2);
    
    % Display MSE results
    fprintf('MSE between true and predicted y: %.5f\n', mse_y);
    fprintf('MSE for coefficient a: %.5f\n', mse_a);
    fprintf('MSE for coefficient b: %.5f\n', mse_b);
    fprintf('MSE for coefficient c: %.5f\n', mse_c);
    
    % Plot coefficient and error
    figure;
    subplot(311)
    plot(true_a, 'b', 'DisplayName','True a(t)')
    hold on;
    plot([a; nan(3,1)], 'r', 'DisplayName','Estimated a(t)')
    title('Coefficient a(t)')
    ylabel('Amplitude');
    xlabel('Number of AR coefficients samples');
    ylim([-2, 2])
    grid on;
    legend;
    
    subplot(312)
    plot(true_b, 'b', 'DisplayName','True b(t)')
    hold on;
    plot([b; nan(3,1)], 'r', 'DisplayName','Estimated b(t)')
    ylabel('Amplitude');
    xlabel('Number of AR coefficients samples');
    title('Coefficient b(t)')
    ylim([-1 0.5])
    grid on;
    legend;
    
    subplot(313)
    plot(true_c, 'b', 'DisplayName','True c(t)')
    hold on;
    plot([c; nan(3,1)], 'r', 'DisplayName','Estimated c(t)')
    title('Coefficient c(t)')
    ylabel('Amplitude');
    xlabel('Number of AR coefficients samples');
    ylim([-1 1])
    grid on;
    legend;
    
    figure;
    subplot(311)
    plot((true_a - a), 'b', 'DisplayName','Error : a(t)')
    title('Error values for a(t) (true_a - Estimated a(t))')
    ylabel('Amplitude');
    xlabel('Number of AR coefficients samples');
    ylim([-2, 2])
    grid on;
    legend;
    
    subplot(312)
    plot((true_b - b), 'b', 'DisplayName','Error : b(t)')
    title('Error values for b(t) (true_b - Estimated b(t))')
    ylabel('Amplitude');
    xlabel('Number of AR coefficients samples');
    ylim([-1 0.5])
    grid on;
    legend;
    
    subplot(313)
    plot((true_c - c), 'b', 'DisplayName','Error : c(t)')
    title('Error values for c(t) (true_c - Estimated c(t))')
    ylabel('Amplitude');
    xlabel('Number of AR coefficients samples');
    ylim([-1 1])
    grid on;
    legend;

else
    disp('Error: change_detection.mat file does not exist in the current directory !!');
end

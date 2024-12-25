%%-------------------------------------------------------------------------
%This MATLAB script implements "Proximal Gradient Descent (PGD)" algorithm
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
    y = data.y;
    T = length(y);
    
    % Initialize variables
    a = zeros(T-3, 1); % Initialize with zeros
    b = zeros(T-3, 1);
    c = zeros(T-3, 1);
    
    lambda_a = 3.4; % Regularization weight for TV of a
    lambda_b = 3.4; % Regularization weight for TV of b
    lambda_c = 3.4; % Regularization weight for TV of c
    
    % Hyper parameters
    eta = 1e-3; % Step size
    max_iter = 25000; % Maximum iterations
    tol = 1e-4; % Convergence tolerance
    
    obj_values = []; % To store the obj values
    
    % Proximal Gradient Descent (PGD) algorithm
    prev_obj = Inf;
    tic; % Start timer
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
        obj_values(iter) = sum((y(4:T) - a .* y(3:T-1) - b .* y(2:T-2) - c .* y(1:T-3)).^2) + ...
              lambda_a * sum(abs(diff(a))) + ...
              lambda_b * sum(abs(diff(b))) + ...
              lambda_c * sum(abs(diff(c)));
    
        % Convergence check
        if abs(prev_obj - obj_values(iter)) <= tol
            disp(['Converged after ', num2str(iter), ' iterations.']);
            break;
        end
        prev_obj = obj_values(iter);
    end
    time_elapsed = toc; % End timer
    disp(['CPU time: ', num2str(time_elapsed), ' seconds']);
        
    % Predicted y
    y_pred = a .* y(3:end-1) + b .* y(2:end-2) + c .* y(1:end-3);
    
    % Compute Mean Squared Error (MSE)
    mse_y = mean((y(4:end) - y_pred).^2); % MSE between true and predicted y
    
    % Compute MSE for coefficients (true vs estimated a, b, c)
    mse_a = mean((data.a - a).^2);
    mse_b = mean((data.b - b).^2);
    mse_c = mean((data.c - c).^2);
    
    % Display Results
    fprintf('MSE between true and predicted y: %.5f\n', mse_y);
    fprintf('MSE for coefficient a: %.5f\n', mse_a);
    fprintf('MSE for coefficient b: %.5f\n', mse_b);
    fprintf('MSE for coefficient c: %.5f\n', mse_c);
    
    fprintf('Optimal value (cvx_optval): %.5f\n', obj_values(iter));
    
    fprintf('Number of iteration executed : %d\n', iter);
    
    % Output results
    figure;
    plot(obj_values, 'g', 'DisplayName','Obj values')
    title('Convergence of the algorithm')
    ylabel('Obj values');
    xlabel('Number of iterations');
    grid on;
    legend;
    
    % Output results
    figure;
    subplot(311)
    plot(data.a, 'b', 'DisplayName','True a(t)')
    hold on;
    plot([a; nan(3,1)], 'r', 'DisplayName','Estimated a(t)')
    title('Coefficient a(t)')
    ylabel('Amplitude');
    xlabel('Number of AR coefficients samples');
    grid on;
    ylim([-2, 2])
    legend;
    
    subplot(312)
    plot(data.b, 'b', 'DisplayName','True b(t)')
    hold on;
    plot([b; nan(3,1)], 'r', 'DisplayName','Estimated b(t)')
    title('Coefficient b(t)')
    ylabel('Amplitude');
    xlabel('Number of AR coefficients samples');
    grid on;
    ylim([-1 0.5])
    legend;
    
    subplot(313)
    plot(data.c, 'b', 'DisplayName','True c(t)')
    hold on;
    plot([c; nan(3,1)], 'r', 'DisplayName','Estimated c(t)')
    title('Coefficient c(t)')
    ylabel('Amplitude');
    xlabel('Number of AR coefficients samples');
    grid on;
    ylim([-1 1])
    legend;
    
    figure;
    subplot(311)
    plot((data.a - a), 'b', 'DisplayName','Error : a(t)')
    title('Error values for a(t) (true_a - Estimated a(t))')
    ylabel('Amplitude');
    xlabel('Number of AR coefficients samples');
    ylim([-2, 2])
    grid on;
    legend;
    
    subplot(312)
    plot((data.b - b), 'b', 'DisplayName','Error : b(t)')
    title('Error values for b(t) (true_b - Estimated b(t))')
    ylabel('Amplitude');
    xlabel('Number of AR coefficients samples');
    ylim([-1 0.5])
    grid on;
    legend;
    
    subplot(313)
    plot((data.c - c), 'b', 'DisplayName','Error : c(t)')
    title('Error values for c(t) (true_c - Estimated c(t))')
    ylabel('Amplitude');
    xlabel('Number of AR coefficients samples');
    ylim([-1 1])
    grid on;
    legend;

else
    disp('Error: change_detection.mat file does not exist in the current directory !!');
end


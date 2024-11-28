[d_star, Fs_star] = audioread("sounds/StarWars3.wav");
[d_noise, Fs_noise] = audioread("sounds/Noise.wav");

function [w_updated, errors] = minimize(d, x, C, mu, max_iter)
    % Inputs:
    % d: Composite signal (desired signal)
    % x: Input signal (noise reference)
    % C: Constraint on the weight vector magnitude
    % mu: Step size for gradient descent
    % max_iter: Maximum number of iterations

    M = 50; % Number of filter weights
    w = zeros(M, 1); % Initialize weight vector
    N = length(d); % Length of the composite signal
    errors = 1:max_iter;

    hWaitbar = waitbar(0, 'Processing...');
    for iter = 1:max_iter
        waitbar(iter / max_iter, hWaitbar, sprintf('Processing... Iteration %d of %d', iter, max_iter));
        for n = M:N
            y_n = w' * x(n:-1:n-M+1);

            e_n = d(n) - y_n;
            
            % Gradient of the objective function
            grad_L = -2 * e_n * x(n:-1:n-M+1); 
            
            % Update weights using gradient descent step

            w_new = w - mu * grad_L;

            if norm(w_new) > C
                w_new = C * (w_new / norm(w_new)); % Projection onto feasible set
            end
            w = w_new;
        end
        errors(iter) = norm(e_n);
    end
    
    w_updated = w; % Return updated weights after optimization
    close(hWaitbar);
end


function noise_cancelled_signal = create_noise_cancelled_signal(d, x, w)
    % Inputs:
    % d: Composite signal (desired signal)
    % x: Input signal (noise reference)
    % w: Optimized weight vector (M x 1)

    M = length(w); % Number of filter weights
    N = length(d); % Length of the composite signal
    noise_cancelled_signal = zeros(N, 1); % Initialize output signal

    for n = M:N
        % Convolution with weights
        noise_cancelled_signal(n) = d(n) - w' * x(n:-1:n-M+1);
    end
end


for alpha = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    [w, error] = minimize(d_star, d_noise, 0.9, alpha, 2);
    d_star_new = create_noise_cancelled_signal(d_star, d_noise, w);
    disp(error);
    figure;
    plot(error);
end
% soundsc(d_star, Fs_star);
% pause(4)
% soundsc(d_star_new, Fs_noise);



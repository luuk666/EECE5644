c = 3; 
% lambdas
lambda_r = 0.1;
lambda_s = 0.5;

P_w_given_x = rand(1, c);
P_w_given_x = P_w_given_x / sum(P_w_given_x);

decision_threshold = 1 - lambda_r / lambda_s;

min_risk = Inf;
chosen_action = -1;

for i = 1:(c + 1)
    if i <= c
        risk = lambda_s * (1 - P_w_given_x(i));
    else
        risk = lambda_r;
    end
    
    if risk < min_risk
        min_risk = risk;
        chosen_action = i;
    end
end

% Display
fprintf('Posterior Probabilities: [');
fprintf('%.2f, ', P_w_given_x(1:end-1));
fprintf('%.2f]\n', P_w_given_x(end));
fprintf('Lambda r: %.1f, Lambda s: %.1f\n', lambda_r, lambda_s);

if chosen_action <= c
    fprintf('Minimum Risk Decision: Classify as w%d\n', chosen_action);
else
    fprintf('Minimum Risk Decision: Reject\n');
end

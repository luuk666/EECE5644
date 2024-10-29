
K = 3;

N = 100;

D = randi(K, N, 1);

theta_hat = zeros(K, 1);

for k = 1:K
    theta_hat(k) = sum(D == k) / N;
end

disp('ML estimator for Θ:');
disp(theta_hat);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha = ones(K, 1);

theta_map = zeros(K, 1);

for k = 1:K
    theta_map(k) = (alpha(k) - 1 + sum(D == k))^(-1);
end

disp('MAP estimator for Θ:');
str=join(string(alpha),', ');
fprintf('With Dirichlet α: %s\n', str);
disp(theta_map);

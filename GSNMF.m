function [U, V, loss] = GSNMF(X, R, k, lambda, niter)
% Inputs:
%   X: User opinion matrix
%   R: Social interaction matrix
%   k: The number of communities
%   lambda: The regularization parameter controlling the contribution of the
%   graph regularizer
%   niter: The number of iterations
% Return:
%   U: Community membership matrix
%   V: Community profile matrix
%   loss: The loss

[m, n] = size(X);
U = rand(n,k);
V = rand(m,k);
D = diag(full(sum(R,2)));
D_ = D.^(-0.5);
D_(isinf(D_)) = 0;
Z = D_*R*D_;
minVal = 1e-1000;

for it = 1:niter
    V = X * U * inv(U'*U);
    V(isnan(V))=0;
    [XTV_pos, XTV_neg] = PosNegSeperation(X'*V);
    [VTV_pos, VTV_neg] = PosNegSeperation(V'*V);
    [lag_pos, lag_neg] = PosNegSeperation(U'*X'*V - V'*V  + lambda*U'*(Z)*U);
    U = U.*(((XTV_pos + U*VTV_neg + lambda*Z*U + U*lag_neg)./max((XTV_neg + U*VTV_pos + U*lag_pos),minVal)).^(0.5)); 
    U(isnan(U))=0;
end
loss = norm(X - V*U', 'fro')^2 + lambda*trace(U'*Z*U);
end


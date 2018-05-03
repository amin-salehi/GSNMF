%   Licensed under the Apache License, Version 2.0 (the "License");
%   you may not use this file except in compliance with the License.
%   You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
%   Unless required by applicable law or agreed to in writing, software
%   distributed under the License is distributed on an "AS IS" BASIS,
%   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%   See the License for the specific language governing permissions and
%   limitations under the License.

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
    [XTV_pos, XTV_neg] = PosNegSeparation(X'*V);
    [VTV_pos, VTV_neg] = PosNegSeparation(V'*V);
    [lag_pos, lag_neg] = PosNegSeparation(U'*X'*V - V'*V  + lambda*U'*(Z)*U);
    U = U.*(((XTV_pos + U*VTV_neg + lambda*Z*U + U*lag_neg)./max((XTV_neg + U*VTV_pos + U*lag_pos),minVal)).^(0.5)); 
    U(isnan(U))=0;
end
loss = norm(X - V*U', 'fro')^2 + lambda*trace(U'*Z*U);

function [ A_pos, A_neg ] = PosNegSeparation( A )
    A_abs = abs(A);
    A_pos = (A_abs + A) / 2;
    A_neg = (A_abs - A) / 2;
end

end


function [tA] = cyclecount(A,kmax)
%NETPROF Calculate network profile
% [t,m] = netprof(e,n,s,N,kmax)
% Inputs:
%   e : edge list of a simple graph (no self-loops or multiple edges)
%       (Assumed to contain no repeated edges. Ex: Where A is binary and
%       symmetric with zero diagonal, [i,j] = find(triu(A,1)); e = [i j];)
%   n : number of vertices in the simple graph represented by edge list e
%   s : subsample size (number of vertices) for profile calculation
%   N : number of replicates for profile calculation
%   kmax : maximum scale for profile calculation (should be small relative
%          to n, as the theory requires kmax fixed as n goes to infinity)
%
% Outputs:
%   t : matrix of each profile scale, from k = 1 to k = kmax (computed from backtracking matrix)
%   tA : matrix of each profile scale, from k = 1 to k = kmax (computed from adjacency matrix)
%   m : vector containing the number of edges in each profile sample
%
% Example:
%   e = load('celegans_metabolic.txt'); % load edgelist of interest
%
%   e = unique(e,'rows'); % remove multiple edges
%   e = e(find(e(:,1)~=e(:,2)),:); % remove self-loops
%   e = e - min(1,min(e(:))) + 1; % Ensure vertex numbering does not start below unity
%   d = accumarray(e(:),1); % calculate degrees from edge list
%   n = length(d); % number of vertices
%   assert(n==max(e(:)));
%   A = sparse([e(:,1); e(:,2)],[e(:,2); e(:,1)],1,n,n);
%   A = min(A,1);
%   assert(all([norm(diag(A),2) norm(A-A','fro') min(A(:)) max(A(:)) range(A(:))]==[0 0 0 1 1]));
%   %[i,j] = find(triu(A,1)); e = [i j];
%
%   s = 25; % subsample size
%   N = 10; % number of replicates
%   kmax = 10;
%
%   [t,tA,m] = netprof(e,n,s,N,kmax);

n = size(A,1);
s = n;
N = 1;

assert((mod(kmax,1)==0)&&(kmax>=2),'Input kmax must be an integer >= 2')

assert(norm(diag(A),2)==0,'Graph represented by input e cannot have self-loops');
assert(norm(A-A','fro')==0,'Graph represented by input e must be symmetric');
assert(norm((A.*A)-A,'fro')==0,'Graph represented by input e must contain only 0s and 1s');

[t,tA] = deal(zeros(N,kmax));
m = zeros(N,1);
for r = 1:N
    u = randsample(n,s); % sample s nodes without replacement
    Au = A(u,u); % vertex-induced subgraph of A (induced by vertex subset u)
    d = full(sum(Au,2));
    m(r) = sum(d)/2;
    if any(d>1), % necc & suff condition for >= 1 non-backtracking edge
        [i,j] = find(triu(Au,1)); % need consecutive edge numbering
        [c,cA] = deal(zeros(1,kmax));
        c(2) = d'*(d-1)/sum(d);
        cA(2) = 0; %2*sum(d); % wrong scale somehow - no difference to take
        unz = find(d>0); % find non-zero-degree nodes in Au
        A1 = Au(unz,unz);
        [A3, A4, A5] = deal(zeros(size(A1)));
        [dA2, dA3, dA4, dA5, dA6, dA7] = deal(zeros(size(d)));
        [sA3, tA3, tA4, tA5, tA6, tA7] = deal(0);
        [XC3, XH5, XH6, XH11, XH12, XH13, XH15, XK4] = deal(0);
            A2 = A1*A1;
            dA2 = diag(A2);
            Ak = full(A2);
            for k = 3:kmax
                Ak = Ak*A1;
                if k > 9
                    %cA(k) = (sum(diag(Ak))-c(k))^(1/k);
                elseif k==3
                    sA3 = sum(sum(Ak));
                    dA3 = diag(Ak);
                    tA3 = sum(dA3);
                    %cA(k) = (tA3-c(k))^(1/k);
                    cA(k) = (tA3)^(1/k);
                    A3 = Ak;                    
                elseif k==4
                    dA4 = diag(Ak);
                    tA4 = sum(dA4);
                    %cA(k) = (tA4+2*m(r)-2*(d(unz)'*d(unz))-c(k))^(1/k);
                    cA(k) = (tA4+2*m(r)-2*(d(unz)'*d(unz)))^(1/k);
                    A4 = Ak;
                elseif k==5
                    dA5 = diag(Ak);
                    tA5 = sum(dA5);
                    %cA(k) = (tA5-5*sum((d(unz)-1).*dA3)-c(k))^(1/k);
                    cA(k) = (tA5-5*sum((d(unz)-1).*dA3))^(1/k);
                    A5 = Ak;
                elseif k==6
                    dA6 = diag(Ak);
                    tA6 = sum(diag(Ak));
                    XC3 = tA3/6;
                    XH6 = (1/2)*(1/2)*sum(sum(((A2).^2-A2).*A1));
                    XH11 = (1/2)*sum((dA3./2).^2-(dA3./2)) - 2*XH6;
                    %c(k) = c(k) -3*sum(dA3.^2)+9*sum(sum((A2.^2).*A1))-4*sum(sum(A2.*A1)) %why not -9 rather than -4?)
                    %c(k) = c(k) - 6*tA3 + 36*(1/4)*sum(sum(A2.*(A2-1).*A1)) - 24*(1/2)*sum((dA3./2).*(dA3./2-1));
                    cA(k) = (tA6-3*sum(dA3.^2)+9*sum(sum((A2.^2).*A1))-6*sum(dA4.*(d(unz)-1))-4*sum(dA3-d(unz).^3)+3*sA3-12*sum(d(unz).^2)+4*sum(d(unz)))^(1/k);
                    %full([ cA(k)^k (c(k) - 24*XH11 - 6*XC3 - 12*XH6) ])
                elseif k==7
                    dA7 = diag(Ak);
                    tA7 = sum(dA7);
                    XH5 = (1/2)*sum(dA3.*(d(unz)-2));
                    XH12 = (1/2)*sum(sum(A3.*A2.*A1)) - 9*XC3 - 2*XH5 - 4*XH6;
                    XH13 = (1/2)*(1/6)*sum(sum(((A2-2).*(A2-1).*A2).*A1));
                    XH15 = (1/2)*(1/2)*sum(dA3.*(sum(A2.*(A2-1),2)-diag(A2.*(A2-1)))) - 6*XH6 - 2*XH12 -6*XH13;
                    %c(k) = c(k) - 28*sum(sum((A2.^2).*A1)) + 21*sum(sum(A3.*A2.*A1)) + 56*tA3 - 77*sum(dA3.*d(unz)) + 7*sum(sum((A2.^3).*A1)) + 7*sum(dA3.*sum(A2,2));
                    cA(k) = (tA7-7*sum(dA3.*dA4)+7*sum(sum((A2.^3).*A1))-7*sum(dA5.*d(unz))+21*sum(sum(A3.*A2.*A1))+7*tA5-28*sum(sum((A2.^2).*A1))+7*sum(sum(A2.*A1.*(d(unz)*d(unz)')))+14*sum(dA3.*(d(unz).^2))+7*sum(dA3.*sum(A2,2))-77*sum(dA3.*d(unz))+56*tA3)^(1/k);
                    %full([ cA(k)^k (c(k) - 28*XH15 - 14*XH12 - 28*XH6 - 84*XH13 ) ])
                elseif k==8
                    cA(k) = sum(diag(Ak)) - 4*sum(dA4.*dA4) - 8*sum(dA3.*dA5) - 8*sum(dA2.*dA6) + 16*sum(dA2.*dA3.*dA3) + 8*sum(dA6) + 16*sum(dA4.*dA2.*dA2) - 72*sum(dA3.*dA3) - 96*sum(dA4.*dA2) - 12*sum(dA2.*dA2.*dA2.*dA2) + 64*sum(dA2.*dA3) + 73*sum(dA4) + 72*sum(dA2.*dA2.*dA2) - 112*sum(dA3) + 36*sum(dA2);
                    cA(k) = cA(k) + 2*sum(sum(A2.*A2.*A2.*A2)) + 24*sum(sum(A2.*A2.*A1.*A3)) + 4*sum(sum((dA3*dA3').*A1)) + 16*sum(sum((dA2*dA3').*A1.*A2)) + 12*sum(sum(A1.*A3.*A3)) + 24*sum(sum(A1.*A4.*A2)) + 4*sum(sum((dA2*dA2').*A2.*A2)) + 8*sum(sum(A2*diag(dA4))) + 8*sum(sum((dA2*dA2').*A1.*A3)) - 16*sum(sum(A2.*A2.*A2)) - 32*sum(sum(A1.*A2.*A3)) - 96*sum(sum(diag(dA2)*(A2.*A2.*A1))) - 4*sum(sum(A4)) - 16*sum(sum(diag(dA2.*dA2)*A2)) + 272*sum(sum(A2.*A2.*A1)) + 48*sum(sum(A3)) - 132*sum(sum(A2));
                    cA(k) = cA(k) - 64*sum(sum(A1.*((A1.*A2)^2))) - 24*sum(sum(A1.*(A1*diag(dA2)*A1).*A2));
                    for i1 = 1:size(A1,1)
                        XK4 = XK4 + A1(i1,:)*(A1.*(A1*diag(A1(i1,:))*A1))*A1(i1,:)';
                    end
                    cA(k) = cA(k) + 22*XK4;
                    cA(k) = cA(k)^(1/k);
                elseif k==9
                    cA(k) = sum(diag(Ak)) - 9*sum(dA4.*dA5) - 9*sum(dA3.*dA6) - 9*sum(dA2.*dA7) + 6*sum(dA3.*dA3.*dA3) + 36*sum(dA4.*dA3.*dA2) + 9*sum(dA7) + 18*sum(dA2.*dA2.*dA5) - 171*sum(dA4.*dA3) - 117*sum(dA2.*dA5) - 54*sum(dA3.*dA2.*dA2.*dA2) + 72*sum(dA3.*dA3) + 81*sum(dA5) + 504*sum(dA3.*dA2.*dA2) - 1746*sum(dA3.*dA2) + 1148*sum(dA3);
                    cA(k) = cA(k) + 9*sum(sum(A2.*A2.*A2.*A3)) + 9*sum(sum((dA3*dA3').*A1.*A2)) + 27*sum(sum(A1.*A3.*A3.*A2)) + 27*sum(sum(A2.*A2.*A1.*A4)) + 9*sum(sum((dA3*dA4').*A1)) + 9*sum(sum((dA2*dA3').*A2.*A2)) + 18*sum(sum((dA2*dA4').*A1.*A2)) + 18*sum(sum((dA2*dA3').*A1.*A3)) + 27*sum(sum(A1.*A4.*A3)) + 27*sum(sum(A1.*A5.*A2)) + 9*sum(sum((dA2*dA2').*A2.*A3)) + 9*sum(sum(A2*diag(dA5))) + 9*sum(sum((dA2*dA2').*A4.*A1)) - 72*sum(sum(diag(dA2)*(A2.*A2.*A2.*A1))) - 108*sum(sum(diag(dA3)*(A2.*A2.*A1))) - 36*sum(sum(A2.*A2.*A3)) - 36*sum(sum(A4.*A1.*A2)) - 216*sum(sum(diag(dA2)*(A1.*A3.*A2))) - 9*sum(sum(A3*diag(dA3))) - 36*sum(sum(diag(dA3.*dA2)*A2)) - 18*sum(sum(((dA2.*dA2)*dA3').*A1)) - 36*sum(sum(((dA2.*dA2)*dA2').*A1.*A2)) + 336*sum(sum(A1.*A2.*A2.*A2)) + 288*sum(sum(diag(dA2)*(A1.*A2.*A2))) + 684*sum(sum(A1.*A3.*A2)) + 171*sum(sum(A2*diag(dA3))) + 252*sum(sum((dA2*dA2').*A1.*A2)) - 1296*sum(sum(A1.*A2.*A2));
                    cA(k) = cA(k) - 48*sum(sum(A1.*A2.*((A1.*A2)^2))) - 27*sum(sum((diag(dA2)*A1).*(A1*(A1.*A2.*A2)))) - 72*sum(sum(A1.*((A1.*A2)*(A2.*A2)))) - 27*sum(sum(A1.*A2.*(A1*diag(dA3)*A1))) - 144*sum(sum(A1.*((A1.*A3)*(A1.*A2)))) - 27*sum(sum(A1.*A3.*(A1*diag(dA2)*A1))) - 54*sum(sum(A1.*A2.*(A2*diag(dA2)*A1))) - 18*sum(sum((diag(dA2)*(A1.*A2)).*repmat(sum(A2,1),size(A2,1),1))) - 3*sum(sum((dA2*dA2').*(A1.*(A1*diag(dA2)*A1)))) + 324*sum(sum(A1.*A2.*((A1.*A2)*A1))) + 180*sum(sum((diag(dA2)*A1).*(A1*(A1.*A2))));
                    for i1 = 1:size(A1,1)
                        cA(k) = cA(k) + 99*( A1(i1,:)*(A1.*(A1*diag(A2(i1,:))*A1))*A1(i1,:)' );
                        cA(k) = cA(k) + 99*( A1(i1,:)*(A1.*(A1*diag(A1(i1,:))*(A1.*A2)))*A1(i1,:)' );
                    end
                    cA(k) = cA(k) - 156*XK4;
                    cA(k) = cA(k)^(1/k);
                end
            end
         
        t(r,:) = c./s;
        tA(r,:) = cA./s;
    end
end
end % of function netprof()


function [b,e] = nbmat(eu)
%NBMAT Calculate non-backtracking matrix
% [b,e] = nbmat(eu);
% Inputs:
%   eu: edge list of a simple graph (no self-loops or multiple edges)
%       Edge list vertex numbering is implicitly assumed to start at 1
%       Example : Where A is binary and symmetric with zero diagonal, 
%                 [i,j] = find(triu(A,1)); eu = [i j];
% Outputs:
%   b : non-backtracking edge list, indexing elements of e
%   e : lexicographically sorted list of the oriented edges of eu

% Calculate degrees and lexicographically sort oriented edges
d = accumarray(eu(:),1); % calculate degrees from edge list
e = sortrows([eu; eu(:,2) eu(:,1)]); % lexicographic sort oriented edges

% Find non-backtracking edges (e(i,1),j)-->(j,(c,2)), e(c,2)~=e(i,1)
dind = [0; cumsum(d)]; % indexes degrees in order of vertices
b = zeros(d'*(d-1),2); % Intialize to store non-backtracking edges
bi = 0; % indexes rows of b (to later fill B)
for i = 1:size(e,1) % loop over all edge origins
    j = e(i,2);
    c = dind(j) + (1:d(j))'; % indexes degrees of j
    cnb = c(e(c,2)~=e(i,1)); % get non-backtracking indices
    k = d(j) - 1; % number of non-backtracking paths
    b((bi+1):(bi+k),:) = [repmat(i,k,1) cnb];
    bi = bi + k;
end

end % of function nbmat()








% Copyright (c) 2015, Holger Hoffmann
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%
% modified by charles dufour for better python integration in 2021
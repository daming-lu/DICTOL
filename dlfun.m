function dls = dlfun(dls, X, verbose)
% dlfun           A general Dictionary Learning function which do one
% complete interation through the set of training vectors, X.
% The implementation is done entirely in Matlab, except for sparse
% approximation which may be done by the java mpv2 class or the SPAMS mex
% files, sparseapprox can also use only Matlab code, see help for it.
% Examples for how to use the java mpv2 class are in ex111.m.
% This function is NOT optimized for speed.
%
%   Use of function:
%   ----------------
%   dls = dlfun(dls, X, verbose)
%     X is data, a matrix of size NxL
%     dls is a Dictionary Learning Structure, the following fields may be
%     .D      dictionary NxK, must be given and will be updated
%     .Met    method is RLS, ILS (or MOD), ODL or K-SVD
%     .lastError
%     .noErrors    total number of errors noticed
%     .vsMet  vector selection method, as in sparseapprox.m
%     .vsArg  vector selection arguments, a struct as in sparseapprox.m
%             used as: w = sparseapprox(x, D, vsMet, vsArg)
%     .A      sum( w*w' ),  for ODL and MOD/ILS, KxK
%     .B      sum( x*w' ),  for ODL and MOD/ILS, NxK
%     .C      inv( A ),     for RLS
%     .W      KxL sparse matrix
%     .ssx    1xL,  ||x||_2^2
%     .sse    1xL,  ||x-D*w||_2^2
%     .w0     1xL,  ||w||_0, always as returned from sparseapprox
%     .w1     1xL,  ||w||_1, always as returned from sparseapprox
%     .it     increment each time this function starts
%     .snr    an array, snr(it) = 10*log10(sum(ssx)/sum(sse))
%     .lam    lambda for RLS (and ODL)
%     .lamStep    When to use lam
%     .ndStep     When to normalized dictionary
%     verbose 0: only errors displayed, (default)
%             1: summary displayed
%----------------------------------------------------------------------
%    examples:  (using 7-12 sec per iteration)
% clear all;  load('dataXforAR1.mat');  [N,L] = size(X); K=50;
% D0 = dictnormalize(X(:,3000+(1:K))); nofIt = 10;
%
% dlsMOD = struct('D',D0, 'Met','MOD', 'vsMet','ORMP', 'vsArg',struct('tnz',4));
% dlsMOD.snr = zeros(1,nofIt);
% tic; for i=1:nofIt; dlsMOD = dlfun(dlsMOD, X, 1); end; toc;  
% 
% dlsKSVD = struct('D',D0, 'Met','K-SVD', 'vsMet','ORMP', 'vsArg',struct('tnz',4));
% dlsKSVD.snr = zeros(1,nofIt);
% tic; for i=1:nofIt; dlsKSVD = dlfun(dlsKSVD, X, 1); end; toc;
%
% dlsODL = struct('D',D0, 'Met','ODL', 'vsMet','ORMP', 'vsArg',struct('tnz',4));
% dlsODL.lamStep = 20; dlsODL.ndStep = 50;
% dlsODL.A = eye(K); dlsODL.B = dlsODL.D;
% dlsODL.snr = zeros(1,nofIt);          
% tic; for i=1:nofIt; dlsODL.lam = lambdafun(i, 'C', nofIt, 0.99, 1); dlsODL = dlfun(dlsODL, X, 1); end; toc;  
%
% dlsRLS = struct('D',D0, 'Met','RLS', 'vsMet','ORMP', 'vsArg',struct('tnz',4));
% dlsRLS.lamStep = 20; dlsRLS.ndStep = 50;
% dlsRLS.C = eye(K); 
% dlsRLS.snr = zeros(1,nofIt);          
% tic; for i=1:nofIt; dlsRLS.lam = lambdafun(i, 'C', nofIt, 0.99, 1); dlsRLS = dlfun(dlsRLS, X, 1); end; toc;  


%----------------------------------------------------------------------
% Copyright (c) 2010.  Karl Skretting.  All rights reserved.
% University of Stavanger (Stavanger University), Signal Processing Group
% Mail:  karl.skretting@uis.no   Homepage:  http://www.ux.his.no/~karlsk/
%
% HISTORY:  dd.mm.yyyy
% Ver. 1.0  29.03.2010  Made function
%           08.04.2010  Only ILS seems to work properly !! 
%                       (K-SVD not tested yet)
% Ver. 1.1  14.04.2010  Function seems to work ok
% Ver. 1.2  08.08.2011  Some minor changes
% Ver. 1.3  28.03.2012  The W matrix is now sparse.
%----------------------------------------------------------------------

% Note that for RLS (and ILS/MOD) faster methods using the Java package mpv2
% are available, but only using xxMP vector selection (sparse coding). Ex:
%   java_access;
%   jD0 = mpv2.SimpleMatrix(D0);
%   jDicLea  = mpv2.DictionaryLearning(jD0, 1);
%   jDicLea.setORMP(int32(4), 1e-6, 1e-6);  % LARS not available!
%   jDicLea.setLambda( 'C', 0.99, 1.0, 200*L );
%   tic; jDicLea.rlsdla( X(:), 250 ); t=toc;  % do 250 complete iterations
%   snrTab = jDicLea.getSnrTab();
%   figure(2); plot(snrTab);

[N,L] = size(X);
K = size(dls.D, 2);
if (nargin < 3); verbose = 0; end;
if verbose > 0
    dls.ssx = zeros(1,L);
    dls.sse = zeros(1,L);
    dls.w0 = zeros(1,L);
    dls.w1 = zeros(1,L);
end

% initialize
if (strcmpi(dls.Met, 'ILS') || strcmpi(dls.Met, 'MOD'))
    dls.A = zeros(K,K);
    dls.B = zeros(N,K);
end
if (strcmpi(dls.Met, 'ILS') || strcmpi(dls.Met, 'MOD') || strcmpi(dls.Met, 'K-SVD'))
    dls.W = sparse(K,L);
end
if isfield(dls,'it')
    dls.it = dls.it+1;
else
    dls.it = 1;
end
if ~isfield(dls,'noErrors')
    dls.noErrors = 0;
end

for i=1:L
    x = X(:,i);
    if isfield(dls,'ssx')  
        dls.ssx(i) = x'*x;  
    end
    %
    try
        w = sparseapprox(x, dls.D, dls.vsMet, dls.vsArg);
    catch ME
        % if (sum(isinf(w)) + sum(isnan(w))) > 0
        em = ['dlfun (i=',int2str(i),...
            ', it=',int2str(dls.it),', Met=',dls.Met,...
            '): ERROR in sparse approximation: ',ME.message];
        disp(em);
        dls.noErrors = dls.noErrors+1;
        w = zeros(K,1);  % set w to zero
        % 
        if isfield(dls,'C')
            em = checkC(dls.C, i, dls.it);
            if numel(em) > 0
                dls.noErrors = dls.noErrors+1;
                dls.C = 0.01*eye(K);
            end
        end
        em = checkD(dls.D, i, dls.it, dls.Met);
        if numel(em) > 0
            dls.noErrors = dls.noErrors+1;
            dls.lastError = em;
            dls.D = X(:,ceil(L*rand(1,K))) + 0.1*randn(N,K);
            dls.D = dls.D - ones(N,1)*mean(dls.D);
            dls.D = dictnormalize(dls.D);
            if isfield(dls,'C')
                dls.C = eye(K);
            end
            if isfield(dls,'B')
                dls.B = dls.D;
            end
            if isfield(dls,'A')
                dls.A = eye(K);
            end
        end
    end
    %
    if isfield(dls,'W');  dls.W(:,i) = w;  end;
    if isfield(dls,'w0');  dls.w0(i) = sum(w~=0);  end;
    if isfield(dls,'w1');  dls.w1(i) = sum(abs(w));  end;
    %
    if (strcmpi(dls.Met, 'ILS') ||strcmpi(dls.Met, 'MOD') || strcmpi(dls.Met, 'K-SVD'))
        if isfield(dls,'sse');
            r = x - dls.D*w;   % here r is only needed to get sse
            dls.sse(i) = r'*r;
        end
        continue;    % nothing more to do on this x for ILS or K-SVD
    end
    %
    % this part (end of loop for i=1:L) is only for ODL and RLS
    r = x - dls.D*w;
    if isfield(dls,'sse');  dls.sse(i) = r'*r;  end;
    %
    if strcmpi(dls.Met, 'RLS')
%         em = checkC(dls.C, i, dls.it);
%         if (numel(em) > 0)  % error
%             dls.noErrors = dls.noErrors+1;
%             dls.lastError = [em,' - RLS before lambda is used.'];
%             dls.w = w;
%             return;
%         end
        if isfield(dls,'lam')
            if (mod(i, dls.lamStep)==0)
                dls.C = (1/dls.lam) * dls.C;
            end
        end
        u =  dls.C * w;
        alpha = 1/(1+(w'*u));
        dls.D = dls.D + (alpha*r)*u';
        dls.C = dls.C - (alpha*u)*u';
    end
    if strcmpi(dls.Met, 'ODL')
        if isfield(dls,'lam')
            if (mod(i, dls.lamStep)==0)
                dls.A = dls.lam * dls.A;
                dls.B = dls.lam * dls.B;    % unchanged D = B*inv(A)
            end
        end
        dls.A = dls.A + w*w';
        dls.B = dls.B + x*w';
        % the simple matlab equation for one iteration, without normalization
        % dls.D = dls.D + (dls.B-dls.D*dls.A)*diag(1./diag(dls.A));
        % Now as given in the ODL paper (including normalization as in paper)
        for j=1:K
            uj = (1/dls.A(j,j))*(dls.B(:,j)-dls.D*dls.A(:,j)) + dls.D(:,j);
            normuj = sqrt(uj'*uj);
            if (normuj > 1) && ~isfield(dls,'ndStep') 
                % this normalizaion does not adjust A and B the way it
                % should, and what is done when 'ndStep' is used
                dls.D(:,j) = uj/normuj;
            else
                dls.D(:,j) = uj;
            end
        end
        % note that D = B*inv(A)  (if A is diagonal)
    end
    %
    % normalization is often much to do, so it is not done for all iterations
    if ( isfield(dls,'ndStep') && ((mod(i, dls.ndStep)==0) || (i==L)) )
        em = checkD(dls.D, i, dls.it, dls.Met);
        if (numel(em) > 0)  % error: 'reset' D and A, B, C
            dls.noErrors = dls.noErrors+1;
            dls.lastError = em;
            dls.D = X(:,ceil(rand(1,K)*L))+0.1*randn(N,K);
            dls.D = dls.D - ones(N,1)*mean(dls.D);
            dls.D = dictnormalize(dls.D);
            if strcmpi(dls.Met, 'ODL')
                dls.A = eye(K);
                dls.B = dls.D;
            end
            if strcmpi(dls.Met, 'RLS')
                dls.C = eye(K);
            end
        end
        ig = reshape(sqrt(sum(dls.D.*dls.D)),K,1);  % column vector, diag(inv(G))
        g = 1./ig; % column vector, diag(G)
        %
        % D = D*G;   where G = diag(G)
        dls.D = dls.D .* (ones(N,1)*g'); 
        % for k1 = 1:K
        %     dls.D(:,k1) = g(k1)*dls.D(:,k1);  
        % end
        if isfield(dls,'A')
            % A = inv(G)*A*inv(G);   where inv(G) = diag(ig)
            % dls.A = dls.A .* (ig*ig');     
            for k1 = 1:K
                for k2 = k1:K
                    temp = dls.A(k1,k2)*ig(k1)*ig(k2);
                    dls.A(k1,k2) = temp;
                    dls.A(k2,k1) = temp; % enforce symmetry
                end
            end
        end
        if isfield(dls,'B')
            % B = B*inv(G);   where inv(G) = diag(ig)
            dls.B = dls.B .* (ones(N,1)*ig');  
            % for k1 = 1:K
            %     dls.B(:,k1) = ig(k1)*dls.B(:,k1);
            % end
        end
        if isfield(dls,'C') && strcmpi(dls.Met, 'RLS')
            % C = G*C*G;   where G = diag(g)
            % dls.C = dls.C .* (g * g');        
            % statement above seems to be 'unstable' ???
            % ex: a=1-eps;b=1/7;c=1/13; disp(1e18*(c*a*b - c*b*a));
            % this may cause that (g*g') is not exactly symmetric
            for k1 = 1:K  
                for k2 = k1:K
                    temp = dls.C(k1,k2)*(g(k1)*g(k2));
                    dls.C(k1,k2) = temp;
                    dls.C(k2,k1) = temp;  % enforce symmetry
                end
            end
            em = checkC(dls.C, i, dls.it);
            if (numel(em) > 0)  % error: 'reset' D and C
                dls.noErrors = dls.noErrors+1;
                dls.lastError = em;
                dls.D = X(:,ceil(rand(1,K)*L))+0.1*randn(N,K);
                dls.D = dls.D - ones(N,1)*mean(dls.D);
                dls.D = dictnormalize(dls.D);
                dls.C = eye(K);
            end
        end
        if isfield(dls,'W')
            dls.W = diag(ig) * dls.W;  % W = inv(G)*W
            % note w1 is not updated!
        end
    end
    %
end

if isfield(dls,'it')
    if ( isfield(dls,'snr') && isfield(dls,'sse') && isfield(dls,'ssx') )
        dls.snr(dls.it) = 10*log10(sum(dls.ssx)/sum(dls.sse));
    end
end

if (verbose > 0)
    if (strcmpi(dls.Met, 'ILS') || strcmpi(dls.Met, 'MOD') || strcmpi(dls.Met, 'K-SVD'))
        disp(['dlfun: iteration ',int2str(dls.it),' (just before dictionary update.)']); 
    else
        disp(['dlfun: after iteration ',int2str(dls.it),...
            ' (iteration count number of training sets, not number of training vectors.)']); 
    end
    fprintf(' L = %i, ssx = %10.1f, sse = %10.1f, snr = %5.2f\n', ...
              L, sum(dls.ssx), sum(dls.sse), dls.snr(dls.it));
end

if (strcmpi(dls.Met, 'ILS') || strcmpi(dls.Met, 'MOD'))
    dls.A = full(dls.W * dls.W');
    dls.B = X * full(dls.W');
    dls.D = dls.B / dls.A;
    % normalize dictionary (but not A or B)
    igd = sqrt(sum(dls.D.*dls.D)); % row vector, diag(inv(G))
    gd = 1./igd; % row vector, diag(G)
    dls.D = dls.D .* (ones(N,1)*gd);  % nomalize D = D*G
end
if strcmpi(dls.Met, 'K-SVD')
    for k=1:K
        R = X - dls.D*full(dls.W);
        I = find(dls.W(k,:));
        Ri = R(:,I) + dls.D(:,k)*dls.W(k,I);
        [U,S,V] = svds(Ri,1,'L');
        dls.D(:,k) = U;
        dls.W(k,I) = S*V';
    end
end

return;   % dlfun


%----------------------------------------------------------------------
%
function em = checkC(C, i, it)
% check that C is symmetric and positive definite
em = '';
c = diag(C);
if (min(c) <= 0) %
    em = ['dlfun (i=',int2str(i),', it=',int2str(it),...
        '): ERROR (min(diag(C)) = ',num2str(min(c)),').'];
    disp(em);
end
%
H = C - C';
temp = sum(H(:).^2);
if temp > 0.0001
    em = ['dlfun (i=',int2str(i),', it=',int2str(it),...
        '): ERROR C ~= C'', (temp=',num2str(temp),')'];
    disp(em);
end
%
return

function em = checkD(D, i, it, met)
% check that D is real (not NaN)
em = '';
temp = sum(isnan(D));
if (sum(temp) > 0)
    em = ['dlfun (i=',int2str(i),...
        ', it=',int2str(it),', Met=',met,...
        '): ERROR ',int2str(sum(temp)),' entries in D are NaN.'];
    disp(em);
else
    temp = sum(isinf(D));
    if (sum(temp) > 0)
        em = ['dlfun (i=',int2str(i),...
            ', it=',int2str(it),', Met=',met,...
            '): ERROR ',int2str(sum(temp)),' entries in D are infinite.'];
        disp(em);
    else
        temp = sum(D.*D);
        if (max(temp) > 10)
            em = ['dlfun (i=',int2str(i),...
                ', it=',int2str(it),', Met=',met,...
                '): ERROR some atoms in D have large 2-norm, max is ',num2str(max(temp))];
            disp(em);
        elseif min(temp) < 0.1
            em = ['dlfun (i=',int2str(i),...
                ', it=',int2str(it),', Met=',met,...
                '): ERROR some atoms in D have small 2-norm, min is ',num2str(min(temp))];
            disp(em);
        else
            % D may be OK
        end
        if (max(temp)<1000) && (min(temp)>0.001) 
            em = '';   % only severe errors are handled by calling function
        end
    end
end

return


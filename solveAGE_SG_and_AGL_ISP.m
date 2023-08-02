function [S,beta_i] = solveAGE_SG_and_AGL_ISP(options)
%% Input
%%% k                           The numer of neighbors
%%%
%%% dist                        The distance matrix with n2 * n1
%%%
%%% G                           The semantic graph with n2 * n1
%%%
%%% F                           The penalty coefficient of low-rankness
%%%
%% Output
%%% S                           The similar matrix n2 * n1

    %% ===== init parameters =====
    if ~isfield(options,'G')
        options.G = -1;
    end
    if ~isfield(options,'slack')
        options.slack = 0;
    end
    F=options.F;
    dist=options.dist;
    G=options.semanticGraph;
    k=options.k;
    beta=options.beta;
    epsilon=1e-3;
    lambda=options.lambda;
    [n2,n1]=size(dist);
    if 0<k&&k<1
       k=min(max(1,floor(k*n1)),n1-1);
       fprintf('set k=%d\n',k);
    end
    % relax the semantic graph to achieve discriminative results if necessary.
    if options.slack==1
        G(G==0)=-1;
    end
    [d, idx] = sort(dist,2,'ascend');
    sortF=zeros(n2,n1);
    sortG=zeros(n2,n1);
    for i=1:n2
        sortG(i,:)=G(i,idx(i,:));
        sortF(i,:)=F(i,idx(i,:));
    end
    meanD = repmat(mean(d,2),[1,n1]);
    meanG = repmat(mean(sortG,2),[1,n1]);
    meanF = repmat(mean(sortF,2),[1,n1]);
    % DD: `B` in the paper
    % GG: `A` in the paper
    DD = beta*d+lambda*sortF-(beta*meanD+lambda*meanF);
    GG = sortG-meanG;
    % solve beta and gamma
    k=k*ones(n2,1);
    % Compute beta_i
    [beta_i,solvable,k]=computeBeta(DD,GG,k);
    S=zeros(n2,n1);
    for i=1:n2
        idxa=idx(i,1:k(i));
        if solvable(i)==1
            % Solve AGE-SG
           if beta_i(i)>0&&(abs(beta_i(i))>1e-6)
               tmp = beta_i(i) * GG(i,1:k(i)) - 0.5 * (DD(i,1:k(i))) + 1/n1;
               S(i,idxa)= EProjSimplex_new(tmp,1);
           else
               % Solve AGL-ISP
               idxx=idx(i,1:k(i));
               di=d(i,:);
               tmp=(di(k(i)+1)-di(1:k(i)))/(k(i)*di(k(i)+1)-sum(di(1:k(i)))+eps);
               S(i,idxx)=tmp;
           end
        else
            % Solve AGL-ISP
            idxx=idx(i,1:k(i));
            di=d(i,:);
            tmp=(di(k(i)+1)-di(1:k(i)))/(k(i)*di(k(i)+1)-sum(di(1:k(i)))+eps);
            S(i,idxx)=tmp;
        end
    end
    fixedAble=abs(beta_i)<epsilon;
    solvable=logical(solvable);
end

function [beta_i,solvable,k]=computeBeta(DD,GG,k)
    %% Input:
    %%%         DD: d - mean d
    %%%         GG: g - mean g
    %%%         gamma_i: gamma_i
    [n,nu]=size(DD);
    beta_i=zeros(n,1);
    solvable=ones(n,1);
    A = GG;
    B = DD - 2/nu;
    A1= A(:,1:end-1)>0;
    A2= A(:,2:end)>0;
   
    B1 =B(:,1:end-1)>0;
    B2 =B(:,2:end)>0;
    %%% Init k
    conditions=abs(B);
    [~,k2]=min(conditions,[],2);
    k2=min(k2+1,nu-1);
    % cond1
    cond1= A1&A2;
    cond2= A1&~A2;
    cond3= ~A1&A2;
    cond4= ~A1&~A2;
    cond21= B1&B2;
    cond22 = ~B1&~B2;
    for i = 1:n
         if k(i)==-1
             k(i)=k2(i);
         end
         C_left = B(i,k(i))/(2*A(i,k(i))+eps);
         C_right = B(i,k(i)+1)/(2*A(i,k(i)+1)+eps);
         if cond1(i,k(i))||cond4(i,k(i))
             % \beta_i = 0.5 (left+right)
             if cond21(i,k(i))||cond22(i,k(i))
                 if (cond1(i,k(i))&&C_left<=C_right)||(cond4(i,k(i))&&C_left>=C_right)
                     if abs(C_left)<abs(C_right)
                         beta_i(i)=C_left;
                     else
                         beta_i(i)=C_right;
                     end
                 else
                     % almost impossible
                     beta_i(i)=-1;
                     solvable(i) = 0;
                 end
             else
                beta_i(i)=0;
             end
         elseif cond2(i,k(i))
             % \beta_i > max (left, right)
             tmp = max(C_left,C_right);
             if tmp < 0
                 beta_i(i)=0;
             else
                 beta_i(i)=tmp;
             end
         elseif cond3(i,k(i))
             % \beta_i < min (left, right)
             tmp = min(C_left,C_right);
             if tmp > 0
                 beta_i(i)=0;
             else
                 beta_i(i)=tmp;
             end
         else
             beta_i(i)=-1;
             solvable=0;
         end
    end
end

function [x,ft] = EProjSimplex_new(v, k)

%
%% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=1
%
if nargin < 2
    k = 1;
end
ft=1;
n = length(v);

v0 = v-mean(v) + k/n;
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end
    end
    x = max(v1,0);
else
    x = v0;
end
end
function [acc,acc_ite,W] = AGLSP(Xs,Ys,Xt,Yt,options)
%% Implementation of AGLSP
%%% Authors                      Zheng et al.
%%% Title                        Adaptive Graph Learning with Semantic Promotability for Domain Adaptation
%% intput
%%%
%%% Xs                           The source samples with m * ns
%%%
%%% Ys                           The labels of source samples with ns * 1
%%%
%%% Xt                           The target samples with m * nt
%%%
%%% Yt                           The labels of source samples with nt * 1
%%%
%% options
%%% T                    -       The iterations
%%%
%%% dim                  -       The dimensions
%%%
%%% mu                   -       The weight between marginal and conditional matrix
%%%
%%% k1                   -       The number of neighbors w.r.t (3)
%%%
%%% k2                   -       The number of neighbors w.r.t AGE-SG and
%%%                              AGL-ISP
%%%
%%% alpha                -       The weight of Ls in L_{struct}
%%%
%%% delta                -       The weight of Lt in L_{struct}
%%%
%%% zeta                 -       The weight of AGE-SG and AGL-ISP
%%%
%%% sigma                -       The weight of SPSE
%%%
%%% lambda               -       The weight of regularization for projection matrix
%%%
%%% Kernel               -       Kernel function of orignal sample set
%%%
%%% SRM_Kernel           -       The hyperparameter of SRM classifier w.r.t
%%%                              kernel projection (e.g. rbf)
%%%
%%% SRM_mu               -       The hyperparameter of SRM classifier w.r.t
%%%                              regularization
%%%
%%% SRM_gamma            -       The hyperparameter of SRM classifier w.r.t
%%%                              kernel parameter
%%%
%% output
%%% acc                      	 The accuracy                  (number)
%%%
%%% acc_ite                      The accuracies of iterations  (list)
%%%
%%% W                            The projection matrix
%%%
%% === Version ===
%%%     Upload                   2023-08-02
    options=defaultOptions(options,...
                'T',10,...
                'dim',30,...
                'alpha',5,...
                'zeta',5,...
                'delta',1,... 
                'lambda',0.1,...
                'Kernel',0,...
                'k1',5,...
                'k2',10,...
                'sigma',1,...
                'mu',0.1,...
                'SRM_Kernel',2,...
                'SRM_mu',0.1,...
                'SRM_gamma',-1);
    %% init parameters
    T=options.T;
    dim=options.dim;
    alpha=options.alpha;
    zeta=options.zeta;
    delta=options.delta;
    sigma=options.sigma;
    lambda=options.lambda;
    mu=options.mu;
    k1=options.k1;
    k2=options.k2;
    Xs=normr(Xs')';
    Xt=normr(Xt')';
    acc=0;
    acc_ite=[];
    W=[];
    %% Init
    X=[Xs,Xt];
    X=L2Norm(X')';
    [m,ns]=size(Xs);
    nt=size(Xt,2);
    n=ns+nt;
    C=length(unique(Ys));
    if options.Kernel ~= 0
        X=kernelProject(options.Kernel,X,[],-1);
        m=n;
    end
    % Initialize pseudo label
    Ytpseudo=classifySVM(Xs,Ys,Xt);
    % Initialize `Ls` by (21)
    [Lss] = 1/ns*intraScatter(Xs,Ys,C);
    % Initialize `Lt` by (3)
    clear manifold;
    manifold.k = min(k1,nt-1);
    manifold.Metric = 'Cosine';
    manifold.WeightMode = 'Cosine';
    manifold.NeighborMode = 'KNN';
    [Ltt,~,~]=computeL(Xt,manifold);
    Lss=Lss./norm(Lss,'fro');
    Ltt=Ltt./norm(Ltt,'fro');
    % Initialize `LX` in (18)
    LX = blkdiag(alpha*(Lss),delta*Ltt);
    XLX_const=X*LX*X';
    % Initialize vector `v` for Lemma 1
    [F, ~, ~]=eig1(LX, C, 0);
    % Initialize M0 and Mc by (19) and (20), respectively
    M0 = marginalDistribution(Xs,Xt,C);
    Mc = conditionalDistribution(Xs,Xt,Ys,Ytpseudo,C);
    M = (1-mu)*M0+mu*Mc;
    M = M./norm(M,'fro');
    % Initialize centering matrix in (18)
    H=centeringMatrix(n);
    XHX=X*H*X';
    % Learning projection matrix `W` without `AGE-SP`, `SPSE`, and `AGL-ISP`
    left=X*M*X'+XLX_const+lambda*eye(m);
    right=XHX;%+alpha*(Xs*Lss2*Xs');
    [W,~]=eigs(left,right,dim,'sm');
    W=real(W);
    AX=W'*X;
    AX=L2Norm(AX')';
    AXs=AX(:,1:ns);
    AXt=AX(:,ns+1:ns+nt);
    % Initialize pseudo label
    Ytpseudo=classifySVM(AXs,Ys,AXt);
    fprintf('[init] acc:%.4f \n',getAcc(Ytpseudo,Yt)*100);
    %% Initialize the setting of AGE-SP and AGE-ISP
    clear opt;
    % Distance matrix in Lemma 1
    opt.dist= my_dist(AXt',AXs'); 
    % Semantic graph in Theorem 1
    opt.semanticGraph=hotmatrix(Ytpseudo,C,0)*hotmatrix(Ys,C,0)'; 
    % The number of neighbors
    opt.k=k2; 
    % The vector `v` in Lemma 1
    opt.F=my_dist(F,F);
    % A constant to control the rankness of L
    opt.lambda=1;
    % The weight of L_{AGLSP}
    opt.beta=zeta;
    % Solve AGE-SG and AGL-ISP by Theorem 1 and (17)
    [Ss,~]=solveAGE_SG_and_AGL_ISP(opt);
    %% Update Zs1, Zt1
    Lyst=0;
    for i=1:T
        % Update L_{AGE-SP}+L_{AGL-ISP} 
        Sim=[zeros(ns,ns),Ss';Ss,zeros(nt,nt)];
        [Lxu,~] = crossMicro_getL(Sim,1);
        % Initialize vector `v` for Lemma 1
        [F, ~, ~]=eig1(Lxu, C, 0);
        Lxu(isnan(Lxu))=0; % avoid invalid value
        Lxu(isinf(Lxu))=0; % avoid invalid value
        Lxst=Lxu./norm(Lxu,'fro');
        % Update `M0` and `Mc` by (19) and (20), respectively
        Mc=conditionalDistribution(Xs,Xt,Ys,Ytpseudo,C);
        M=(1-mu)*M0+mu*Mc;
        M=M./norm(M,'fro');
        % Learning projection matrix `W` by (18)
        left=X*(M+LX+sigma*Lyst+zeta*Lxst)*X'...
            +lambda*eye(m);
        [W,~]=eigs(left,XHX,dim,'sm');
        % Project the source and target sample set
        W=real(W);
        AX=W'*X;
        AX=L2Norm(AX')';
        AXs=AX(:,1:ns);
        AXt=AX(:,ns+1:n);
        %% Update AGE-SG and AGL-ISP by Theorem 1 and (17)
        clear opt;
        opt.dist= my_dist(AXt',AXs');
        graphYs=hotmatrix(Ytpseudo,C,0)*hotmatrix(Ys,C,0)';
        opt.semanticGraph=graphYs;
        opt.k=k2;
        opt.beta=zeta;
        opt.lambda=1;
        opt.F=my_dist(F,F);
        [Ss,beta_ti]=solveAGE_SG_and_AGL_ISP(opt);
        %% Classification
        SRMopt.mu=options.SRM_mu;
        SRMopt.Kernel=options.SRM_Kernel;
        SRMopt.gamma=options.SRM_gamma;
        [Ytpseudo,~,~,~] = SRM(AXs,AXt,Ys,SRMopt);
        acc=getAcc(Ytpseudo,Yt);
        acc_ite(i)=acc;
        %% Update SPSE by (22) and (23)
        betaFlag=beta_ti<0;
        Ytpseudo2=Ytpseudo;
        Ytpseudo2(betaFlag)=0;
        [Lyst] = intraScatter(X,[Ys;Ytpseudo2],C);
        Lyst=Lyst./norm(Lyst,'fro');
        %% Print the accuracy
        fprintf('[%2d] acc:%.4f\n',i,acc);
    end
end


function D = my_dist(fea_a,fea_b)
%% input:
%%% fea_a: n1*m
%%% fea_b: n2*m
    if nargin==1
        fea_b=0;
    end
    if nargin<=2
       [n1,n2]=size(fea_b);
       if n1==n2&&n1==1
           bSelfConnect=fea_b;
           fea_mean=mean(fea_a,1);
           fea_a=fea_a-repmat(fea_mean, [size(fea_a,1),1]);
           D=EuDist2(fea_a,fea_a,1);
           if bSelfConnect==0
                maxD=max(max(D));
                D=D+2*maxD*eye(size(D,1));
           end
           return ;
       end
    end
    fea_mean=mean([fea_a;fea_b],1);
    fea_a=fea_a-repmat(fea_mean, [size(fea_a,1),1]);
    fea_b=fea_b-repmat(fea_mean, [size(fea_b,1),1]);
    D=EuDist2(fea_a,fea_b,1);
end

function [L,D]=crossMicro_getL(S,normFlag)
    if nargin==1
       normFlag=1; 
    end
    n=size(S,1);
    D=diag(sparse(sum(S)));
    Dw = diag(sparse(sqrt(1 ./ (sum(S)+eps))));
    Dw(isinf(Dw))=0;
    if normFlag==1
        L = eye(n) - Dw * S * Dw;
    else
        L=D-S;
    end

end
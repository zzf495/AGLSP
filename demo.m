%% Implementation of AGLSP
%%% Paper: Adaptive Graph Learning with Semantic Promotability for Domain Adaptation
%%% Note: SVM is used to initialize pseudo-labels, but its performance varies between window and unix.
%%%       The experiments reported are performed in a parallel environment with multiple `Intel(R) Xeon(R) Gold 6230R`
clc; clear all;
addpath(genpath('./util/'));
srcStr = {'caltech','caltech','caltech','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgtStr = {'amazon','webcam','dslr','caltech','webcam','dslr','caltech','amazon','dslr','caltech','amazon','webcam'};
finalResult=[];

%% initialize the Ytpseudo by a basic classifier or not 
options=defaultOptions([],'T',10,...The iterations
            'dim',30,...            The dimensions
            'mu',[0.6],...          The weight between marginal and conditional matrix
            'alpha',0.2,...         The weight of Ls in L_{struct}
            'zeta',0.2,...          The weight of AGE-SG and AGL-ISP
            'delta',0.2,...         The weight of Lt in L_{struct}
            'sigma',0.01,...        The weight of SPSE
            'SRM_Kernel',2,...      The hyperparameter of SRM classifier w.r.t kernel projection
            'SRM_mu',0.1,...        The hyperparameter of SRM classifier w.r.t regularization
            'SRM_gamma',-1,...      The hyperparameter of SRM classifier w.r.t kernel parameter
            'Kernel',[2],...        The kernel function of orignal sample set
            'k1',[10],...           The number of neighbors w.r.t (3)
            'k2',[8],...            The number of neighbors w.r.t AGE-SG and AGL-ISP
            'lambda',[0.02]);%      The weight of regularization for projection matrix
%% Run the experiments
for i = 1:12
    src = char(srcStr{i});
    tgt = char(tgtStr{i});
    fprintf('%d: %s_vs_%s\n',i,src,tgt);
    load(['./data/' src '_SURF_L10.mat']);
    Xs = processData(fts',0,{'L2norm'  'zscore'  'normr'});
    Ys = labels;
    load(['./data/' tgt '_SURF_L10.mat']);
    Xt = processData(fts',0,{'L2norm'  'zscore'  'normr'});
    Yt = labels;
    [~,result,~] = AGLSP(Xs,Ys,Xt,Yt,options);
    finalResult=[finalResult;result];
end

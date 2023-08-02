function [Fs,label,W,Windex,similarXsIndex] = getCriticalClassPoints(Zs,Ys,Zt,index,mode)
    [m,ns]=size(Zs);
    nt=size(Zt,2);
    ni=length(index);
    C=length(unique(Ys));
    Fs={};%
    label=zeros(ni,1);
    epsilon=1e-5;
    % KNN
    % get KNN distance
    dist=EuDist2(Zt(:,index)',Zs');
    [~,sortIdx]=sort(dist,2,'ascend');
    Fs_index=zeros(m,C);
    similarXsIndex=zeros(C,ni);
    for i=1:ni
        posIdx=sortIdx(i,:);
        sortYs=Ys(posIdx);
        % sort for xt
        label(i)=sortYs(1);
        for k=1:C
           targetC=find(sortYs==k);
           targ=posIdx(targetC(1));
           Fs_index(:,k)=Zs(:,targ);
           similarXsIndex(k,i)=targ;
        end
        Fs{i}=Fs_index;
    end
    if strcmpi(mode,'NCM')
        % get centroids
        hotYs=hotmatrix(Ys,C,1);
        Fs_index=Zs*hotYs;
        % get distance
        dist=EuDist2(Zt(:,index)',Fs_index');
        [~,sortIdx]=sort(dist,2,'ascend');
        label=sortIdx(:,1);
        for i=1:ni
            Fs{i}=Fs_index;
        end
    elseif  strcmpi(mode,'SVM')
         cmd='-s 1 -q';
         svmmodel = train(double(Ys), sparse(double(Zs')),cmd);
         [label,~,~] = predict(zeros(ni,1), sparse(Zt(:,index)'), svmmodel,'-q');
    end
    % learn W
    W={};
    Windex={};
    if strcmpi(mode,'SVM')
        Wtmp=svmmodel.w';
        for i=1:ni
            Ytpseudo=label(i);
            weight_c=Wtmp(:,Ytpseudo);
            weight_c=repmat(weight_c,[1,C]);
            Wc=weight_c-Wtmp;
            Wc(abs(Wc)<epsilon)=0;
            [~,indexWc]=sort(Wc,1,'ascend');
            W{i}=Wc;
            Windex{i}=indexWc;
        end
    else
       for i=1:ni
            tmpZt=Zt(:,index(i));
            tmpFs=Fs{i};
            Ytpseudo=label(i);
            sameSemanticCentroid=tmpFs(:,Ytpseudo);
            same=tmpZt-sameSemanticCentroid;
            const=same*same';
            Wc=zeros(m,C);
            for k=1:C
                if k~=Ytpseudo
                    diffSemanticCentroids=tmpFs(:,k);
                    diff=tmpZt-diffSemanticCentroids;
                    % w: m * 1
                    % Aeq:  1 * m
                    cost=const-diff*diff';
    %                 [Wc(:,k),eigV]=eig1(cost,1,0);
                    cost=cost./norm(cost,'fro');
                    Wc(:,k)=eigs(cost,m,'sa');
                else
                    Wc(:,k)=eigs(const./norm(const,'fro'),m,'sa');
                end
            end
            Wc(abs(Wc)<epsilon)=0;
            [~,indexWc]=sort(Wc,1,'ascend');
            W{i}=Wc;
            Windex{i}=indexWc;
        end 
    end

end


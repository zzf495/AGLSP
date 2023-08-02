function [Zs,Zt,Ytpseudo,ns,nt,acc]=initializationEnviroment(Xs,Ys,Xt,Yt,W_pca,A)
    [~,ns]=size(Xs);
    [~,nt]=size(Xt);
    Xs2=W_pca'*Xs;
    Xt2=W_pca'*Xt;
    Xs2=normc(Xs2);
    Xt2=normc(Xt2);
    Z=A'*[Xs2,Xt2];
    Z=L2Norm(Z')';
    Zs=Z(:,1:ns);
    Zt=Z(:,ns+1:end);
    Ytpseudo=classifySVM(Zs,Ys,Zt);
    acc=getAcc(Ytpseudo,Yt);
end
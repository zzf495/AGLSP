function Ytpseudo=classifySVM(Xs,Ys,Xt,cmd)
    if nargin==3
        cmd='-s 1 -B 1.0 -q';
    elseif isempty(cmd)
        cmd='-s 1 -B 1.0 -q';
    end
     nt=size(Xt,2);
     Y=zeros(nt,1);
     svmmodel = train(double(Ys), sparse(double(Xs')),cmd);
     [Ytpseudo,~,~] = predict(Y, sparse(Xt'), svmmodel,'-q');
end
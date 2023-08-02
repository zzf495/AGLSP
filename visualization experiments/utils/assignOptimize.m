function [result]=assignOptimize(f,k)
% target: max       x * f * x'
%%%     f           m * m
    
    [m,~]=size(f);
    x=binvar(m,1);
%     x=sdpvar(m,1);
    z=x'*f*x;
    F=[];%[x'*ones(m,1)>=k];
    options=sdpsettings('solver','Cplex');
    options.verbose=0;
%     options.warning=1;
    options.debug=1;
    options.showprogress=1;
    sol=optimize(F,z,options);
%     optimize(F,z);
    result=value(x);
end
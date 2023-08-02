function res=formatPath(path,isDir)
    if nargin==1
       isDir=false; 
    end
    res=replace(path,'\','/');
    res=replace(res,'//','/');
    if isDir
        if ~endsWith(res,'/')
            res=[res '/'];
        end
    end
end
function flag=isInList(path,list)
    flag = false;
    if isempty(list)
        flag = false;
        return;
    end
    if iscell(list)
        n = length(list);
        for i = 1:n
           if endsWith(path,list{i})
              flag=true;
              return;
           end
         end
    else
        n = length(list);
        for i = 1:n
           if path==list(i)
             flag=true;
             return;
           end
        end
    end
end
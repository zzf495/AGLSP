function [newImg,cam]=calculateWeight(path,weights,target,ProW,iCombOpt)
%%% Input:
%%%       weight: n * 1
%%%       target: n * k * k
%%%       ProW:   n * d
%%%       inputN: the size of imgage
    [n,k,~]=size(target);
    cam=ones(k,k);
    if isempty(ProW)
        projectFlag=false;
    else
        projectFlag=true;
        ProW=mean((ProW),2);
        ProW=(ProW-mean(ProW))./(max(ProW)-min(ProW)); % from -1 to 1
    end
    for i=1:n
        if projectFlag
            w=weights(i)*(1+ProW(i));
        else
            w=weights(i);
        end
       tmp=target(i,:,:);
       tmp=reshape(tmp,k,k);
       cam=cam+w*tmp;
    end
%     cam=max(cam,0);
    cam = (cam - min(min(cam))) / (max(max(cam)) - min(min(cam)));
    cam = cam*255;

    % draw picture
    if isstr(path)
        img=imread(path);
    else
        img=path;
    end
    [newImg] = imCombine(cam,img,iCombOpt);
end


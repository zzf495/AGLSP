function [newImg,flag] = pixelsMerged(resizeCam,options)
    options=defaultOptions(options,'level',10,...
                            'top',2,...
                            'mergeTop',true,...
                            'mergeLow',true,...
                            'low',2,...
                            'standardMerged',false);
    level=options.level;
    low=options.low;
    top=options.top;
    mergeTop=options.mergeTop;
    mergeLow=options.mergeLow;
    if length(size(resizeCam))==3
        resizeCam=rgb2gray(resizeCam);
    end
    [m,n]=size(resizeCam);
    newImg=zeros(m,n);
    maxPixel=max(max(resizeCam));
    minPixel=min(min(resizeCam));
    if level == -1
        camStd=(mean(std(resizeCam,0,1))+mean(std(resizeCam,0,2)))/2;
        level=ceil((maxPixel-minPixel)/(camStd));
        offset=(maxPixel-minPixel)/level;
    else
        offset=(maxPixel-minPixel)/level;
    end
    
    right = maxPixel;
    right_lev=level;
    while(right>minPixel)
        flag = (resizeCam <= right)&(resizeCam > right-offset);
        if right_lev>(level-top)
            if mergeTop
                newImg(flag)=level; 
            else
               newImg(flag)=right_lev; 
            end
        elseif right_lev<=low
            if mergeLow
                newImg(flag)=1;
            else
                newImg(flag)=right_lev; 
            end
        end
        right = right- offset;
        right_lev = right_lev - 1;
    end
    b1=newImg(1:end-2,2:end-1);
    b2=newImg(2:end-1,2:end-1);
    b3=newImg(3:end,2:end-1);
    r1=newImg(2:end-1,1:end-2);
    r2=newImg(2:end-1,2:end-1);
    r3=newImg(2:end-1,3:end);
    f1=(b1==b2);
    f2=(b2==b3);
    f3=(r1==r2);
    f4=(r2==r3);
    flagRow1=f1&f2;
    flagCo1=f3&f4;
    %%% setting
    flag=true(m,n);
    flag(1,:)= newImg(1,:)==newImg(2,:);
    flag(end,:)= newImg(end,:)==newImg(end-1,:);
    flag(:,1)= newImg(:,1)==newImg(:,2);
    flag(:,end)= newImg(:,end)==newImg(:,end-1);
    flag(2:end-1,2:end-1)=~(flagRow1&flagCo1);
    newImg(~flag)=0;
    
    flag=flag&(newImg~=0);
%     newImg=(newImg-min(min(newImg)))./(max(max(newImg))-min(min(newImg)));
end


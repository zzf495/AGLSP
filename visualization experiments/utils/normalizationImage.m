function [newImg] = normalizationImage(img)
    [m,n,channels]=size(img);
    newImg=zeros(m,n,channels);
    for i=1:channels
        tmp=img(:,:,i);
        minTmp=min(min(tmp));
        maxTmp=max(max(tmp));
        tmp=(tmp-minTmp)/(maxTmp-minTmp);
        newImg(:,:,i)=tmp;
    end
end


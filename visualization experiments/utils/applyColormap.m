function [img] = applyColormap(resizeCam)
    [m,n]=size(resizeCam);
    resizeCam = (resizeCam - min(min(resizeCam))) / (max(max(resizeCam)) - min(min(resizeCam)));
    mycolorMap=colormap('Turbo');
%     load('myColor.mat')
%     mycolorMap=ColormapUtils.myColorMapRedToBlue(255);
    colorNum=size(mycolorMap,1);
    img=zeros(m,n,3);
    for i=1:m
        for j=1:n
            val=resizeCam(i,j);
            idx=min(floor(val*colorNum)+1,colorNum);
            for k=1:3
               img(i,j,k)=mycolorMap(idx,k);
            end
        end
    end
end


function imgsList=appendCells(imgsList,varargin)
    count=length(imgsList)+1;
    for z=1:nargin-1
        imgs=varargin{z};
        for k=1:length(imgs)
            imgsList{count}=imgs{k};
            count=count+1;
        end
    end
end
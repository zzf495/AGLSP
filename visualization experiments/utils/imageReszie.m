function [F] = imageReszie(varargin)
    % Create time: 21-06-14 by Zefeng Zheng
    % Description: Resize image
    % input: 
    %       path (necessary): where to read
    %       resizeSize (necessary): the resize size, rate (0-1) or 2*1
    %                               matrix (specific value)
    %       savePath (optional): save to where
    % output: resized-Image
if nargin <=1
    error('At least two parameter is required\n');
    return ;
end
path=varargin{1};
if isstr(path)
    I=imread(path);
else
    I=(path);
end
resizeSize=varargin{2};
F=imresize(I,resizeSize,'bilinear');
if nargin >=3
    savePath=varargin{3};
    imwrite(F,savePath);
end
end


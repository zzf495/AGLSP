function [newImage,bottomMarginList,rightMarginList] = imageSplicing(options)
% Create time: 21-06-14 by zefengzheng
% Modified: 23-04-09 by zefengzheng
%%% Description: stitch several images together
%%  Input
%       imageSize               The size of images and row/col (row, col, width,height)
%
%       path (cells)            The stitched images
%
%       rightMargin             The right margin (list)
%
%       bottomMargin            The bottom margin (list)
%
%       tightFlag               Indicate whether to generate the margin at
%                               the last row
%
%%   Output
%       newImage                The stitched-image with size
%
%                               Width: sum(rightMargin)+imageSize.width * imageSize.col
%
%                               Height: sum(bottomMargin)+imageSize.height * imageSize.row
%

options=defaultOptions(options,'imageSize',[4,4,224,224],...
                               'rightMargin',0,...
                               'bottomMargin',0,...
                               'tightFlag',true);
imageSize=options.imageSize;
rightMargin=options.rightMargin;
bottomMargin=options.bottomMargin;
imagePath=options.path;
tightFlag=options.tightFlag;
n=length(imagePath);
%% start

%%% setup row and col
row=imageSize(1);
col=imageSize(2);
if row == -1 && col == -1
    row=ceil(sqrt(n));
    col = row;
elseif row==-1 && col > 0
    row = ceil(n / col);
elseif col==-1 && row > 0
    col =ceil(n / row);
end

if n>row*col
    error('row*col < n!\n');
    return;
end
%%% setup margin
if length(rightMargin)==1
   rightMargin = double(rightMargin)*ones(n,1);
elseif length(rightMargin)<n
    if size(rightMargin,2)>0
        rightMargin=rightMargin';
    end
    rightMargin=repmat(rightMargin,[ceil(n/length(rightMargin)),1]);
end

if length(bottomMargin)==1
   bottomMargin = double(bottomMargin)*ones(n,1);
elseif length(bottomMargin)<n
    if size(bottomMargin,2)>0
        bottomMargin=bottomMargin';
    end
    bottomMargin=repmat(bottomMargin,[ceil(n/length(bottomMargin)),1]);
end

image={};
for i=1:n
    path=imagePath{i};
    if isstr(path)
        image{i}=imread(path);
    else
        image{i}=path;
    end
end
x=imageSize(3);
y=imageSize(4);

for i=1:n
    iSize=size(image{i});
    z=iSize(3);
    if x~=-1&&y~=-1
        if ~((iSize(1)==x)&&(iSize(2)==y))
            % resize
            image{i}=imageReszie(image{i},[x,y]);
        end
    end
    nowRow=fix( (i-1)/col+1);
    nowCol=mod(i-1,col)+1 ;
    if nowRow > 1
        btmMargin=sum(bottomMargin(1:nowRow-1));
    else
        btmMargin=0;
    end
    
    if nowCol > 1
        rMargin=sum(rightMargin(1:nowCol-1));
    else
        rMargin=0;
    end
    L1=btmMargin + (nowRow-1)*x+1;
    L2=btmMargin + nowRow*x;
    L3=rMargin+(nowCol-1)*y+1;
    L4=rMargin+nowCol*y;
    newImage(L1:L2,L3:L4,:)=image{i};
    if i == n
        bottomMarginList=[];
        rightMarginList=[];
        pos=nowRow;
        if tightFlag
            pos = pos - 1;
        end
        for k=1:pos
            if k > 1
                btmMargin=sum(bottomMargin(1:k-1));
            else
                btmMargin=0;
            end
            L1=k*x+btmMargin+1;
            L2=bottomMargin(k)+k*x+btmMargin;
            if L1<L2
                newImage(L1:L2,:,:)=1;
            end
            bottomMarginList=[bottomMarginList;L1,L2];
        end
        for k=1:nowCol-1
            if k > 1
                rMargin=sum(rightMargin(1:k-1));
            else
                rMargin=0;
            end
            
            L3=k*y+rMargin+1;
            L4=rightMargin(k)+k*y+rMargin;
            if L3<L4
                newImage(:,L3:L4,:)=1;
            end
            rightMarginList=[rightMarginList;L3,L4];
        end
    end
end
end

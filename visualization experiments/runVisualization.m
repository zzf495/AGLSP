%% load path
clear;
addpath(genpath('./utils'));

%% ============== Configuration =======================
%% Note: we only use the first twenty images as example
%%% If you want to generate more images, please annotate the following code
%%%      `selectedFlag(10:end)=True;` at line 110

%% The weight of each image extracted by Grad-CAM.
%%%%
%%%%  Each weight is named as `img[idx].mat`, where `[idx]` denotes the order of images extracted by ResNet50
%%%%
%%%%    For example:
%%%%
%%%%        /amazon
%%%%            /img1.mat
%%%%            /img2.mat
%%%%            ... 
%%%%
%%%%
%%%% 
targetParamDir='./params/Office31/dslr'; % img_idx
sourceParamDir='./params/Office31/amazon'; % img_idx
%% The raw images, the structure likes
%%%%        /amazon
%%%%            /back_pack
%%%%                /frame_0001.jpg
%%%%                /frame_0002.jpg
%%%%                ...
%%%%            /bike
%%%%                /frame_0001.jpg
%%%%                ...
imgTargetDir='./raw/Office31/dslr'; % xxx.jpg
imgSourceDir='./raw/Office31/amazon';

%% The features extracted by ResNet50
sourceDataPath='./data/office-A-resnet50-noft.mat';
targetDataPath='./data/office-D-resnet50-noft.mat';
% The folder that saves the images
saveDir='./tmp/';

%% ================ Program Start =======================
sourceParamDir=formatPath(sourceParamDir,true);
targetParamDir=formatPath(targetParamDir,true);
imgSourceDir=formatPath(imgSourceDir,true);
imgTargetDir=formatPath(imgTargetDir,true);
saveDir=formatPath(saveDir,true);
prefix='img';
inputN=128;%224;
alpha=0.8; % The alpha value of cam
verbose=true;
%%%% load folder
[targetPathList,targetImageName,targetLabels,targetCategories] = loadFolderImages(imgTargetDir,verbose);
[sourcePathList,sourceImageName,sourceLabels,sourceCategories] = loadFolderImages(imgSourceDir,verbose);

%% start
rng(1);
%%% Mode:
%%%         0       Show origin
%%%         1       Show ResNet50
%%%         2       Show ResNet50 + Algorithm
%%%         3       Show difference between ResNet50 and Algorithm
%%%         4       Show difference between Algorithm A and Algorithm B
% Setting
number=10;  % Number of selected pictures
repeat=50;  % Repeat number (how many images to be produced)
%% Load data
cmd_processData={'L2norm','zscore','normr'};
clear resnet50_features labels;
load(sourceDataPath);
Xs = processData(resnet50_features',0,cmd_processData);
Ys = double(labels')+1;
clear resnet50_features labels;
load(targetDataPath);
Xt = processData(resnet50_features',0,cmd_processData);
Yt = double(labels')+1;
X=[Xs,Xt];
%% Initialize the projection matrix
projection={};
projection{1}=[];

%% load algorithm 1 (AGE-CS)
clear AXs AXt Ytpseudo;
load('./algorithm_on_Office31/AGE-CS.mat');
AXs1=AXs;
AXt1=AXt;
Ytpseudo1=Ytpseudo;
AX=[AXs1,AXt1];
W=(X*AX')/(AX*AX'+eye(size(AX,1)));
loss1=norm(X-W*AX,'fro');
projection{2}=W;

%% load algorithm 2 (AGLSP)
clear result;
load('./algorithm_on_Office31/AGLSP.mat');
AXs2=AXs;
AXt2=AXt;
Ytpseudo2=Ytpseudo;
AX=[AXs2,AXt2];
W=(X*AX')/(AX*AX'+eye(size(AX,1)));
loss2=norm(X-W*AX,'fro');
projection{3}=W;
fprintf('Loss1: %.4f, loss2: %.4f\n',loss1,loss2);

%% Select the images which are correctly classified
% Ytpseudo1(~(Ytpseudo1==Yt))=0;
%% start up
n = length(targetPathList);
selectedFlag=false(n,1);
selectedFlag(20:end)=true;  %% Note: we only use the first twenty images as example

text_params={'HorizontalAlignment','center','FontName','Times New Roman',...
    'Units','data','interpreter','latex',...
    'Clipping','off','FontSize',10};
for currentRepeat=1:repeat
    [selectImg,selectedFlag] = selectImagesFromClass(number,Ytpseudo1,selectedFlag,verbose);
    if length(selectImg)<number
       return; 
    end
    labels={};
    for tmpIdx=1:length(selectImg)
        name=targetCategories{Yt(selectImg(tmpIdx))};
        labels{tmpIdx}=replace(name,'_',' ');
    end
    %%% set options
    options.selectImg=selectImg;
    options.pathList=targetPathList;
    options.paramDir=targetParamDir;
    options.prefix=prefix;
    options.projection=projection;
    options.inputN=inputN;
    options.alpha=alpha;
    options.verbose=verbose;
    %%% deal with images
    imgsList={};
    count=1;
    options.standardMerged=false;
    % Origin
    options.mode=0;
    imgs1 = getMultiImagesFromExtractor(options);

    % AGE-CS
    options.mode=2;
    options.index=[2];
    imgs2 = getMultiImagesFromExtractor(options);

    % AGLSP
    options.mode=2;
    options.index=[3];
    imgs3 = getMultiImagesFromExtractor(options);

    % AGLSP ==> AGE-CS
    options.mode=3;
    options.index=[3,2];
    imgs4 = getMultiImagesFromExtractor(options);
    % concatenate the images
    splOpt.path=appendCells(imgs1,imgs2,imgs3,imgs4);
    imgConcatSize=[-1,number,inputN,inputN];
    splOpt.imageSize=imgConcatSize;
    splOpt.rightMargin=[10,10,10,10];
    splOpt.bottomMargin=[10];
    splOpt.tightFlag=true;
    figs=imageSplicing(splOpt);
    imshow(figs);
    for tmpIdx=1:length(labels)
       xx=(inputN+10)*(tmpIdx-1)+inputN/2;
       text(xx,(inputN+10)*4+20,labels{tmpIdx},text_params{:}); 
    end
    colorbar
    dirSave=[saveDir '/Office31'];
    if ~exist(dirSave,'dir')
        mkdir(dirSave)
    end
%     saveFigure(gcf,dirSave,'pic','.tiff',currentRepeat,'',300);
%     close;
end


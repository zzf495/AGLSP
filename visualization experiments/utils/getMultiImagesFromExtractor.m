%% getMultiImagesFromExtractor
% Concat several images by setting
%%% selectImg       The index of selected images
%%% pathList        The cell list that stores the path of pictures
%%% paramDir        The directory path that stores `.mat`
%%% alpha           The alpha value of picture for mixup
%%% inputN          The size of picture
%%% prefix          The prefix name of image.mat with `weights` and
%%%                 `target`
%%% projection      The projection matrix with m * d
%%% index           An indicator matrix with 2 * 1
%%% Mode:
%%%         0       Show origin
%%%         1       Show ResNet50
%%%         2       Show ResNet50 + Algorithm
%%%         4       Show difference between Algorithm A and Algorithm B
%%%                 indicated by `options.index=[idxA,idxB]`
%%%                 where `options.projection{idxA}=A` and `options.projection{idxB}=B`
%%
function imgs=getMultiImagesFromExtractor(options)
    %% load parameters
    options=defaultOptions(options,...
                        'prefix','img',...
                        'inputN',256,...
                        'index',[1,1],...
                        'alpha',0.8,...
                        'standardMerged',false,...
                        'verbose',false);
    selectImg=options.selectImg;
    pathList=options.pathList;
    paramDir=options.paramDir;
    prefix=options.prefix;
    mode=options.mode;
    inputN=options.inputN;
    alpha=options.alpha;
    verbose=options.verbose;
    standardMerged=options.standardMerged;
    %% start
    imgs={};
    iCombOpt=options;
    for i=1:length(selectImg)
        idx=selectImg(i);
        path=pathList{idx};
        % Origin
        originImg=imread(path);
        if mode == 0 % origin
           currentImg=double(originImg)./255;
           ProA=[];
        elseif mode>0
            % loadWeight
            paramPath=[paramDir prefix num2str(idx) '.mat'];
            clear target weights;
            load(paramPath);
            if verbose
                fprintf('Load weight graph from %s\n',paramPath);
            end
            if mode==1 % origin + ResNet50
                [img2,~]=calculateWeight(originImg,weights,target,[],iCombOpt);
                currentImg=img2;
                ProA=[];
            elseif mode==2 % ResNet50 + Algorithm
                index=options.index;
                projection=options.projection;
                ProA=projection{index(1)};
                
                [img1,~]=calculateWeight(originImg,weights,target,ProA,iCombOpt);
                currentImg=img1;
            elseif mode==3 % diff between Alg1 and Alg2
                index=options.index;
                projection=options.projection;
                ProA=projection{index(1)};
                ProB=projection{index(2)};
                [~,cam1]=calculateWeight(originImg,weights,target,ProA,iCombOpt);
                [~,cam2]=calculateWeight(originImg,weights,target,ProB,iCombOpt);
                cam3=cam1-cam2;
                [img3] = imCombine(cam3,originImg,iCombOpt);
                currentImg=img3;
            end
        end
        imgs{i}=currentImg;
    end
    fprintf('Sum of ProA: %2f\n',norm(ProA,'fro'));
end
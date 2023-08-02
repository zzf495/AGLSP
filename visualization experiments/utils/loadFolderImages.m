function [pathList,imageName,labels,categories] = loadFolderImages(imgDir,verbose)
    imgList={'.jpg','.jpeg','.png','.tiff','.bmp','.gif'};
    imgDir=formatPath(imgDir,true);
    %% load images
    import java.util.LinkedList
    folderList = LinkedList();
    pathList={};
    imageName={};
    categories={};
    folderList.add(imgDir);
    count = 1;
    labels=[];
    labelCount=-1;
    while (folderList.size()>0)
        tmpPath=folderList.remove();
        labelCount=labelCount+1; % dir
        if verbose
            fprintf('Find folder: %s\n',tmpPath);
        end
        folders = dir(fullfile(tmpPath));
        n = length(folders);
        labCount=1;
        for i = 1:n
            folder=folders(i);
            if folder.isdir==1&&~strcmp(folder.name,'.')&&~strcmp(folder.name,'..')&&~strcmp(folder.name,'.git')
                folderPath=[folder.folder '/' folder.name '/'];
                folderList.add(folderPath);
                if labelCount==0
                    categories{labCount}=folder.name;
                    labCount=labCount+1;
                end
            elseif folder.isdir==0
                name=folder.name;
                if isInList(name,imgList)
                    filesPath=[folder.folder '/' name];
                    pathList{count}=formatPath(filesPath);
                    imageName{count}=name;
                    labels(count)=labelCount;
                    count=count+1;
                end
            end
        end
    end
end


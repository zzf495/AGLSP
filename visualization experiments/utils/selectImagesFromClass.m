function [selectImg,selectedFlag] = selectImagesFromClass(number,labels,selectedFlag,verbose)
    selectImg=[];
    %%% Select by class, then select randomly
    C=length(unique(labels));
    if number>C
        selectedNum=max(1,floor(number/C));
        if verbose
            fprintf('Select %d images from classes\n',selectedNum);
        end
        for i=1:C
            idxC=(labels==i)&~selectedFlag;
            idxx=find(idxC);
            nc=length(idxx);
            selectedNumC=selectedNum;
            while selectedNumC>0 && nc>0
               idx=idxx(randperm(nc,1));
               if ~selectedFlag(idx)
                   selectImg=[selectImg,idx];
                   selectedFlag(idx)=true;
                   selectedNumC=selectedNumC-1;
                   nc=nc-1;
               end
            end
        end
        residualNumber=number-length(selectImg);
    else
        residualNumber=number;
    end
    if verbose
        fprintf('Select %d images randomly\n',residualNumber);
    end
    idxx=find(~selectedFlag);
    num1=length(idxx);
    if (residualNumber>0)&&(num1<residualNumber)
        if verbose
            fprintf('[Warning] The number of images is %d, but we need %d.\n',num1,residualNumber);
        end
        selectImg=[selectImg,idxx];
        selectedFlag(idxx)=true;
        return ;
    elseif residualNumber>0
        idxx2=randperm(num1,residualNumber);
        idx=idxx(idxx2);
        selectImg=[selectImg,idx];
        selectedFlag(idx)=true;
    end
end


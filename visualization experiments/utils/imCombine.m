function [newImg] = imCombine(cam,img,options)
    options=defaultOptions(options,'alpha',0.8,...
                            'inputN',224,...
                            'standardMerged',false);
    inputN=options.inputN;
    alpha=options.alpha;
    standardMerged=options.standardMerged;
    % change the size of input image
    resizeImg = double(imresize(img,[inputN,inputN],...
        'Method','bilinear',...
        'Antialiasing',true))./255;
    
    % change the size of `cam`
    if standardMerged
        resizeCam = double(imresize(cam,[inputN,inputN],...
            'Method','box',...
            'Antialiasing',true))./255;
        [resizeCam,flag] = pixelsMerged(resizeCam,options);
        alpha=double(flag);
        % use colormap to draw `cam`
        img2 = (applyColormap(resizeCam));
        % alpha mixup
        [newImg] = alphaCombine(img2,resizeImg,alpha);
    else
        resizeCam = double(imresize(cam,[inputN,inputN],...
            'Method','bicubic',...
            'Antialiasing',true))./255;
        % use colormap to draw `cam`
        img2 = (applyColormap(resizeCam));
        % alpha mixup
        [newImg] = alphaCombine(img2,resizeImg,alpha);
    end
    
    
end


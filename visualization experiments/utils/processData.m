function [data,msg] = processData(X,reportMessage,varargin)
%% input:
%%% X: the processed data, m*n
%%% varargin: the processing
%%%         optional: for a sampels set X (m*n)   
%%%             normr/L2norm: map into [-1,1], and all(sum(X.^2,1))=1 
%%%                         (the squred sum of the features of each sampels is 1)
%%%             zscore: the mean of the features is 0, and the std of the
%%%                     features is 1
%%%             sumNorm: x./(sum(x)) 
%%%                         (the sum of the features of each sampels is 1)
%%%             minMax: map into [0,1]
    inputSize=(nargin-2);
    if reportMessage
        fprintf('===>  Begin process data (input:(%d,%d))\n',size(X,1),size(X,2));
    end
    if inputSize>=1
         msg='';
         for i=1:inputSize
            [X,msgi]=begin_with_cells(X,reportMessage,varargin{i});
            msg=[msg msgi];
         end
    end
    if reportMessage
        fprintf('\n===>  End process data (output:(%d,%d))\n',size(X,1),size(X,2));
    end
    data=X;
end
function [data,msg]=begin_with_cells(X,reportMessage,cells_input)
    % if 'cells_input' is a cell, then loop by stack
    if iscell(cells_input)
        msg=[];
        for i=1:length(cells_input)
            [X,msg_i]=begin_with_cells(X,reportMessage,cells_input{i});
            msg=[msg msg_i];
        end
    else
       
        [X,flag]=begin_process(X,cells_input);
        if reportMessage
            fprintf('%s (%s) ',cells_input,flag);
        end
        msg=[ '_' cells_input '(' flag ')'];
    end
    data=X;
end
function [data,flag]=begin_process(X,i_cmd)
%% input:
%%%      X: m*n
    flag='matched';
    if strcmpi(i_cmd,'normr')||strcmpi(i_cmd,'L2norm')
        data=L2Norm(X')';
    elseif strcmpi(i_cmd,'zscore')
        data= double(zscore(X',1))';
    elseif strcmpi(i_cmd,'sumNorm')
        fts=X';
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
        data= fts';
    elseif strcmpi(i_cmd,'minMax')
        n=size(X,2);
        minX=min(X,[],2);
        maxX=max(X,[],2);
        minusX=maxX-minX;
        % expand
        minX=repmat(minX,1,n);
        minusX=repmat(minusX,1,n);
        data=(X-minX)./minusX;
        
    else
        
        flag='unmatched';
        data=X;
    end
end


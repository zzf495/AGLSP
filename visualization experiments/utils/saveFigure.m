function saveFigure(gcf,path,prefix,suffix,arabicNumerals,letter,dtiff)
%% input
%%%           dtiff         -       The resolution of the figure, such as
%%%                                 150, 220, 300, 600, etc.
       if isempty(arabicNumerals)
          arabicNumerals=[];
       else
          arabicNumerals=num2str(arabicNumerals);
       end
       if isempty(prefix)
          prefix=[];
       end
       if isempty(suffix)
          suffix='.tiff';
       elseif ~strcmpi(suffix(1),'.')
          suffix=['.' suffix];
       end
       if nargin <= 5
           dtiff = 600;
       end
       if ~isempty(letter)
          if isnumeric(letter)
             % char(97) => `a`
             idx =  (96 + mod(letter-1,26)+1);
             letter = char(idx);
          end
       end
       filename = [path '/' prefix arabicNumerals letter suffix];
       fprintf('Save to `%s`\n',filename);
       dtiff=['-r' num2str(dtiff)];
       print(gcf,'-dtiff',dtiff,filename)
end


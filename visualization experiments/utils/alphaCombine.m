function [newImg] = alphaCombine(img1,img2,alpha)
    [R1,G1,B1]=getRGB(img1);
    [R2,G2,B2]=getRGB(img2);
    newImg=zeros(size(img1));
    newImg(:,:,1)=alpha.*R1+(1-alpha).*R2;
    newImg(:,:,2)=alpha.*G1+(1-alpha).*G2;
    newImg(:,:,3)=alpha.*B1+(1-alpha).*B2;
end

function [R,G,B]=getRGB(img)
    R=img(:,:,1);
    G=img(:,:,2);
    B=img(:,:,3);
end


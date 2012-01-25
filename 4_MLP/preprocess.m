function immat= preprocess(im)
    %im=double(im);
    immat=imresize(im, [40 40]);
    %figure,imshow(immat);
    immat=im2bw(immat,graythresh(immat));
    %figure,imshow(immat);
    immat=double(immat);
end
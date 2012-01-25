function data=myfilter(I, fs)
    %I = imread('C:\\Users\\nautilus\\Desktop\D49_3.bmp');
    [r,c,p] = size(I);
    
    sv = ones(fs,fs) .* 1/(fs*fs);

    fd = imfilter(double(I(:,:,1)),sv,'replicate');

    data = zeros(1, r*c);
    for i=1:r
        data((i-1)*c+1:i*c) = fd(i,1:c);
    end
   
end

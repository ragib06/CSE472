test_path = 'bang_numeral_test/';
train_path = 'bang_numeral_train/';

NI = 25;

class = zeros(1, 250);

HL = [5, 10, 25, 50];
ITR = [100, 300, 500, 1000];
BS = [3, 5, 9, 15];

imd = 40;

for i=0:9
    for j=1:NI
        path = strcat (train_path, num2str(i), '/', num2str(j+NI), '.tif');
        I = imread(path);
        I = preprocess(I);
        
        train_images(i*imd+1:(i+1)*imd, (j-1)*imd+1:j*imd ) = I;
        class(i*NI+j) = i;
        
        path = strcat (test_path, num2str(i), '/', num2str(j), '.tif');
        I = imread(path);
        I = preprocess(I);
        
        test_images(i*imd+1:(i+1)*imd, (j-1)*imd+1:j*imd) = I;
        
    end
end

%run(train_images, test_images, class, 50, 1000, 5)



fid = fopen('out.txt','w');

for i=1:4
    fprintf(fid, 'NHL: %d\n',HL(i));
    fprintf(fid, '-----\n\n');
    fprintf(fid, '\t%-8s%8s%8s%8s%8s\n','bs','100', '300', '500', '1000');
    fprintf(fid, '\t%-8s%8s%8s%8s%8s\n','----','-------', '-------', '-------', '-------');
    for j=1:4
        fprintf(fid, '\t%-8g',BS(j));
        for k=1:4
            
            acc = 0;
            for l=1:25
                acc = acc + 1-run(train_images, test_images, class, HL(i), ITR(k), BS(j));
            end
            acc = acc./25;
            
            fprintf(fid, '%7g%%',acc*100);
        end
        fprintf(fid, '\n');
    end
    fprintf(fid, '\n');
end

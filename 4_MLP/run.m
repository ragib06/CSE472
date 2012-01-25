function error= run(trainIm, testIm, class, hl, itr, bs)

    imd = 40;
    
    train_feature = zeros(250, 1600);
    test_feature = zeros(250, 1600);
    
    for i=1:10
        for j=1:25
            I = trainIm( (i-1)*imd+1:i*imd, (j-1)*imd+1:j*imd );
            train_feature((i-1)*25 + j, 1:1600) = myfilter(I, bs);
            
            I = testIm( (i-1)*imd+1:i*imd, (j-1)*imd+1:j*imd );
            test_feature((i-1)*25 + j, 1:1600) = myfilter(I, bs);
        end
    end
    
    train = dataset(train_feature,class');
    test  = dataset(test_feature,class');

    td = bpxnc(train,hl,itr);
    result = test * td;
    error=testc(result);
    %confmat(result);
    
end
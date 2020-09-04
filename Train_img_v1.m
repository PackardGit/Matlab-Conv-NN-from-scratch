clear all;

warning off parallel:gpu:device:DeviceLibsNeedsRecompiling
try
    gpuArray.eye(2)^2;
catch ME
end
try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end

nr_img=20000;
[imgs_l labels] = readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte', nr_img, 0);
y = gpuArray.zeros(10,20000); %Correct outputs vector
for i = 1:nr_img
    y(labels(i)+1,i) = 1;
end
for i=1:nr_img
images_temp(:,i)=reshape(imgs_l(:,:,i),[28*28,1]);
end
images=gpuArray(images_temp);
 %Input vectors
hn1 = 80; %Number of neurons in the first hidden layer
hn2 = 60; %Number of neurons in the second hidden layer
%Initializing weights and biases
w12 = gpuArray.randn(hn1,784)*sqrt(2/784);
w23 = gpuArray.randn(hn2,hn1)*sqrt(2/hn1);
w34 = gpuArray.randn(10,hn2)*sqrt(2/hn2);
b12 = gpuArray.randn(hn1,1);
b23 = gpuArray.randn(hn2,1);
b34 = gpuArray.randn(10,1);

%learning rate
eta = 0.0058;
%Initializing errors and gradients

epochs = 10;
m = 10; %Minibatch size
tic
for k = 1:epochs %Outer epoch loop
    
    batches = 1;
    
    for j = 1:nr_img/m
            error4 = gpuArray.zeros(10,1);
            error3 = gpuArray.zeros(hn2,1);
            error2 = gpuArray.zeros(hn1,1);
            errortot4 = gpuArray.zeros(10,1);
            errortot3 = gpuArray.zeros(hn2,1);
            errortot2 = gpuArray.zeros(hn1,1);
            grad4 = gpuArray.zeros(10,1);
            grad3 = gpuArray.zeros(hn2,1);
            grad2 = gpuArray.zeros(hn1,1);
       
        for i = batches:batches+m-1 %Loop over each minibatch
    
             %Feed forward
             a1 = images(:,i);
             z2 = w12*a1 + b12;
              a2 = gpuArray.zeros(length(z2),1);
             for it = 1:length(z2)
                 if z2(it)>=0
                   a2(it) = z2(it);
                 else
                 a2(it) = 0.2*(exp(z2(it))-1);
                 end
             end
             z3 = w23*a2 + b23;
             a3 = gpuArray.zeros(length(z3),1);
             for it = 1:length(z3)
                 if z3(it)>=0
                   a3(it) = z3(it);
                 else
                 a3(it) = 0.2*(exp(z3(it))-1);
                 end
             end
             z4 = w34*a3 + b34;
             a4 = gpuArray.zeros(length(z4),1);
             for it = 1:length(z4)
                 if z4(it)>=0
                   a4(it) = z4(it);
                 else
                 a4(it) = 0.2*(exp(z4(it))-1);
                 end
             end %Output vector
    
            
              error4 = (a4-y(:,i));%.*elup_gpu(z4);
              error3 = (w34'*error4);%.*elup_gpu(z3);
              error2 = (w23'*error3);%.*elup_gpu(z2);
    
              errortot4 = errortot4 + error4;
              errortot3 = errortot3 + error3;
              errortot2 = errortot2 + error2;
              grad4 = grad4 + error4*a3';
              grad3 = grad3 + error3*a2';
              grad2 = grad2 + error2*a1';
        end
    
         %Gradient descent
         w34 = w34 - eta/m*grad4;
         w23 = w23 - eta/m*grad3;
         w12 = w12 - eta/m*grad2;
         b34 = b34 - eta/m*errortot4;
         b23 = b23 - eta/m*errortot3;
         b12 = b12 - eta/m*errortot2;
    
         batches = batches + m;
    
    end
    fprintf('Epochs:');
    disp(k) %Track number of epochs
   % [images,y] = shuffle_gpu(images,y); %Shuffles order of the images for next epoch
end
toc
disp('Training done!')
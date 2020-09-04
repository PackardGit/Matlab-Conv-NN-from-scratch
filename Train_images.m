clear all;
nr_img=60000;
[imgs_l labels] = readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte', nr_img, 0);
y = zeros(10,nr_img); %Correct outputs vector
for i = 1:nr_img
    y(labels(i)+1,i) = 1;
end
for i=1:nr_img
images(:,i)=reshape(imgs_l(:,:,i),[28*28,1]);
end
 %Input vectors
hn1 = 80; %Number of neurons in the first hidden layer
hn2 = 60; %Number of neurons in the second hidden layer
%Initializing weights and biases
w12 = randn(hn1,784)*sqrt(2/784);
w23 = randn(hn2,hn1)*sqrt(2/hn1);
w34 = randn(10,hn2)*sqrt(2/hn2);
b12 = randn(hn1,1);
b23 = randn(hn2,1);
b34 = randn(10,1);
%learning rate
eta = 0.0058;
%Initializing errors and gradients
error4 = zeros(10,1);
error3 = zeros(hn2,1);
error2 = zeros(hn1,1);
errortot4 = zeros(10,1);
errortot3 = zeros(hn2,1);
errortot2 = zeros(hn1,1);
grad4 = zeros(10,1);
grad3 = zeros(hn2,1);
grad2 = zeros(hn1,1);
epochs = 1;
m = 10; %Minibatch size
tic
for k = 1:epochs %Outer epoch loop
    
    batches = 1;
    
    for j = 1:nr_img/m
        error4 = zeros(10,1);
        error3 = zeros(hn2,1);
        error2 = zeros(hn1,1);
        errortot4 = zeros(10,1);
        errortot3 = zeros(hn2,1);
        errortot2 = zeros(hn1,1);
        grad4 = zeros(10,1);
        grad3 = zeros(hn2,1);
        grad2 = zeros(hn1,1);
       
        for i = batches:batches+m-1 %Loop over each minibatch
    
             %Feed forward
             a1 = images(:,i);
             z2 = w12*a1 + b12;
             a2 = elu(z2);
             z3 = w23*a2 + b23;
             a3 = elu(z3);
             z4 = w34*a3 + b34;
             a4 = linear1(z4); %Output vector
    
             %backpropagation
              error4 = (a4-y(:,i));%.*softmaxp(z4);
              error3 = (w34'*error4).*elup(z3);
              error2 = (w23'*error3).*elup(z2);
    
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
    [images,y] = shuffle(images,y); %Shuffles order of the images for next epoch
end
toc
disp('Training done!')
%Saves the parameters
save('wfour.mat','w34');
save('wthree.mat','w23');
save('wtwo.mat','w12');
save('bfour.mat','b34');
save('bthree.mat','b23');
save('btwo.mat','b12');
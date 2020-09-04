
clear all;
nr_img=10000;
[imgs_l labels] = readMNIST('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte', nr_img, 0);
y = zeros(10,nr_img); %Correct outputs vector
for i = 1:nr_img
    y(labels(i)+1,i) = 1;
end
im_s=max(size(imgs_l(:,:,1)));
%%      net parameters and initialization
%Convultion layer 1
p_nr=6;    %number of kernels in first layer
mn1=5;   %size of kernels in first layer MNxMN
im_s2=im_s-2*floor(mn1/2);  %size of image after convultion
V1(1:im_s2,1:im_s2,1:p_nr)=0;
C1(1:im_s2,1:im_s2,1:p_nr)=0;
S1(1:im_s2/2,1:im_s2/2,1:p_nr)=0;
%Convultion layer 2
mn2=5;   %size of kernels in second layer MNxMN
q_nr=12;    %number of kernels in second layer
im_s3=im_s2/2-2*floor(mn2/2);   %image size after pooling and second convultion
H2(1:im_s3,1:im_s3,1:q_nr)=0;
V2(1:im_s3,1:im_s3,1:q_nr)=0;
C2(1:im_s3,1:im_s3,1:q_nr)=0;
S2(1:im_s3/2,1:im_s3/2,1:q_nr)=0;

f(1:(im_s3/2)^2*q_nr)=0;
f=f';

%Fully Connected layer

nw=10; %neurons in fully conection layer

V=zeros(nw,1);
y_p=zeros(nw,1);%output

WW = matfile('W.mat');
W = WW.W;
bb = matfile('b.mat');
b = bb.b;
bb1 = matfile('b1.mat');
b1 = bb1.b1;
bb2 = matfile('b2.mat');
b2 = bb2.b2;
KK1 = matfile('K1.mat');
K1 = KK1.K1;
KK2 = matfile('K2.mat');
K2 = KK2.K2;
success = 0;

for i=1:nr_img
            for p=1:p_nr  %Convultion Layer C1
            V1(:,:,p) = conv2(imgs_l(:,:,i),K1(:,:,1,p),'valid')+b1(:,:,p);
            C1(:,:,p)=activate_function1(V1(:,:,p));
            %Average pooling
            S1(:,:,p) = Average_pooling(C1(:,:,p));
            end
            H2(1:im_s3,1:im_s3,1:q_nr)=0;
            for q=1:q_nr  %Convultion Layer C2
            for p=1:p_nr 
            H2(:,:,q)= H2(:,:,q)+conv2(S1(:,:,p),K2(:,:,p,q),'valid');
            end
            end
            for q=1:q_nr
            V2(:,:,q)=H2(:,:,q)+b2(:,:,q);
            C2(:,:,q)=activate_function1(V2(:,:,q));
            %Average pooling
            S2(:,:,q) = Average_pooling(C2(:,:,q));
            %vectorization
            f(q*(im_s3/2)^2-(im_s3/2)^2+1:q*(im_s3/2)^2)=reshape( S2(:,:,q),[(im_s3/2)^2,1]);
            end
            
            %Fully Connection layer FC
            V=W*f+b;
            y_p = activate_function_vector1(V);
            
            big = 0;
            num = 0;
            for k = 1:10
             if y_p(k) > big
               num = k-1;
             big = y_p(k);
             end
            end
            if labels(i) == num
             success = success + 1;
            end
            
end
fprintf('Accuracy: ');
fprintf('%f',success/nr_img*100);
disp(' %');
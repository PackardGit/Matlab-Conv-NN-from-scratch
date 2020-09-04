%% info
% structure is as it seems:
%    ____________
%   [           ] -----(*)k1(1,1)-------+b1(1,1)------f(v)------
%   [gray image ]       5x5
%   [   28x28   ] -----(*)k1(1,2)-------+b1(1,2)------f(v)-------
%   [           ]       5x5
%   ------------- -----(*)k1(1,6)-------+b1(1,6)------f(v)-------
%                       5x5              
%     image          kernels 5x5     bias 28x28   activation
%
%       input    |               First convultion layer          |
%
%cd:
%-Loss Function 1/2*sum((Y-Yp)^2)  
%
clear all;
%%
%image gathering
nr_img=60000;
[imgs_l labels] = readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte', nr_img, 0);
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
DV1=V1;
C1(1:im_s2,1:im_s2,1:p_nr)=0;
DC1=C1;
DC1_f=DC1;
S1(1:im_s2/2,1:im_s2/2,1:p_nr)=0;
DS1=S1;
K1 = randn(mn1,mn1,1,p_nr)*sqrt(2/(mn1^3*p_nr*3));    %making kernel 1
DK1(1:mn1,1:mn1,1,1:p_nr)=0;
b1(1:im_s2,1:im_s2,p_nr)=0;
Db1(1:im_s2,1:im_s2,p_nr)=0;
%Convultion layer 2
mn2=5;   %size of kernels in second layer MNxMN
q_nr=12;    %number of kernels in second layer
im_s3=im_s2/2-2*floor(mn2/2);   %image size after pooling and second convultion
H2(1:im_s3,1:im_s3,1:q_nr)=0;
DH2=H2;
V2(1:im_s3,1:im_s3,1:q_nr)=0;
DV2=V2;
C2(1:im_s3,1:im_s3,1:q_nr)=0;
DC2=C2;
DC2_f=DC2;
S2(1:im_s3/2,1:im_s3/2,1:q_nr)=0;
DS2=S2;
f(1:(im_s3/2)^2*q_nr)=0;
f=f';
K2 = randn(mn2,mn2,p_nr,q_nr)*sqrt(2/mn2^4*q_nr*p_nr*3);    %making kernel 2
DK2(1:mn2,1:mn2,1:p_nr,1:q_nr)=0;
b2(1:im_s3,1:im_s3,q_nr)=0;
Db2(1:im_s3,1:im_s3,q_nr)=0;
%Fully Connected layer
%input: f(588x1)
nw=10; %neurons in fully conection layer
W=randn(nw,(im_s3/2)^2*q_nr)*sqrt(2/im_s3/2^4*q_nr*3);
b=zeros(nw,1);
V=zeros(nw,1);
y_p=zeros(nw,1);%output

%learning parameters
n=0.0030;    %learning rate
epochs = 30;
m = 10; %Minibatch size
tic

%%          Training
tic
     nr_img=60000;
     
     
for e=1:epochs
for  i =1:  nr_img
            %% forward propagation

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


            %% Backpropagation
            f_a_d = y_p.*(1-y_p); %this is external derivative of activate function
            Dy = (y_p'-y(:,i)).*f_a_d';
            DW=Dy*f';   %*f becouse internal derivative
            Db=Dy;
            Df= W'*Dy;  %*W becouse of internal derivative
            for k=1:q_nr
            Df_temp(1:(im_s3/2)^2,k)= Df(k*(im_s3/2)^2-(im_s3/2)^2+1:k*(im_s3/2)^2);
            DS2(:,:,k)=reshape(Df_temp(:,k),[im_s3/2,im_s3/2,1]);
            DC2(:,:,k)=un_average_pooling(DS2(:,:,k));
            end
            DC2_temp=activate_function1_derivative(C2);
            DC2_f=DC2(:,:,:).* DC2_temp(:,:,:);
              %Convultion Layer C2
            for p=1:p_nr 
            S1_rot180(:,:,p)=rot90(S1(:,:,p),2);
            end
            for q=1:q_nr
            for p=1:p_nr 
            DK2(:,:,p,q)=conv2(S1_rot180(:,:,p),DC2_f(:,:,q),'valid');
            end
            end
            for q=1:q_nr
               Db2(:,:,q)=sum(sum(DC2_f(:,:,q)));
            end
            DS1(:,:,:)=0;
            for p=1:p_nr
                for q=1:q_nr
                    k2_180(:,:,p,q)= rot90(K2(:,:,p,q),2);
                    DS1(:,:,p)=DS1(:,:,p)+conv2(DC2_f(:,:,q),k2_180(:,:,p,q),'full');
                end
                DC1(:,:,p)=un_average_pooling(DS1(:,:,p));
            end
            DC1_temp=activate_function1_derivative(C1);
            DC1_f=DC1(:,:,:).* DC1_temp(:,:,:);
            Im_180=rot90(imgs_l(:,:,i),2);
            for p=1:p_nr
                DK1(:,:,1,p)=conv2(Im_180, DC1_f(:,:,p),'valid');
            end

            for p=1:p_nr
               Db1(:,:,p)=sum(sum(DC1_f(:,:,p)));
            end


        
                    %%  Parameter Update
         K1=K1-n*DK1;
         b1=b1-n*Db1;
         K2=K2-n*DK2;
         b2=b2-n*Db2;
         W=W-n*DW;
         b=b-n*Db;
        
   
end 
    fprintf('Epochs:');
    disp(e) %Track number of epochs
end
toc      
        
        
disp('Training done!')
%Saves the parameters
save('b.mat','b');
save('b1.mat','b1');
save('b2.mat','b2');
save('W.mat','W');
save('K1.mat','K1');
save('K2.mat','K2');   
        
        
        
    
       
        
        
        
        
        
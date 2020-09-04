function [B,v] = shuffle_gpu(A,y)
cols = size(A,2);
P = gpuArray.randperm(cols);
B = A(:,P);
v = y(:,P);
end
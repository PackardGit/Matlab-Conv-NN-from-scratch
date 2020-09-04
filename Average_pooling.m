function f_output = Average_pooling(x)
    size_xy=max(size(x))/2;
    f=zeros(size_xy,size_xy);
for i = 1:size_xy
    for j = 1:size_xy
        f(i,j) = 0.25*(x(2*i,2*j)+x(2*i,2*j-1)+x(2*i-1,2*j)+x(2*i-1,2*j-1));
    end
end
f_output = f;
end
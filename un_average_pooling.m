function f_output = un_average_pooling(x)
    size_xy=2*max(size(x));
    f=zeros(size_xy, size_xy);
for i = 1:size_xy
    for j = 1:size_xy
        f(i,j) = 0.25*x(ceil(i/2),ceil(j/2));
    end
end
f_output = f;
end
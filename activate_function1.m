function f_output = activate_function1(x)
    size_xy=max(size(x));
    f=size(x);
for i = 1:size_xy
    for j = 1:size_xy
        f(i,j) = 1/(1+exp(-x(i,j)));
    end
end
f_output = f;
end
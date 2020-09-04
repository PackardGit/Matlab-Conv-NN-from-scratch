function f_output = activate_function_vector1(x)
    size_x=length(x);
    f(1:length(x))=0;
for i = 1:size_x
        f(i) = 1/(1+exp(-x(i)));
end
f_output = f;
end
function fr = softmax(x)
    f = zeros(length(x),1);
    fp = 0;
    for i = 1:length(x)

        fp = exp(x(i))+fp;
        f(i)= exp(x(i))/fp;
    end
    fr = f;
end
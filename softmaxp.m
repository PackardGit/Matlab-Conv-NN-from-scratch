function fr = softmaxp(x)
    f = zeros(length(x),1);
    fp = 0;
    for i = 1:length(x)

        fp = exp(x(i))+fp;
        f(i)= (exp(x(i))*fp - (exp(x(i)))^2)/(fp^2);
    end
    fr = f;
end